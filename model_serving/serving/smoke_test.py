"""
SmokeTest — validates the Vertex AI endpoint with real Bigtable rows.

Flow
----
1. Read up to N rows from the Bigtable inventory_features table. A failed
   read, or no rows, fails the test.
2. Turn the rows into endpoint instances with RequestBuilder: the same
   defaults, derived features and vocab mapping the batch scorer applies.
3. POST to /predict via aiplatform.Endpoint.predict().
4. Assert all shortage_prob_7d values are:
   - finite (not NaN, not inf)
   - in [0.0, 1.0]
5. Print a per-instance results table.

Pass criterion
--------------
The smoke test PASSES as long as every shortage_prob_7d is a finite float in
[0, 1]. It does not assert meaningful probability values; use evaluate.py and
the drift monitor for that.

Required packages:
    google-cloud-aiplatform>=1.38.0
    google-cloud-bigtable>=2.22.0
    google-cloud-storage>=2.16.0
    numpy>=1.26.0
    pandas>=2.1.0
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from model_serving.pipeline.feature_client import FeatureClient
from model_serving.pipeline.request_builder import RequestBuilder

logger = logging.getLogger(__name__)

class SmokeTest:
    """
    Parameters
    ----------
    project_id:
        GCP project ID.
    region:
        GCP region (e.g. "us-central1").
    endpoint_resource_name:
        Full Vertex AI endpoint resource name.
    artifacts_dir:
        Local path to feature_metadata.json.
    credentials_path:
        Path to SA key JSON. None = application default credentials.
    """

    def __init__(
        self,
        project_id: str,
        region: str,
        endpoint_resource_name: str,
        artifacts_dir: str | Path = "ml_training/artifacts",
        credentials_path: str | None = None,
    ) -> None:
        self.project_id             = project_id
        self.region                 = region
        self.endpoint_resource_name = endpoint_resource_name
        self.artifacts_dir          = Path(artifacts_dir)

        self._credentials_path = credentials_path
        if credentials_path:
            from google.oauth2 import service_account
            self._creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            self._creds = None

        from google.cloud import aiplatform
        init_kwargs: dict = {"project": project_id, "location": region}
        if self._creds:
            init_kwargs["credentials"] = self._creds
        aiplatform.init(**init_kwargs)

    def run(
        self,
        instance_id: str,
        n_samples: int = 3,
        table_id: str = "inventory_features",
    ) -> bool:
        """
        Run the smoke test.

        Parameters
        ----------
        instance_id:
            Bigtable instance ID (e.g. "scip-feature-store").
        n_samples:
            Number of entities to score. Default 3.
        table_id:
            Bigtable table ID. Default "inventory_features".

        Returns
        -------
        True if all predictions are finite and in [0, 1].
        False if any prediction is NaN, inf, or out of range, or if
        Bigtable returned no rows to score.

        Raises
        ------
        Exception on Bigtable or endpoint failure (not caught here —
        let the caller see the full stack trace).
        """
        instances = self._instances_from_bigtable(instance_id, table_id, n_samples)
        if not instances:
            logger.error(
                "No feature rows to score in Bigtable %s/%s — run the Silver job first",
                instance_id, table_id,
            )
            return False

        logger.info(
            "Calling endpoint %s with %d instances",
            self.endpoint_resource_name, len(instances),
        )
        from google.cloud import aiplatform
        endpoint  = aiplatform.Endpoint(endpoint_name=self.endpoint_resource_name)
        response  = endpoint.predict(instances=instances)
        preds     = response.predictions

        return self._validate_and_print(instances, preds)

    # ── Instance building ──────────────────────────────────────────────────────

    def _instances_from_bigtable(
        self,
        instance_id: str,
        table_id: str,
        n_samples: int,
    ) -> list[dict]:
        """
        Sample n rows through FeatureClient and build endpoint instances with
        RequestBuilder, so the smoke test sends exactly what the batch scorer
        sends (absent raw features defaulted, derived features computed,
        vocab mapped; unknown entities skipped with a warning).
        """
        with FeatureClient(
            project_id=self.project_id,
            instance_id=instance_id,
            table_id=table_id,
            credentials_path=self._credentials_path,
        ) as client:
            rows = client.scan_all(limit=n_samples)

        requests = RequestBuilder(artifacts_dir=self.artifacts_dir).build(rows)
        for req in requests:
            logger.info("  Row %-20s -> wh_vocab=%d  sku_vocab=%d",
                        req.row_key, req.instance["warehouse_id"], req.instance["sku_id"])
        return [req.instance for req in requests]

    # ── Validation + reporting ─────────────────────────────────────────────────

    def _validate_and_print(
        self,
        instances: list[dict],
        predictions: list,
    ) -> bool:
        sep = "=" * 72
        print(f"\n{sep}")
        print("  Smoke Test Results")
        print(sep)
        print(f"  {'#':<3} {'WH-vocab':>8} {'SKU-vocab':>9} {'prob_7d':>10}  OK")
        print(f"  {'-'*3} {'-'*8} {'-'*9} {'-'*10}  --")

        all_pass = True
        for i, (inst, pred) in enumerate(zip(instances, predictions, strict=True)):
            prob = pred.get("shortage_prob_7d", float("nan"))

            ok = _is_valid_prob(prob)
            if not ok:
                all_pass = False

            status = "YES" if ok else "NO "
            print(
                f"  {i:<3} {inst['warehouse_id']:>8} {inst['sku_id']:>9} "
                f"{prob:>10.4f}  {status}"
            )

        print()
        if all_pass:
            print("  PASSED — all predictions are finite and in [0.0, 1.0]")
        else:
            print("  FAILED — one or more predictions are NaN, inf, or out of [0, 1]")

        if predictions:
            prob_mean = float(np.mean([p.get("shortage_prob_7d", 0) for p in predictions]))
            print(f"\n  mean shortage_prob_7d = {prob_mean:.4f}")
        print(sep)
        return all_pass


# ── Module-level helpers ──────────────────────────────────────────────────────

def _is_valid_prob(value) -> bool:
    """Return True iff value is a finite float in [0.0, 1.0]."""
    try:
        v = float(value)
        return not (np.isnan(v) or np.isinf(v)) and 0.0 <= v <= 1.0
    except (TypeError, ValueError):
        return False
