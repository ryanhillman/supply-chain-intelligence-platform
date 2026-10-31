"""
SmokeTest — validates the deployed Vertex AI endpoint with real Bigtable rows.

Flow
----
1. Load feature_metadata.json (vocab maps) from the local artifacts directory.
2. Try to read up to N rows from Bigtable inventory_features table.
   - If Bigtable is unavailable: fall back to synthetic zero-vector instances
     with a clear warning. See note below on expected outputs.
3. For each Bigtable row (read through FeatureClient.scan_all, the same
   decoder the batch scorer uses):
   - Fill missing raw features with safe domain defaults (_fill_defaults).
   - Compute derived features (days_of_supply, effective_supply, etc.) via
     engineer_features() from ml_training.datasets.feature_engineer.
   - Map warehouse_id / sku_id strings to vocab ints.
4. POST to /predict via aiplatform.Endpoint.predict().
5. Assert all shortage_prob_7d values are:
   - finite (not NaN, not inf)
   - in [0.0, 1.0]
6. Print a per-instance results table.

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

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing from ml_training when running from the repo root.
_REPO_ROOT = Path(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml_training.datasets.feature_engineer import (
    NUMERIC_FEATURE_COLS,
    engineer_features,
    fill_defaults,
)
from model_serving.pipeline.feature_client import FeatureClient

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
        Local or GCS path to feature_metadata.json.
        If local, checked first; GCS used as fallback.
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

        self._metadata = self._load_feature_metadata()
        self._wh_vocab : dict[str, int] = self._metadata["warehouse_vocab"]
        self._sku_vocab: dict[str, int] = self._metadata["sku_vocab"]

    def run(
        self,
        n_samples: int = 3,
        instance_id: str | None = None,
        table_id: str = "inventory_features",
    ) -> bool:
        """
        Run the smoke test.

        Parameters
        ----------
        n_samples:
            Number of entities to score. Default 3.
        instance_id:
            Bigtable instance ID (e.g. "scip-feature-store").
            If None, synthetic instances are used.
        table_id:
            Bigtable table ID. Default "inventory_features".

        Returns
        -------
        True if all predictions are finite and in [0, 1].
        False if any prediction is NaN, inf, or out of range.

        Raises
        ------
        Exception on connection or endpoint failure (not caught here —
        let the caller see the full stack trace).
        """
        from google.cloud import aiplatform

        instances, source = self._build_instances(
            n_samples, instance_id, table_id
        )

        logger.info(
            "Calling endpoint with %d instances (source: %s)", len(instances), source
        )
        endpoint  = aiplatform.Endpoint(endpoint_name=self.endpoint_resource_name)
        response  = endpoint.predict(instances=instances)
        preds     = response.predictions

        return self._validate_and_print(instances, preds, source)

    # ── Instance building ──────────────────────────────────────────────────────

    def _build_instances(
        self,
        n_samples: int,
        instance_id: str | None,
        table_id: str,
    ) -> tuple[list[dict], str]:
        if instance_id is not None:
            try:
                instances = self._instances_from_bigtable(
                    instance_id, table_id, n_samples
                )
                if instances:
                    return instances, "bigtable"
                logger.warning("No rows returned from Bigtable — using synthetic instances.")
            except Exception as exc:
                logger.warning(
                    "Bigtable read failed (%s) — falling back to synthetic instances.", exc
                )

        return self._synthetic_instances(n_samples), "synthetic"

    def _instances_from_bigtable(
        self,
        instance_id: str,
        table_id: str,
        n_samples: int,
    ) -> list[dict]:
        """Sample n rows through FeatureClient and turn them into endpoint instances."""
        with FeatureClient(
            project_id=self.project_id,
            instance_id=instance_id,
            table_id=table_id,
            credentials_path=self._credentials_path,
        ) as client:
            rows = client.scan_all(limit=n_samples)

        instances: list[dict] = []
        for fr in rows:
            instance = self._features_to_instance(fr.features, fr.warehouse_id, fr.sku_id)
            if instance is not None:
                instances.append(instance)
                logger.info("  Row %-20s -> wh_vocab=%d  sku_vocab=%d",
                            fr.row_key, instance["warehouse_id"], instance["sku_id"])
        return instances

    def _features_to_instance(
        self,
        raw_features: dict[str, float],
        warehouse_id: str,
        sku_id: str,
    ) -> dict | None:
        """Apply fill_defaults → engineer_features → vocab mapping (raw units)."""
        wh_vocab_id  = self._wh_vocab.get(warehouse_id)
        sku_vocab_id = self._sku_vocab.get(sku_id)

        if wh_vocab_id is None or sku_vocab_id is None:
            logger.warning(
                "Unknown warehouse_id=%s or sku_id=%s — not in training vocab. Skipping.",
                warehouse_id, sku_id,
            )
            return None

        # Build a single-row DataFrame so engineer_features() can run
        row = {**raw_features, "warehouse_id": warehouse_id, "sku_id": sku_id}
        df  = pd.DataFrame([row])

        # Fill missing raw features with domain defaults
        fill_defaults(df)

        # Compute derived features (days_of_supply, effective_supply, etc.)
        df = engineer_features(df)

        # Raw numeric features; the ONNX graph standardizes (ADR-005 Decision 3).
        x_raw = df[NUMERIC_FEATURE_COLS].values.astype(np.float32).squeeze(0).tolist()

        return {
            "numeric_features": x_raw,
            "warehouse_id":     wh_vocab_id,
            "sku_id":           sku_vocab_id,
        }

    def _synthetic_instances(self, n_samples: int) -> list[dict]:
        """
        Generate zero-vector instances as a minimal connectivity check.

        Note: an all-zero raw feature vector (no stock, no sales) is a
        degenerate entity; the smoke test only checks shape and range validity.
        """
        logger.warning(
            "Using synthetic zero-vector instances. "
            "The resulting probabilities are not meaningful. "
            "Provide --bigtable-instance for real data."
        )
        return [
            {
                "numeric_features": [0.0] * len(NUMERIC_FEATURE_COLS),
                "warehouse_id":     i % self._metadata["n_warehouses"],
                "sku_id":           i % self._metadata["n_skus"],
            }
            for i in range(n_samples)
        ]

    # ── Validation + reporting ─────────────────────────────────────────────────

    def _validate_and_print(
        self,
        instances: list[dict],
        predictions: list,
        source: str,
    ) -> bool:
        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  Smoke Test Results  (source: {source})")
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

        if source == "bigtable" and predictions:
            prob_mean = float(np.mean([p.get("shortage_prob_7d", 0) for p in predictions]))
            print(f"\n  mean shortage_prob_7d = {prob_mean:.4f}")
        print(sep)
        return all_pass

    # ── Artifact loading ───────────────────────────────────────────────────────

    def _load_feature_metadata(self) -> dict:
        local = self.artifacts_dir / "feature_metadata.json"
        if local.exists():
            with open(local) as f:
                return json.load(f)
        raise FileNotFoundError(
            f"feature_metadata.json not found at {local}. "
            "Run ml_training/scripts/train.py or download from GCS artifact path."
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _is_valid_prob(value) -> bool:
    """Return True iff value is a finite float in [0.0, 1.0]."""
    try:
        v = float(value)
        return not (np.isnan(v) or np.isinf(v)) and 0.0 <= v <= 1.0
    except (TypeError, ValueError):
        return False
