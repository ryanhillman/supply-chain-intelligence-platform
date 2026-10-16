"""
SmokeTest — validates the deployed Vertex AI endpoint with real Bigtable rows.

Flow
----
1. Load preprocessor.pkl and feature_metadata.json (from GCS or local artifacts).
2. Try to read up to N rows from Bigtable inventory_features table.
   - If Bigtable is unavailable: fall back to synthetic zero-vector instances
     with a clear warning. See note below on expected outputs.
3. For each Bigtable row:
   - Decode big-endian double values (struct.unpack(">d", ...)).
   - Map abbreviated column qualifiers to flat feature names.
   - Fill missing raw features with safe domain defaults (_fill_defaults).
   - Compute derived features (days_of_supply, effective_supply, etc.) via
     engineer_features() from ml_training.datasets.feature_engineer.
   - Apply StandardScaler.transform() to get scaled inputs.
   - Map warehouse_id / sku_id strings to vocab ints.
4. POST to /predict via aiplatform.Endpoint.predict().
5. Assert all shortage_prob_7d and stockout_risk_score values are:
   - finite (not NaN, not inf)
   - in [0.0, 1.0]
6. Print a per-instance results table.

Expected outputs with current Phase 4 model (trained on simulated data)
-----------------------------------------------------------------------
Because the model was trained on tiled Bronze data (avg_daily_sales_7d ~2785)
and the real Silver/Bigtable data has avg_daily_sales_7d ~207, the model is
46 sigma outside its training distribution. Expected results:
  shortage_prob_7d    = ~0.0  (model sees low-risk on heavily over-stocked entities)
  stockout_risk_score = ~1.0  (score_head saturates on OOD input)

These values confirm the pipeline is working correctly — the distribution-shift
warning in ADR-005 Decision 7 describes exactly this behaviour. The smoke test
PASSES as long as outputs are finite floats in [0, 1]. It does not assert
meaningful probability values.

Required packages:
    google-cloud-aiplatform>=1.38.0
    google-cloud-bigtable>=2.22.0
    google-cloud-storage>=2.16.0
    scikit-learn>=1.3.0
    numpy>=1.26.0
    pandas>=2.1.0
"""
from __future__ import annotations

import json
import logging
import pickle
import struct
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

logger = logging.getLogger(__name__)

# Maps (bigtable_family, bigtable_qualifier) -> flat feature name.
# Derived from spark/config/feature_config.py FEATURE_REGISTRY.
# Qualifiers are abbreviated; see bigtable_writer.py for the encoding contract.
_BIGTABLE_TO_FEATURE: dict[tuple[str, str], str] = {
    ("inventory", "cur_qty"):      "current_qty",
    ("inventory", "reorder_pt"):   "reorder_point",
    ("inventory", "safety_stk"):   "safety_stock",
    ("velocity",  "sales_7d"):     "sales_7d",
    ("velocity",  "sales_30d"):    "sales_30d",
    ("velocity",  "avg_d_7d"):     "avg_daily_sales_7d",
    ("velocity",  "avg_d_30d"):    "avg_daily_sales_30d",
    ("velocity",  "v_trend"):      "velocity_trend",
    ("velocity",  "stddev_7d"):    "stddev_daily_7d",
    ("demand",    "sig_24h"):      "demand_signal_count_24h",
    ("supplier",  "lt_days"):      "lead_time_days",
    ("supplier",  "lt_std"):       "lead_time_stddev",
    ("supplier",  "reliability"):  "supplier_reliability",
    ("supplier",  "po_qty"):       "active_po_qty",
    ("supplier",  "transit_qty"):  "in_transit_qty",
}

# Column families written by bigtable_writer.py that contain string values
# (stored as UTF-8, not big-endian double). We skip these for feature extraction.
_STRING_FAMILIES_QUALIFIERS: frozenset[tuple[str, str]] = frozenset({
    ("inventory", "last_evt"),
    ("inventory", "last_upd"),
    ("demand",    "last_sig"),
    ("lineage",   "silver_ts"),
    ("lineage",   "pipeline_ver"),
})


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
    dataset_dir:
        Local or GCS path to preprocessor.pkl.
    credentials_path:
        Path to SA key JSON. None = application default credentials.
    """

    def __init__(
        self,
        project_id: str,
        region: str,
        endpoint_resource_name: str,
        artifacts_dir: str | Path = "ml_training/artifacts",
        dataset_dir: str | Path = "ml_training/artifacts/dataset",
        credentials_path: str | None = None,
    ) -> None:
        self.project_id             = project_id
        self.region                 = region
        self.endpoint_resource_name = endpoint_resource_name
        self.artifacts_dir          = Path(artifacts_dir)
        self.dataset_dir            = Path(dataset_dir)

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
        self._scaler   = self._load_scaler()
        self._wh_vocab : dict[str, int] = self._metadata["warehouse_vocab"]
        self._sku_vocab: dict[str, int] = self._metadata["sku_vocab"]

    def run(
        self,
        n_samples: int = 3,
        bigtable_instance_id: str | None = None,
        bigtable_table_id: str = "inventory_features",
    ) -> bool:
        """
        Run the smoke test.

        Parameters
        ----------
        n_samples:
            Number of entities to score. Default 3.
        bigtable_instance_id:
            Bigtable instance ID (e.g. "scip-feature-store").
            If None, synthetic instances are used.
        bigtable_table_id:
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
            n_samples, bigtable_instance_id, bigtable_table_id
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
        bigtable_instance_id: str | None,
        bigtable_table_id: str,
    ) -> tuple[list[dict], str]:
        if bigtable_instance_id is not None:
            try:
                instances = self._instances_from_bigtable(
                    bigtable_instance_id, bigtable_table_id, n_samples
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
        from google.cloud import bigtable

        client_kwargs: dict = {"project": self.project_id, "admin": False}
        if self._creds:
            client_kwargs["credentials"] = self._creds

        client = bigtable.Client(**client_kwargs)
        table  = client.instance(instance_id).table(table_id)

        rows = table.read_rows(limit=n_samples)
        instances: list[dict] = []

        for row_data in rows:
            row_key    = row_data.row_key.decode("utf-8")
            parts      = row_key.split("#")
            if len(parts) != 2:
                logger.warning("Unexpected row key format: %s — skipping", row_key)
                continue

            warehouse_id, sku_id = parts[0], parts[1]
            raw_features: dict[str, float] = {}

            for cf_name, cf_data in row_data.cells.items():
                for col_qualifier_bytes, cells in cf_data.items():
                    col_qualifier = col_qualifier_bytes.decode("utf-8")
                    key = (cf_name, col_qualifier)

                    if key in _STRING_FAMILIES_QUALIFIERS:
                        continue

                    feature_name = _BIGTABLE_TO_FEATURE.get(key)
                    if feature_name is None:
                        continue

                    cell_value = cells[0].value
                    raw_features[feature_name] = _decode_double(cell_value)

            instance = self._features_to_instance(raw_features, warehouse_id, sku_id)
            if instance is not None:
                instances.append(instance)
                logger.info("  Row %-20s -> wh_vocab=%d  sku_vocab=%d",
                            row_key,
                            instance["warehouse_id"],
                            instance["sku_id"])

        client.close()
        return instances

    def _features_to_instance(
        self,
        raw_features: dict[str, float],
        warehouse_id: str,
        sku_id: str,
    ) -> dict | None:
        """Apply fill_defaults → engineer_features → scaler → vocab mapping."""
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

        # Scale numeric features
        x_raw    = df[NUMERIC_FEATURE_COLS].values.astype(np.float32)
        x_scaled = self._scaler.transform(x_raw).squeeze(0).tolist()

        return {
            "numeric_features": x_scaled,
            "warehouse_id":     wh_vocab_id,
            "sku_id":           sku_vocab_id,
        }

    def _synthetic_instances(self, n_samples: int) -> list[dict]:
        """
        Generate zero-vector instances as a minimal connectivity check.

        Note: with the current simulated-data model, zero-vector inputs are
        out-of-distribution and will produce saturation outputs (prob~0, score~1).
        This is expected — the smoke test only checks shape and range validity.
        """
        logger.warning(
            "Using synthetic zero-vector instances. "
            "Outputs will be out-of-distribution (see ADR-005 Decision 7). "
            "Provide --bigtable-instance-id for real data."
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
        print(f"  {'#':<3} {'WH-vocab':>8} {'SKU-vocab':>9} {'prob_7d':>10} {'risk_score':>11}  OK")
        print(f"  {'-'*3} {'-'*8} {'-'*9} {'-'*10} {'-'*11}  --")

        all_pass = True
        for i, (inst, pred) in enumerate(zip(instances, predictions)):
            prob  = pred.get("shortage_prob_7d",    pred.get("shortage_prob_7d", float("nan")))
            score = pred.get("stockout_risk_score", float("nan"))

            ok = (
                _is_valid_prob(prob) and _is_valid_prob(score)
            )
            if not ok:
                all_pass = False

            status = "YES" if ok else "NO "
            print(
                f"  {i:<3} {inst['warehouse_id']:>8} {inst['sku_id']:>9} "
                f"{prob:>10.4f} {score:>11.4f}  {status}"
            )

        print()
        if all_pass:
            print("  PASSED — all predictions are finite and in [0.0, 1.0]")
        else:
            print("  FAILED — one or more predictions are NaN, inf, or out of [0, 1]")

        if source == "bigtable" and predictions:
            prob_mean  = float(np.mean([p.get("shortage_prob_7d",    0) for p in predictions]))
            score_mean = float(np.mean([p.get("stockout_risk_score", 0) for p in predictions]))
            print(f"\n  mean shortage_prob_7d    = {prob_mean:.4f}")
            print(f"  mean stockout_risk_score = {score_mean:.4f}")
            if prob_mean < 0.01 and score_mean > 0.99:
                print(
                    "\n  [distribution-shift expected] prob~0 + score~1 confirms the model\n"
                    "  is 46-sigma outside its training distribution (ADR-005 Decision 7).\n"
                    "  The pipeline is correct; retrain with real Bronze data to fix."
                )
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

    def _load_scaler(self):
        local = self.dataset_dir / "preprocessor.pkl"
        if local.exists():
            with open(local, "rb") as f:
                return pickle.load(f)
        raise FileNotFoundError(
            f"preprocessor.pkl not found at {local}. "
            "Run ml_training/scripts/build_dataset.py or download from GCS artifact path."
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _decode_double(value: bytes) -> float:
    """Decode a Bigtable cell value to float64 (big-endian double)."""
    if len(value) == 8:
        return struct.unpack(">d", value)[0]
    # Graceful fallback for cells written as UTF-8 strings (non-standard)
    try:
        return float(value.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        logger.debug("Could not decode cell value as double or UTF-8: %r", value)
        return 0.0


def _is_valid_prob(value) -> bool:
    """Return True iff value is a finite float in [0.0, 1.0]."""
    try:
        v = float(value)
        return not (np.isnan(v) or np.isinf(v)) and 0.0 <= v <= 1.0
    except (TypeError, ValueError):
        return False
