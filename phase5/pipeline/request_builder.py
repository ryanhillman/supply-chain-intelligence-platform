"""
request_builder.py — converts Bigtable FeatureRow objects into Vertex AI instances.

Pipeline per entity
-------------------
1. Ensure all 15 raw feature columns are present (add NaN for absent ones).
2. Call fill_defaults() to replace NaN with safe domain defaults.
3. Call engineer_features() to compute 6 derived features, yielding all 21
   NUMERIC_FEATURE_COLS expected by the ONNX model.
4. Apply StandardScaler.transform() — the container does NOT scale; client is responsible.
5. Map warehouse_id / sku_id strings to integer vocab indices.
6. Return a ScoringRequest containing the endpoint instance dict and entity metadata.

All 15 raw feature names and their defaults mirror
phase4/datasets/feature_engineer.fill_defaults().

Required packages:
    numpy>=1.26.0
    pandas>=2.1.0
    scikit-learn>=1.3.0
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from phase5.pipeline.feature_client import FeatureRow

logger = logging.getLogger(__name__)

# Raw features Bigtable provides (before derived engineering).
# Mirrors phase4/datasets/feature_engineer.fill_defaults() keys.
_RAW_FEATURES: tuple[str, ...] = (
    "current_qty",
    "reorder_point",
    "safety_stock",
    "sales_7d",
    "sales_30d",
    "avg_daily_sales_7d",
    "avg_daily_sales_30d",
    "velocity_trend",
    "stddev_daily_7d",
    "lead_time_days",
    "lead_time_stddev",
    "supplier_reliability",
    "active_po_qty",
    "in_transit_qty",
    "demand_signal_count_24h",
)


@dataclass
class ScoringRequest:
    """One entity prepared for endpoint scoring.

    Attributes
    ----------
    warehouse_id:
        Original warehouse string ("W-001").
    sku_id:
        Original SKU string ("SKU-005").
    row_key:
        Bigtable row key ("W-001#SKU-005").
    instance:
        Endpoint-ready dict:
          {"numeric_features": [21 floats],
           "warehouse_id": int,
           "sku_id": int}
    """

    warehouse_id: str
    sku_id:       str
    row_key:      str
    instance:     dict


class RequestBuilder:
    """
    Transforms FeatureRow objects into endpoint-ready ScoringRequest objects.

    Parameters
    ----------
    artifacts_dir:
        Directory containing feature_metadata.json.
        Default: "phase4/artifacts"
    dataset_dir:
        Directory containing preprocessor.pkl.
        Default: "phase4/artifacts/dataset"
    """

    def __init__(
        self,
        artifacts_dir: str | Path = "phase4/artifacts",
        dataset_dir: str | Path = "phase4/artifacts/dataset",
    ) -> None:
        self._artifacts_dir = Path(artifacts_dir)
        self._dataset_dir   = Path(dataset_dir)

        self._metadata    = self._load_metadata()
        self._scaler      = self._load_scaler()
        self._wh_vocab:  dict[str, int] = self._metadata["warehouse_vocab"]
        self._sku_vocab: dict[str, int] = self._metadata["sku_vocab"]

        # Import from phase4 — sys.path must include the repo root.
        from phase4.datasets.feature_engineer import (
            NUMERIC_FEATURE_COLS,
            engineer_features,
            fill_defaults,
        )
        self._NUMERIC_FEATURE_COLS = NUMERIC_FEATURE_COLS
        self._engineer_features    = engineer_features
        self._fill_defaults        = fill_defaults

        logger.info(
            "RequestBuilder ready: %d numeric features, %d warehouses, %d SKUs",
            len(self._NUMERIC_FEATURE_COLS),
            len(self._wh_vocab),
            len(self._sku_vocab),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self, feature_rows: list[FeatureRow]) -> list[ScoringRequest]:
        """
        Convert a list of FeatureRows into endpoint-ready ScoringRequests.

        Entities whose warehouse_id or sku_id are not in the training vocab
        are skipped with a warning (unseen entities cannot be embedded).

        Parameters
        ----------
        feature_rows:
            Output of FeatureClient.scan_all().

        Returns
        -------
        list[ScoringRequest]
            One entry per entity that passed vocab lookup.
            The list preserves the input order (minus skipped entities).
        """
        if not feature_rows:
            logger.warning("build() called with empty feature_rows list")
            return []

        # Filter out entities with unknown vocab entries before building the DataFrame.
        valid_rows:     list[FeatureRow] = []
        wh_vocab_ids:   list[int]        = []
        sku_vocab_ids:  list[int]        = []

        for fr in feature_rows:
            wh  = self._wh_vocab.get(fr.warehouse_id)
            sku = self._sku_vocab.get(fr.sku_id)
            if wh is None or sku is None:
                logger.warning(
                    "Skipping %s: warehouse_id=%r or sku_id=%r not in training vocab",
                    fr.row_key, fr.warehouse_id, fr.sku_id,
                )
                continue
            valid_rows.append(fr)
            wh_vocab_ids.append(wh)
            sku_vocab_ids.append(sku)

        if not valid_rows:
            logger.warning("No valid rows after vocab filtering — returning empty list")
            return []

        logger.info(
            "Building requests: %d/%d rows passed vocab filter",
            len(valid_rows), len(feature_rows),
        )

        # Build DataFrame with all raw feature columns, including absent ones as NaN.
        records = []
        for fr in valid_rows:
            row = {col: fr.features.get(col, np.nan) for col in _RAW_FEATURES}
            row["warehouse_id"] = fr.warehouse_id
            row["sku_id"]       = fr.sku_id
            records.append(row)

        df = pd.DataFrame(records)

        # Fill NaN in raw features with domain defaults (in-place).
        # Any raw feature completely absent from Bigtable for a given entity
        # was already set to NaN above; _fill_defaults replaces NaN with the
        # domain default (e.g. velocity_trend=1.0, supplier_reliability=0.9).
        self._fill_defaults(df)

        # Compute 6 derived features (days_of_supply, effective_supply, etc.).
        df = self._engineer_features(df)

        # Scale all 21 numeric features in one vectorized call.
        x_raw    = df[self._NUMERIC_FEATURE_COLS].values.astype(np.float32)
        x_scaled = self._scaler.transform(x_raw)

        # Assemble ScoringRequest objects.
        requests: list[ScoringRequest] = []
        for i, (fr, wh_id, sku_id) in enumerate(zip(valid_rows, wh_vocab_ids, sku_vocab_ids)):
            instance = {
                "numeric_features": x_scaled[i].tolist(),
                "warehouse_id":     wh_id,
                "sku_id":           sku_id,
            }
            requests.append(ScoringRequest(
                warehouse_id=fr.warehouse_id,
                sku_id=fr.sku_id,
                row_key=fr.row_key,
                instance=instance,
            ))

        logger.info("Built %d scoring requests", len(requests))
        return requests

    # ── Artifact loading ───────────────────────────────────────────────────────

    def _load_metadata(self) -> dict:
        path = self._artifacts_dir / "feature_metadata.json"
        if not path.exists():
            raise FileNotFoundError(
                f"feature_metadata.json not found at {path}. "
                "Run phase4/scripts/train.py or download from GCS."
            )
        with open(path) as f:
            return json.load(f)

    def _load_scaler(self):
        path = self._dataset_dir / "preprocessor.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"preprocessor.pkl not found at {path}. "
                "Run phase4/scripts/build_dataset.py or download from GCS."
            )
        with open(path, "rb") as f:
            return pickle.load(f)
