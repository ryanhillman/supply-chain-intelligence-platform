"""
request_builder.py — converts Bigtable FeatureRow objects into Vertex AI instances.

Pipeline per entity
-------------------
1. Ensure all 15 raw feature columns are present (add NaN for absent ones).
2. Call fill_defaults() to replace NaN with safe domain defaults.
3. Call engineer_features() to compute 6 derived features, yielding all 21
   NUMERIC_FEATURE_COLS expected by the ONNX model.
4. Map warehouse_id / sku_id strings to integer vocab indices. Numeric features
   are sent in raw units — the ONNX graph standardizes them (ADR-005 Decision 3).
5. Compute stockout_risk_score with feature_engineer.normal_stockout_prob —
   the closed-form Normal approximation, not a model output.
6. Return a ScoringRequest containing the endpoint instance dict, the risk
   score and entity metadata.

All 15 raw feature names and their defaults mirror
ml_training/datasets/feature_engineer.fill_defaults().

Required packages:
    numpy>=1.26.0
    pandas>=2.1.0
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from model_serving.pipeline.feature_client import FeatureRow

logger = logging.getLogger(__name__)

# Raw features Bigtable provides (before derived engineering).
# Mirrors ml_training/datasets/feature_engineer.fill_defaults() keys.
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
          {"numeric_features": [21 raw floats],
           "warehouse_id": int,
           "sku_id": int}
    stockout_risk_score:
        Closed-form Normal-approximation P(stockout ≤ horizon) from the raw
        current_qty / avg_daily_sales_7d / stddev_daily_7d — the same formula
        as the continuous_risk_score training column. Published next to the
        endpoint's shortage_prob_7d; not a model output.
    """

    warehouse_id:        str
    sku_id:              str
    row_key:             str
    instance:            dict
    stockout_risk_score: float


class RequestBuilder:
    """
    Transforms FeatureRow objects into endpoint-ready ScoringRequest objects.

    Parameters
    ----------
    artifacts_dir:
        Directory containing feature_metadata.json.
        Default: "ml_training/artifacts"
    """

    def __init__(
        self,
        artifacts_dir: str | Path = "ml_training/artifacts",
    ) -> None:
        self._artifacts_dir = Path(artifacts_dir)

        self._metadata    = self._load_metadata()
        self._wh_vocab:  dict[str, int] = self._metadata["warehouse_vocab"]
        self._sku_vocab: dict[str, int] = self._metadata["sku_vocab"]

        # Import from ml_training — sys.path must include the repo root.
        from ml_training.datasets.feature_engineer import (
            NUMERIC_FEATURE_COLS,
            engineer_features,
            fill_defaults,
            normal_stockout_prob,
        )
        self._NUMERIC_FEATURE_COLS  = NUMERIC_FEATURE_COLS
        self._engineer_features     = engineer_features
        self._fill_defaults         = fill_defaults
        self._normal_stockout_prob  = normal_stockout_prob
        self._horizon_days: int     = int(self._metadata["horizon_days"])

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

        # Raw numeric features in NUMERIC_FEATURE_COLS order. No scaling here:
        # the ONNX graph carries the training mean/scale (ADR-005 Decision 3).
        x_raw = df[self._NUMERIC_FEATURE_COLS].values.astype(np.float32)

        # Closed-form risk score from the raw features (ADR-004 Decision 6).
        risk_scores = self._normal_stockout_prob(
            df["current_qty"].values,
            df["avg_daily_sales_7d"].values,
            df["stddev_daily_7d"].values,
            self._horizon_days,
        )

        # Assemble ScoringRequest objects.
        requests: list[ScoringRequest] = []
        for i, (fr, wh_id, sku_id) in enumerate(zip(valid_rows, wh_vocab_ids, sku_vocab_ids)):
            instance = {
                "numeric_features": x_raw[i].tolist(),
                "warehouse_id":     wh_id,
                "sku_id":           sku_id,
            }
            requests.append(ScoringRequest(
                warehouse_id=fr.warehouse_id,
                sku_id=fr.sku_id,
                row_key=fr.row_key,
                instance=instance,
                stockout_risk_score=float(risk_scores[i]),
            ))

        logger.info("Built %d scoring requests", len(requests))
        return requests

    # ── Artifact loading ───────────────────────────────────────────────────────

    def _load_metadata(self) -> dict:
        path = self._artifacts_dir / "feature_metadata.json"
        if not path.exists():
            raise FileNotFoundError(
                f"feature_metadata.json not found at {path}. "
                "Run ml_training/scripts/train.py or download from GCS."
            )
        with open(path) as f:
            return json.load(f)
