"""
SilverSnapshotLoader — reads current Silver Parquet from GCS for batch inference.

Maps Silver family__attribute column names to the flat names used in training.
Applies the same engineer_features() transformations so inference features are
computed identically to training.

Silver columns intentionally NOT mapped (leakage or target outputs):
  risk__reorder_urgency      — derived from threshold logic correlated with target
  risk__stockout_risk_score  — target output (null placeholder)
  risk__shortage_prob_7d     — target output (null placeholder)
  lineage__*                 — pipeline metadata
  demand__demand_forecast_*  — reserved null columns
"""
from __future__ import annotations

import logging

import gcsfs
import pandas as pd
import pyarrow.dataset as pads
from pyarrow.fs import FSSpecHandler, PyFileSystem

from ml_training.datasets.feature_engineer import (
    CATEGORICAL_FEATURE_COLS,
    NUMERIC_FEATURE_COLS,
    engineer_features,
)

logger = logging.getLogger(__name__)

_SILVER_COL_MAP: dict[str, str] = {
    "inventory__current_qty":          "current_qty",
    "inventory__reorder_point":        "reorder_point",
    "inventory__safety_stock":         "safety_stock",
    "velocity__sales_7d":              "sales_7d",
    "velocity__sales_30d":             "sales_30d",
    "velocity__avg_daily_sales_7d":    "avg_daily_sales_7d",
    "velocity__avg_daily_sales_30d":   "avg_daily_sales_30d",
    "velocity__velocity_trend":        "velocity_trend",
    "velocity__stddev_daily_7d":       "stddev_daily_7d",
    "supplier__lead_time_days":        "lead_time_days",
    "supplier__lead_time_stddev":      "lead_time_stddev",
    "supplier__supplier_reliability":  "supplier_reliability",
    "supplier__active_po_qty":         "active_po_qty",
    "supplier__in_transit_qty":        "in_transit_qty",
    "demand__demand_signal_count_24h": "demand_signal_count_24h",
}

_IDENTITY_COLS: list[str] = ["warehouse_id", "sku_id", "row_key"]


def load_silver_features(
    bucket: str,
    credentials_path: str | None = None,
) -> pd.DataFrame:
    """
    Read the latest Silver snapshot from GCS and return a feature DataFrame
    ready for inference/predictor.ShortagePredictor.predict().

    When multiple Silver epochs are present, only the most recent snapshot
    per (warehouse_id, sku_id) pair is retained (deduplication on window_end).

    Returns
    -------
    pd.DataFrame
        Columns: warehouse_id, sku_id, row_key, all NUMERIC_FEATURE_COLS,
        all CATEGORICAL_FEATURE_COLS.
    """
    fs = gcsfs.GCSFileSystem(
        token=credentials_path if credentials_path else "google_default"
    )
    silver_path = f"{bucket}/silver"
    logger.info("Reading Silver from gs://%s", silver_path)

    requested = list(_SILVER_COL_MAP.keys()) + _IDENTITY_COLS + ["window_end"]
    try:
        all_paths = fs.glob(f"{silver_path}/**/*.parquet")
        file_paths = [p for p in all_paths if "_temporary" not in p]
        if not file_paths:
            raise FileNotFoundError(f"No committed Parquet files found in Silver at {silver_path}")
        logger.info("Discovered %d Silver files", len(file_paths))
        pa_fs = PyFileSystem(FSSpecHandler(fs))
        dataset = pads.dataset(file_paths, filesystem=pa_fs, format="parquet", partitioning="hive")
        available = {f.name for f in dataset.schema}
        cols = [c for c in requested if c in available]
        df = dataset.to_table(columns=cols).to_pandas()
    except Exception as exc:
        logger.error("Failed to read Silver: %s", exc)
        raise

    logger.info("Silver rows before dedup: %d", len(df))

    # Keep the most recent snapshot per entity
    if "window_end" in df.columns:
        df = (
            df.sort_values("window_end", ascending=False)
            .drop_duplicates(subset=["warehouse_id", "sku_id"])
            .reset_index(drop=True)
        )
        df = df.drop(columns=["window_end"])

    logger.info("Silver rows after dedup: %d", len(df))

    # Map Silver column names → flat training names
    df = df.rename(columns=_SILVER_COL_MAP)

    # Placeholder for sample_date (required by engineer_features, unused at inference)
    df["sample_date"] = pd.Timestamp.now("UTC").date()

    df = engineer_features(df)

    # Return only identity + feature columns; drop sample_date placeholder.
    # Deduplicate: warehouse_id/sku_id appear in both _IDENTITY_COLS and
    # CATEGORICAL_FEATURE_COLS — keep order, remove second occurrence.
    keep = _IDENTITY_COLS + NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS
    cols = list(dict.fromkeys(c for c in keep if c in df.columns))
    return df[cols].copy()
