"""
feature_engineer.py — derived features and leakage controls.

Input:  labeled DataFrame from label_generator.generate_labels()
Output: clean feature DataFrame ready for DatasetBuilder

Derived features added (all deterministic formulas, no data-distribution fitting):
  days_of_supply       = current_qty / max(avg_daily_sales_7d, EPS)
  effective_supply     = current_qty + in_transit_qty + active_po_qty
  buffer_ratio         = current_qty / max(safety_stock, 1.0)
  demand_cv            = stddev_daily_7d / max(avg_daily_sales_7d, EPS)
  supply_coverage_days = effective_supply / max(avg_daily_sales_7d, EPS)
  replenishment_gap    = lead_time_days - days_of_supply

Leakage columns explicitly absent from NUMERIC_FEATURE_COLS:
  risk__reorder_urgency      — derived from the same threshold logic as the label
  All lineage__* columns     — pipeline metadata with no predictive value
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

EPS: float = 1e-6

# Canonical ordered list of numeric features fed to the model.
# Order is fixed here and must match the column order used when fitting
# the StandardScaler in DatasetBuilder.
NUMERIC_FEATURE_COLS: list[str] = [
    # Raw inventory state
    "current_qty",
    "reorder_point",
    "safety_stock",
    # Velocity
    "sales_7d",
    "sales_30d",
    "avg_daily_sales_7d",
    "avg_daily_sales_30d",
    "velocity_trend",
    "stddev_daily_7d",
    # Supplier
    "lead_time_days",
    "lead_time_stddev",
    "supplier_reliability",
    # Pipeline
    "active_po_qty",
    "in_transit_qty",
    # Demand signals
    "demand_signal_count_24h",
    # Derived (computed below)
    "days_of_supply",
    "effective_supply",
    "buffer_ratio",
    "demand_cv",
    "supply_coverage_days",
    "replenishment_gap",
]

CATEGORICAL_FEATURE_COLS: list[str] = ["warehouse_id", "sku_id"]

# Label and auxiliary target columns (not features)
TARGET_COL:     str = "stockout_label_7d"
RISK_SCORE_COL: str = "continuous_risk_score"
BASELINE_COL:   str = "baseline_pred"

# Passthrough identity columns kept in dataset rows for split logic and reporting
ID_COLS: list[str] = ["warehouse_id", "sku_id", "sample_date"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features, fill missing values, clip outliers.

    All transformations use deterministic formulas with no dependency on
    the training set distribution. Safe to apply identically to train/val/test
    and to inference data from Silver Parquet.
    """
    df = df.copy()

    # ── NaN fill before derivation ────────────────────────────────────────────
    fill_defaults(df)

    # ── Derived features ──────────────────────────────────────────────────────
    avg_d7 = df["avg_daily_sales_7d"].clip(lower=EPS)

    df["days_of_supply"]       = (df["current_qty"] / avg_d7).clip(upper=365.0)
    df["effective_supply"]     = df["current_qty"] + df["in_transit_qty"] + df["active_po_qty"]
    df["buffer_ratio"]         = (df["current_qty"] / df["safety_stock"].clip(lower=1.0)).clip(upper=10.0)
    df["demand_cv"]            = (df["stddev_daily_7d"] / avg_d7).clip(upper=5.0)
    df["supply_coverage_days"] = (df["effective_supply"] / avg_d7).clip(upper=365.0)
    df["replenishment_gap"]    = (df["lead_time_days"] - df["days_of_supply"]).clip(-365.0, 365.0)

    # ── Validation ────────────────────────────────────────────────────────────
    missing = [c for c in NUMERIC_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after engineering: {missing}")

    null_counts = df[NUMERIC_FEATURE_COLS].isna().sum()
    if null_counts.any():
        logger.warning("Remaining NaN in numeric features:\n%s", null_counts[null_counts > 0])

    logger.info("engineer_features: %d rows, %d numeric features", len(df), len(NUMERIC_FEATURE_COLS))
    return df


def fill_defaults(df: pd.DataFrame) -> None:
    """Fill missing raw features with safe domain defaults (in-place)."""
    defaults: dict[str, float] = {
        "current_qty":            0.0,
        "reorder_point":          0.0,
        "safety_stock":           0.0,
        "sales_7d":               0.0,
        "sales_30d":              0.0,
        "avg_daily_sales_7d":     0.0,
        "avg_daily_sales_30d":    0.0,
        "velocity_trend":         1.0,   # flat demand trend
        "stddev_daily_7d":        0.0,
        "lead_time_days":         14.0,  # 2-week default
        "lead_time_stddev":       0.0,
        "supplier_reliability":   0.9,   # high-reliability default
        "active_po_qty":          0.0,
        "in_transit_qty":         0.0,
        "demand_signal_count_24h": 0.0,
    }
    for col, val in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
