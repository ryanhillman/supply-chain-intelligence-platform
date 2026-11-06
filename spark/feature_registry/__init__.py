"""
The Bigtable feature-store layout: one definition for everything that writes
or reads it (the Spark writer, FeatureClient, RequestBuilder, the feature
dashboard and the training code).

  FEATURE_REGISTRY      family -> {feature: qualifier} for the 22 Silver columns
  STRING_FEATURES       Silver columns stored as UTF-8 rather than a double
  RAW_FEATURE_COLS      the 15 stored features the model takes, in model order
  RAW_FEATURE_CELLS     (family, qualifier) -> feature, for RAW_FEATURE_COLS
  RAW_FEATURE_FAMILIES  the families that hold RAW_FEATURE_COLS
  encode_cell, decode_double   the cell encoding

It also holds HORIZON_DAYS, the label horizon that the Spark training-set job
and the training code share.

This package lives under spark/ because the Spark container holds only that
directory; pyproject.toml installs it as `feature_registry` for the rest of
the project. Standard library only.
"""
from __future__ import annotations

import logging
import struct

logger = logging.getLogger(__name__)

# {family: {feature: qualifier}}. A feature's Silver column is
# "{family}__{feature}"; the qualifier is its short column name in Bigtable.
FEATURE_REGISTRY: dict[str, dict[str, str]] = {
    "inventory": {
        "current_qty":       "cur_qty",
        "reorder_point":     "reorder_pt",
        "safety_stock":      "safety_stk",
        "days_of_supply":    "dos",
        "last_event_type":   "last_evt",    # string
        "last_updated":      "last_upd",    # string (ISO-8601)
    },
    "velocity": {
        "sales_7d":             "sales_7d",
        "sales_30d":            "sales_30d",
        "avg_daily_sales_7d":   "avg_d_7d",
        "avg_daily_sales_30d":  "avg_d_30d",
        "velocity_trend":       "v_trend",   # ratio: avg_d_7d / avg_d_30d
        "stddev_daily_7d":      "stddev_7d",
    },
    "demand": {
        "demand_signal_count_24h":  "sig_24h",
        "last_signal_type":         "last_sig",  # string
    },
    "supplier": {
        "lead_time_days":           "lt_days",
        "lead_time_stddev":         "lt_std",
        "supplier_reliability":     "reliability",  # 0.0–1.0
        "active_po_qty":            "po_qty",
        "in_transit_qty":           "transit_qty",
    },
    "risk": {
        "reorder_urgency":          "urgency",      # 0|1|2|3
    },
    "lineage": {
        "last_silver_write":    "silver_ts",      # string ISO-8601
        "pipeline_version":     "pipeline_ver",   # string
    },
}

# Silver columns stored as UTF-8 bytes; every other column is a big-endian double.
STRING_FEATURES: frozenset[str] = frozenset({
    "inventory__last_event_type",
    "inventory__last_updated",
    "demand__last_signal_type",
    "lineage__last_silver_write",
    "lineage__pipeline_version",
})

# The stored features the model takes as input, in the order the model reads
# them. NUMERIC_FEATURE_COLS (ml_training/datasets/feature_engineer.py) is
# this list followed by the six derived features.
RAW_FEATURE_COLS: tuple[str, ...] = (
    # Inventory state
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
)

RAW_FEATURE_CELLS: dict[tuple[str, str], str] = {
    (family, qualifier): feature
    for family, features in FEATURE_REGISTRY.items()
    for feature, qualifier in features.items()
    if feature in RAW_FEATURE_COLS
}
RAW_FEATURE_FAMILIES: frozenset[str] = frozenset(family for family, _ in RAW_FEATURE_CELLS)

# The stockout label looks this many days ahead of the feature snapshot.
HORIZON_DAYS: int = 7


def encode_cell(column: str, value) -> bytes | None:
    """
    A Silver column's value as Bigtable cell bytes: UTF-8 for STRING_FEATURES,
    a big-endian IEEE 754 double (8 bytes) otherwise. None for a null, or for
    a value that cannot be encoded; the writer deletes that cell.
    """
    if value is None:
        return None
    if column in STRING_FEATURES:
        return str(value).encode("utf-8")
    try:
        return struct.pack(">d", float(value))
    except (TypeError, ValueError) as exc:
        logger.warning("Cannot encode %s=%r as double: %s", column, value, exc)
        return None


def decode_double(value: bytes) -> float | None:
    """A numeric cell, an 8-byte big-endian double, as a float. None for anything else."""
    return struct.unpack(">d", value)[0] if len(value) == 8 else None
