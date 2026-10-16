"""
PySpark schema for the Silver feature layer.

Defines:
  SILVER_FEATURE_SCHEMA — StructType for Silver Parquet output.
  SILVER_PARTITION_COLS — partition columns for the Silver write.
  SILVER_DEDUP_COLS     — dedup key for the Silver write.

The Bigtable mutation path (spark/utils/bigtable_writer.py) writes directly
from the Silver DataFrame's columns via FEATURE_REGISTRY in feature_config.py
— there is no separate Bigtable-specific schema.

Column naming convention: {family}__{attribute}
  Matches FEATURE_REGISTRY keys in feature_config.py exactly.
  Double-underscore chosen because neither family names nor attribute names
  contain __ so splitting on __ is unambiguous.

All numeric features are DoubleType for consistency.
String features (event types, timestamps, version strings) are StringType.
Partition columns (warehouse_id, sku_id, year, month, day) appended last.
"""
from __future__ import annotations

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ── Identity / partition columns ──────────────────────────────────────────────

_IDENTITY_FIELDS: list[StructField] = [
    StructField("warehouse_id",   StringType(),    nullable=False),
    StructField("sku_id",         StringType(),    nullable=False),
    # Bigtable row key: warehouse_id + "#" + sku_id (pre-computed for dual-write)
    StructField("row_key",        StringType(),    nullable=False),
    # Window close time — the "as-of" timestamp for this feature snapshot
    StructField("window_end",     TimestampType(), nullable=False),
]

# ── inventory family ──────────────────────────────────────────────────────────

_INVENTORY_FIELDS: list[StructField] = [
    StructField("inventory__current_qty",    DoubleType(),  nullable=True),
    StructField("inventory__reorder_point",  DoubleType(),  nullable=True),
    StructField("inventory__safety_stock",   DoubleType(),  nullable=True),
    StructField("inventory__days_of_supply", DoubleType(),  nullable=True),
    StructField("inventory__last_event_type",StringType(),  nullable=True),
    StructField("inventory__last_updated",   StringType(),  nullable=True),  # ISO-8601
]

# ── velocity family ───────────────────────────────────────────────────────────

_VELOCITY_FIELDS: list[StructField] = [
    StructField("velocity__sales_7d",            DoubleType(), nullable=True),
    StructField("velocity__sales_30d",           DoubleType(), nullable=True),
    StructField("velocity__avg_daily_sales_7d",  DoubleType(), nullable=True),
    StructField("velocity__avg_daily_sales_30d", DoubleType(), nullable=True),
    # velocity_trend = avg_daily_7d / avg_daily_30d; null when 30d avg is 0
    StructField("velocity__velocity_trend",      DoubleType(), nullable=True),
    StructField("velocity__stddev_daily_7d",     DoubleType(), nullable=True),
]

# ── demand family ─────────────────────────────────────────────────────────────

_DEMAND_FIELDS: list[StructField] = [
    # Reserved demand forecast columns; not populated by any current pipeline stage
    StructField("demand__demand_forecast_7d",      DoubleType(),  nullable=True),
    StructField("demand__demand_forecast_30d",     DoubleType(),  nullable=True),
    StructField("demand__demand_signal_count_24h", DoubleType(),  nullable=True),
    StructField("demand__last_signal_type",        StringType(),  nullable=True),
    # demand_multiplier = forecast_7d / (avg_daily_7d * 7); null when forecast absent
    StructField("demand__demand_multiplier",       DoubleType(),  nullable=True),
]

# ── supplier family ───────────────────────────────────────────────────────────

_SUPPLIER_FIELDS: list[StructField] = [
    StructField("supplier__lead_time_days",      DoubleType(), nullable=True),
    StructField("supplier__lead_time_stddev",    DoubleType(), nullable=True),
    StructField("supplier__supplier_reliability",DoubleType(), nullable=True),
    StructField("supplier__active_po_qty",       DoubleType(), nullable=True),
    StructField("supplier__in_transit_qty",      DoubleType(), nullable=True),
]

# ── risk family ───────────────────────────────────────────────────────────────

_RISK_FIELDS: list[StructField] = [
    # stockout_risk_score and shortage_prob_7d are reserved null columns
    StructField("risk__stockout_risk_score", DoubleType(),  nullable=True),
    StructField("risk__shortage_prob_7d",    DoubleType(),  nullable=True),
    # 0=OK, 1=WATCH, 2=REORDER, 3=CRITICAL (derived from inventory__days_of_supply)
    StructField("risk__reorder_urgency",     IntegerType(), nullable=False),
]

# ── lineage family ────────────────────────────────────────────────────────────

_LINEAGE_FIELDS: list[StructField] = [
    StructField("lineage__last_bronze_epoch",  LongType(),   nullable=False),
    StructField("lineage__last_silver_write",  StringType(), nullable=False),  # ISO-8601
    StructField("lineage__source_event_count", LongType(),   nullable=False),
    StructField("lineage__pipeline_version",   StringType(), nullable=False),
]

# ── Partition columns (appended last for Parquet partition pruning) ────────────

_PARTITION_FIELDS: list[StructField] = [
    StructField("year",  IntegerType(), nullable=False),
    StructField("month", IntegerType(), nullable=False),
    StructField("day",   IntegerType(), nullable=False),
]

# ── Assembled schemas ─────────────────────────────────────────────────────────

SILVER_FEATURE_SCHEMA = StructType(
    _IDENTITY_FIELDS
    + _INVENTORY_FIELDS
    + _VELOCITY_FIELDS
    + _DEMAND_FIELDS
    + _SUPPLIER_FIELDS
    + _RISK_FIELDS
    + _LINEAGE_FIELDS
    + _PARTITION_FIELDS
)
"""
Full schema for Silver Parquet output.
30 feature columns + 4 identity/window columns + 3 partition columns = 37 total.
"""

# Columns used as partition keys when writing Silver Parquet.
SILVER_PARTITION_COLS: list[str] = ["warehouse_id", "year", "month", "day"]

# Columns that uniquely identify a feature snapshot row.
# Used for deduplication via dropDuplicates() before Silver write.
SILVER_DEDUP_COLS: list[str] = ["warehouse_id", "sku_id", "window_end"]
