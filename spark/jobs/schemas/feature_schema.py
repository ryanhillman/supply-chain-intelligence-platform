"""
PySpark schema for the Silver feature layer.

Defines:
  SILVER_FEATURE_SCHEMA — StructType for Silver Parquet output.
  SILVER_PARTITION_COLS — partition columns for the Silver write.

The Bigtable mutation path (spark/utils/bigtable_writer.py) writes directly
from the Silver DataFrame's columns via feature_registry.FEATURE_REGISTRY
— there is no separate Bigtable-specific schema.

Column naming convention: {family}__{attribute}
  Matches the feature_registry.FEATURE_REGISTRY keys exactly.
  Double-underscore chosen because neither family names nor attribute names
  contain __ so splitting on __ is unambiguous.

All numeric features are DoubleType for consistency.
String features (event types, timestamps, version strings) are StringType.
Partition columns (warehouse_id, sku_id, year, month, day) appended last.
"""
from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
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
    StructField("demand__demand_signal_count_24h", DoubleType(),  nullable=True),
    StructField("demand__last_signal_type",        StringType(),  nullable=True),
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
    # 0=OK, 1=WATCH, 2=REORDER, 3=CRITICAL (derived from inventory__days_of_supply)
    StructField("risk__reorder_urgency",     IntegerType(), nullable=False),
]

# ── lineage family ────────────────────────────────────────────────────────────

_LINEAGE_FIELDS: list[StructField] = [
    StructField("lineage__last_silver_write",  StringType(), nullable=False),  # ISO-8601
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
22 feature columns + 4 identity/window columns + 3 partition columns = 29 total.
"""

# Columns used as partition keys when writing Silver Parquet.
SILVER_PARTITION_COLS: list[str] = ["warehouse_id", "year", "month", "day"]


def conform_to_silver_schema(df: DataFrame) -> DataFrame:
    """
    Exactly SILVER_FEATURE_SCHEMA's columns, in order, cast to its types.

    Raises ValueError if a schema column is missing from `df` or a
    non-nullable column holds nulls, so a feature change that breaks the
    Silver contract fails the job instead of writing a different layout.
    """
    fields  = SILVER_FEATURE_SCHEMA.fields
    missing = [f.name for f in fields if f.name not in df.columns]
    if missing:
        raise ValueError(f"Silver output is missing schema columns: {missing}")

    out = df.select([F.col(f.name).cast(f.dataType).alias(f.name) for f in fields])
    required = [f.name for f in fields if not f.nullable]
    nulls = out.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in required]).first().asDict()
    bad = {c: n for c, n in nulls.items() if n}
    if bad:
        raise ValueError(f"Null values in non-nullable Silver columns: {bad}")
    return out
