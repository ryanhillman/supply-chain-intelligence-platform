from __future__ import annotations

"""
PySpark schema for suppliers.events Bronze topic.
Mirrors supplier_event.avsc (Phase 1.5).

13 fields. affected_sku_ids is an array of SKU IDs impacted by this
supplier event (e.g. a lead time change for multiple SKUs at once).
"""
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    StringType,
    StructField,
    StructType,
)

SUPPLIER_EVENT_SCHEMA = StructType([
    StructField("event_id",             StringType(),              nullable=False),
    StructField("schema_version",       StringType(),              nullable=False),
    StructField("event_type",           StringType(),              nullable=False),
    StructField("supplier_id",          StringType(),              nullable=False),
    StructField("warehouse_id",         StringType(),              nullable=False),
    StructField("lead_time_days",       FloatType(),               nullable=True),
    StructField("on_time_delivery_rate",FloatType(),               nullable=True),
    StructField("reliability_score",    FloatType(),               nullable=True),
    StructField("affected_sku_ids",     ArrayType(StringType()),   nullable=True),
    StructField("event_description",    StringType(),              nullable=True),
    StructField("timestamp",            StringType(),              nullable=False),
    StructField("producer_id",          StringType(),              nullable=True),
    StructField("correlation_id",       StringType(),              nullable=True),
])
