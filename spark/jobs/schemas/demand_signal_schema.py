"""
PySpark schema for demand.signals Bronze topic.

13 fields. Uses signal_type (not event_type) — demand signals are not
"events" in the operational sense; they are forecasting inputs.
The Bronze job normalises signal_type → event_type column for consistent
partitioning across all Bronze topics.
"""
from __future__ import annotations

from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

DEMAND_SIGNAL_SCHEMA = StructType([
    StructField("event_id",              StringType(),  nullable=False),
    StructField("schema_version",        StringType(),  nullable=False),
    StructField("signal_type",           StringType(),  nullable=False),
    StructField("warehouse_id",          StringType(),  nullable=False),
    StructField("sku_id",                StringType(),  nullable=False),
    StructField("demand_quantity",       FloatType(),   nullable=True),
    StructField("confidence_score",      FloatType(),   nullable=True),
    StructField("signal_source",         StringType(),  nullable=True),
    StructField("forecast_horizon_days", IntegerType(), nullable=True),
    StructField("timestamp",             StringType(),  nullable=False),
    StructField("producer_id",           StringType(),  nullable=True),
    StructField("correlation_id",        StringType(),  nullable=True),
    StructField("metadata",              StringType(),  nullable=True),
])
