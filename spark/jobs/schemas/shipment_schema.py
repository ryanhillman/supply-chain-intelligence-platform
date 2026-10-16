"""
PySpark schema for shipments.events Bronze topic.

16 fields. quantity is the number of units in this shipment leg.
actual_arrival is nullable — null until the shipment is DELIVERED.
"""
from __future__ import annotations

from pyspark.sql.types import (
    IntegerType,
    StringType,
    StructField,
    StructType,
)

SHIPMENT_EVENT_SCHEMA = StructType([
    StructField("event_id",           StringType(),  nullable=False),
    StructField("schema_version",     StringType(),  nullable=False),
    StructField("event_type",         StringType(),  nullable=False),
    StructField("shipment_id",        StringType(),  nullable=False),
    StructField("warehouse_id",       StringType(),  nullable=False),
    StructField("sku_id",             StringType(),  nullable=False),
    StructField("quantity",           IntegerType(), nullable=False),
    StructField("supplier_id",        StringType(),  nullable=True),
    StructField("carrier",            StringType(),  nullable=True),
    StructField("tracking_number",    StringType(),  nullable=True),
    StructField("estimated_arrival",  StringType(),  nullable=True),
    StructField("actual_arrival",     StringType(),  nullable=True),
    StructField("timestamp",          StringType(),  nullable=False),
    StructField("producer_id",        StringType(),  nullable=True),
    StructField("correlation_id",     StringType(),  nullable=True),
    StructField("metadata",           StringType(),  nullable=True),
])
