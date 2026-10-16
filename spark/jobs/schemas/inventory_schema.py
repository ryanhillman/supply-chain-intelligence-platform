"""
PySpark schema for inventory.events Bronze topic.

16 fields. FloatType for unit_cost (matches Avro float).
The quantity_* fields are IntegerType (Avro int).
reference_id is nullable — only present on TRANSFER events.
"""
from __future__ import annotations

from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

INVENTORY_EVENT_SCHEMA = StructType([
    StructField("event_id",        StringType(),  nullable=False),
    StructField("schema_version",  StringType(),  nullable=False),
    StructField("event_type",      StringType(),  nullable=False),
    StructField("warehouse_id",    StringType(),  nullable=False),
    StructField("sku_id",          StringType(),  nullable=False),
    StructField("quantity_before", IntegerType(), nullable=False),
    StructField("quantity_after",  IntegerType(), nullable=False),
    StructField("quantity_delta",  IntegerType(), nullable=False),
    StructField("unit_cost",       FloatType(),   nullable=True),
    StructField("reorder_point",   IntegerType(), nullable=True),
    StructField("safety_stock",    IntegerType(), nullable=True),
    StructField("reference_id",    StringType(),  nullable=True),
    StructField("timestamp",       StringType(),  nullable=False),
    StructField("producer_id",     StringType(),  nullable=True),
    StructField("correlation_id",  StringType(),  nullable=True),
    StructField("metadata",        StringType(),  nullable=True),
])
