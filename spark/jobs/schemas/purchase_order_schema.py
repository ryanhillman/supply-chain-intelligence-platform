from __future__ import annotations

"""
PySpark schema for purchase.orders.events Bronze topic.
Mirrors purchase_order_event.avsc (Phase 1.5).

16 fields. quantity_received is nullable — null until order is received.
actual_delivery_date is nullable — null until delivery occurs.
"""
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

PURCHASE_ORDER_EVENT_SCHEMA = StructType([
    StructField("event_id",               StringType(),  nullable=False),
    StructField("schema_version",         StringType(),  nullable=False),
    StructField("event_type",             StringType(),  nullable=False),
    StructField("order_id",               StringType(),  nullable=False),
    StructField("warehouse_id",           StringType(),  nullable=False),
    StructField("sku_id",                 StringType(),  nullable=False),
    StructField("supplier_id",            StringType(),  nullable=False),
    StructField("quantity_ordered",       IntegerType(), nullable=False),
    StructField("quantity_received",      IntegerType(), nullable=True),
    StructField("unit_cost",              FloatType(),   nullable=True),
    StructField("total_cost",             FloatType(),   nullable=True),
    StructField("order_date",             StringType(),  nullable=True),
    StructField("expected_delivery_date", StringType(),  nullable=True),
    StructField("actual_delivery_date",   StringType(),  nullable=True),
    StructField("timestamp",              StringType(),  nullable=False),
    StructField("producer_id",            StringType(),  nullable=True),
])
