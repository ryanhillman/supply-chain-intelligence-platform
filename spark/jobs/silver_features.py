"""
Silver feature computation: one feature snapshot per (warehouse_id, sku_id)
as of the end of a given day.

compute_features() reads Bronze through read_bronze(), which keeps only
events strictly before midnight after the as-of date, so the snapshot never
sees the future. spark/jobs/silver_batch.py is the job that calls it and
writes the result; tests/test_spark_parity.py calls it directly and compares
the output with the training sampler's features for the same date.

Feature groups and their Bronze windows (as-of date D; every window ends at
midnight after D). ml_training/datasets/bronze_temporal_sampler.py computes
the same features for training with the same windows; tests/test_spark_parity.py
checks that the two agree.

  inventory   latest event, D-29..D             current_qty, reorder_point,
                                                safety_stock, last event
  velocity    SALE postings, D-6..D and D-29..D sales, averages, trend, stddev
  demand      demand signals on D               signal count, last signal type
  supplier    supplier events, D-90..D          lead time, stddev, reliability
              (per warehouse)
  pipeline    orders created D-90..D and not    active_po_qty
              yet received
              shipments shipped D-30..D and     in_transit_qty
              not yet delivered
  risk        derived                           days_of_supply, reorder_urgency

The world model posts one SALE per pair per day, so the latest inventory
event is always inside the 30-day window. Bronze may hold the same event
twice (a repeated replay); reads drop duplicate event_ids.
"""
from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F

from config.feature_config import (
    REORDER_CRITICAL_DAYS,
    REORDER_WATCH_DAYS,
    STOCKOUT_CRITICAL_DAYS,
)
from jobs.schemas.demand_signal_schema import DEMAND_SIGNAL_SCHEMA
from jobs.schemas.inventory_schema import INVENTORY_EVENT_SCHEMA
from jobs.schemas.purchase_order_schema import PURCHASE_ORDER_EVENT_SCHEMA
from jobs.schemas.shipment_schema import SHIPMENT_EVENT_SCHEMA
from jobs.schemas.supplier_schema import SUPPLIER_EVENT_SCHEMA

PAIR = ["warehouse_id", "sku_id"]


def _midnight(d: date):
    """Midnight at the start of `d` as a Spark timestamp (session time zone is UTC)."""
    return F.lit(d.isoformat()).cast("timestamp")


# ── Bronze reader ──────────────────────────────────────────────────────────────

def read_bronze(
    spark: SparkSession, bronze_base: str, topic: str, schema, as_of: date, days_back: int
) -> DataFrame:
    """
    Bronze events for `topic` from midnight `days_back` days before `as_of`
    up to (not including) midnight after `as_of`, one row per event_id.

    The year/month/day predicates prune partition directories before any file
    is opened; the timestamp predicates then apply the exact window.
    """
    path  = f"{bronze_base}/{topic}"
    start = as_of - timedelta(days=days_back)
    end   = as_of + timedelta(days=1)
    part_date = F.make_date(F.col("year").cast("int"), F.col("month").cast("int"), F.col("day").cast("int"))
    ts = F.col("timestamp").cast("timestamp")
    return (
        spark.read
        .schema(schema)
        .option("basePath", path)
        .parquet(path)
        .filter((part_date >= F.lit(start)) & (part_date < F.lit(end)))
        .filter((ts >= _midnight(start)) & (ts < _midnight(end)))
        .dropDuplicates(["event_id"])
    )


def latest_event_date(spark: SparkSession, bronze_base: str) -> date | None:
    """Most recent day with inventory events in Bronze (the default as-of date)."""
    path = f"{bronze_base}/inventory.events"
    row = (
        spark.read.schema(INVENTORY_EVENT_SCHEMA).option("basePath", path).parquet(path)
        .agg(F.max(F.to_date(F.col("timestamp").cast("timestamp"))).alias("d"))
        .first()
    )
    return row["d"] if row else None


# ── Feature groups ─────────────────────────────────────────────────────────────

def _inventory_snapshot(inventory_30d: DataFrame) -> DataFrame:
    """Latest inventory event per pair: current_qty, reorder settings, last event."""
    w = Window.partitionBy(*PAIR).orderBy(F.col("timestamp").cast("timestamp").desc())
    return (
        inventory_30d
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .select(
            *PAIR,
            F.col("quantity_after").cast("double").alias("inventory__current_qty"),
            F.col("reorder_point").cast("double").alias("inventory__reorder_point"),
            F.col("safety_stock").cast("double").alias("inventory__safety_stock"),
            F.col("event_type").alias("inventory__last_event_type"),
            F.col("timestamp").alias("inventory__last_updated"),
        )
    )


def _velocity(inventory_30d: DataFrame, as_of: date) -> DataFrame:
    """
    Sales velocity from SALE postings over the last 7 and 30 days.

    Averages divide by the days actually observed (capped at the window),
    not by a fixed 7 or 30: during a pair's first week sales_7d == sales_30d,
    and fixed divisors would turn that into a velocity_trend of 30/7 instead
    of a flat 1.0. stddev_daily_7d is the spread of daily totals, as in the
    training sampler.
    """
    sales = inventory_30d.filter(F.col("event_type") == "SALE")
    units = -F.col("quantity_delta")

    days_observed = (
        sales.groupBy(*PAIR)
        .agg(F.min(F.col("timestamp").cast("timestamp")).alias("first_seen"))
        .withColumn("days_observed_30d", F.greatest(F.lit(1), F.least(
            F.lit(30), F.datediff(F.lit(as_of), F.to_date("first_seen")) + F.lit(1))))
        .withColumn("days_observed_7d", F.least(F.lit(7), F.col("days_observed_30d")))
        .select(*PAIR, "days_observed_30d", "days_observed_7d")
    )
    sales_30d = sales.groupBy(*PAIR).agg(F.sum(units).alias("velocity__sales_30d"))

    daily_7d = (
        sales.filter(F.col("timestamp").cast("timestamp") >= _midnight(as_of - timedelta(days=6)))
        .groupBy(*PAIR, F.to_date(F.col("timestamp").cast("timestamp")).alias("sale_date"))
        .agg(F.sum(units).alias("daily_units"))
    )
    sales_7d = daily_7d.groupBy(*PAIR).agg(
        F.sum("daily_units").alias("velocity__sales_7d"),
        F.stddev_pop("daily_units").alias("velocity__stddev_daily_7d"),
    )

    return (
        sales_30d
        .join(sales_7d, PAIR, "left")
        .join(days_observed, PAIR, "left")
        .withColumn("velocity__avg_daily_sales_30d",
                    F.col("velocity__sales_30d") / F.col("days_observed_30d").cast("double"))
        .withColumn("velocity__avg_daily_sales_7d",
                    F.col("velocity__sales_7d") / F.col("days_observed_7d").cast("double"))
        .withColumn("velocity__velocity_trend", F.when(
            F.col("velocity__avg_daily_sales_30d") > 0,
            F.col("velocity__avg_daily_sales_7d") / F.col("velocity__avg_daily_sales_30d"),
        ).otherwise(F.lit(None).cast("double")))
        .drop("days_observed_30d", "days_observed_7d")
    )


def _demand(demand_24h: DataFrame) -> DataFrame:
    """Demand signal count in the window and the latest signal type."""
    counts = demand_24h.groupBy(*PAIR).agg(
        F.count("*").cast("double").alias("demand__demand_signal_count_24h"))
    w = Window.partitionBy(*PAIR).orderBy(F.col("timestamp").cast("timestamp").desc())
    last = (
        demand_24h.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1)
        .select(*PAIR, F.col("signal_type").alias("demand__last_signal_type"))
    )
    return counts.join(last, PAIR, "left")


def _supplier(supplier_90d: DataFrame, po_90d: DataFrame, shipments_30d: DataFrame,
              pairs: DataFrame) -> DataFrame:
    """
    Lead time and reliability per warehouse; open order and in-transit quantity
    per pair. An order counts until its RECEIVED event and a shipment until its
    DELIVERED event (matched by order_id / shipment_id).
    """
    supplier_agg = (
        supplier_90d.join(pairs.select("warehouse_id").distinct(), "warehouse_id", "inner")
        .groupBy("warehouse_id")
        .agg(
            F.avg("lead_time_days").alias("supplier__lead_time_days"),
            F.stddev_pop("lead_time_days").alias("supplier__lead_time_stddev"),
            # on_time_delivery_rate is 0/1 per completed order; its mean is the reliability
            F.avg("on_time_delivery_rate").alias("supplier__supplier_reliability"),
        )
    )
    received = po_90d.filter(F.col("event_type") == "RECEIVED").select("order_id")
    active_po = (
        po_90d.filter(F.col("event_type").isin("CREATED", "SUBMITTED"))
        .dropDuplicates(["order_id"])
        .join(received, "order_id", "left_anti")
        .groupBy(*PAIR).agg(F.sum("quantity_ordered").cast("double").alias("supplier__active_po_qty"))
    )
    delivered = shipments_30d.filter(F.col("event_type") == "DELIVERED").select("shipment_id")
    in_transit = (
        shipments_30d.filter(F.col("event_type") == "SHIPPED")
        .join(delivered, "shipment_id", "left_anti")
        .groupBy(*PAIR).agg(F.sum("quantity").cast("double").alias("supplier__in_transit_qty"))
    )
    # Anchored on all pairs: a pair with no open order keeps its supplier and
    # in-transit values.
    return (
        pairs
        .join(active_po, PAIR, "left")
        .join(in_transit, PAIR, "left")
        .join(supplier_agg, "warehouse_id", "left")
    )


def _risk(df: DataFrame) -> DataFrame:
    """
    days_of_supply and reorder_urgency (0 OK, 1 WATCH, 2 REORDER, 3 CRITICAL).
    stockout_risk_score and shortage_prob_7d stay null in Silver: predictions
    are stored in Bigtable, never written back here.
    """
    dos = F.when(
        F.col("velocity__avg_daily_sales_7d") > 0,
        F.col("inventory__current_qty") / F.col("velocity__avg_daily_sales_7d"),
    ).otherwise(F.lit(999.0))   # no recent sales: effectively unlimited supply
    return (
        df.withColumn("inventory__days_of_supply", dos)
        .withColumn("risk__reorder_urgency",
            F.when((F.col("inventory__current_qty") == 0)
                   | (F.col("inventory__days_of_supply") <= STOCKOUT_CRITICAL_DAYS), F.lit(3))
            .when(F.col("inventory__days_of_supply") <= REORDER_CRITICAL_DAYS, F.lit(2))
            .when(F.col("inventory__days_of_supply") <= REORDER_WATCH_DAYS, F.lit(1))
            .otherwise(F.lit(0)))
        .withColumn("risk__stockout_risk_score", F.lit(None).cast("double"))
        .withColumn("risk__shortage_prob_7d", F.lit(None).cast("double"))
        # Reserved demand-forecast columns, not populated by any stage
        .withColumn("demand__demand_forecast_7d", F.lit(None).cast("double"))
        .withColumn("demand__demand_forecast_30d", F.lit(None).cast("double"))
        .withColumn("demand__demand_multiplier", F.lit(None).cast("double"))
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def compute_features(spark: SparkSession, bronze_base: str, as_of: date) -> tuple[DataFrame, int]:
    """
    Feature rows for every pair with inventory events in the 30 days up to
    `as_of`, plus the number of inventory events read (for lineage).
    """
    inventory_30d = read_bronze(spark, bronze_base, "inventory.events", INVENTORY_EVENT_SCHEMA, as_of, 29)
    pairs = inventory_30d.select(*PAIR).distinct()

    demand_24h    = read_bronze(spark, bronze_base, "demand.signals", DEMAND_SIGNAL_SCHEMA, as_of, 0)
    supplier_90d  = read_bronze(spark, bronze_base, "suppliers.events", SUPPLIER_EVENT_SCHEMA, as_of, 90)
    po_90d        = read_bronze(spark, bronze_base, "purchase.orders.events",
                                PURCHASE_ORDER_EVENT_SCHEMA, as_of, 90)
    shipments_30d = read_bronze(spark, bronze_base, "shipments.events", SHIPMENT_EVENT_SCHEMA, as_of, 30)

    features = (
        _inventory_snapshot(inventory_30d)
        .join(_velocity(inventory_30d, as_of), PAIR, "left")
        .join(_demand(demand_24h.join(pairs, PAIR, "inner")), PAIR, "left")
        .join(_supplier(supplier_90d, po_90d.join(pairs, PAIR, "inner"),
                        shipments_30d.join(pairs, PAIR, "inner"), pairs), PAIR, "left")
    )
    return _risk(features), inventory_30d.count()
