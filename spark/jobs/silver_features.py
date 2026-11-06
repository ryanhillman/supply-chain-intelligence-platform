"""
Silver feature computation: one feature snapshot per (warehouse_id, sku_id)
as of the end of each day in a date range.

compute_features() reads Bronze through read_bronze(), which returns each
event once for every as-of date whose window contains it and never for a
date before the event, so a snapshot does not see the future. Every feature
group then aggregates by (warehouse_id, sku_id, as_of). A single date is a
range of one day. spark/jobs/silver_batch.py is the job that calls it and
writes the result.

Feature groups and their Bronze windows (as-of date D; every window ends at
midnight after D). The training set is built from these snapshots
(spark/jobs/training_set.py), so training and scoring share one feature
implementation; tests/test_silver_features.py checks the windows.

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
    DAYS_OF_SUPPLY_CAP,
    DAYS_OF_SUPPLY_EPS,
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
KEY  = [*PAIR, "as_of"]

def _event_ts():
    return F.col("timestamp").cast("timestamp")


def _event_date():
    return F.to_date(_event_ts())


def _midnight(d: date):
    """Midnight at the start of `d` as a Spark timestamp (session time zone is UTC)."""
    return F.lit(d.isoformat()).cast("timestamp")


# ── Bronze readers ─────────────────────────────────────────────────────────────

def read_events(
    spark: SparkSession, bronze_base: str, topic: str, schema, start: date, end: date
) -> DataFrame:
    """
    Bronze events for `topic` from midnight at the start of `start` up to (not
    including) midnight at the start of `end`, one row per event_id.

    The year/month/day predicates prune partition directories before any file
    is opened; the timestamp predicates then apply the exact window.
    """
    path = f"{bronze_base}/{topic}"
    part_date = F.make_date(F.col("year").cast("int"), F.col("month").cast("int"), F.col("day").cast("int"))
    return (
        spark.read
        .schema(schema)
        .option("basePath", path)
        .parquet(path)
        .filter((part_date >= F.lit(start)) & (part_date < F.lit(end)))
        .filter((_event_ts() >= _midnight(start)) & (_event_ts() < _midnight(end)))
        .dropDuplicates(["event_id"])
    )


def read_bronze(
    spark: SparkSession, bronze_base: str, topic: str, schema, first: date, last: date, days_back: int
) -> DataFrame:
    """
    The events in each as-of date's window, for the as-of dates `first`..`last`.

    The window of as-of date D runs from midnight `days_back` days before D
    up to midnight after D. An event on day E is therefore in the window of
    every D with E <= D <= E + days_back, and is returned once for each such
    D in the range, with D in the `as_of` column.
    """
    events = read_events(
        spark, bronze_base, topic, schema, first - timedelta(days=days_back), last + timedelta(days=1)
    )
    as_of_dates = F.sequence(
        F.greatest(_event_date(), F.lit(first)),
        F.least(F.date_add(_event_date(), days_back), F.lit(last)),
    )
    return events.withColumn("as_of", F.explode(as_of_dates))


def bronze_date_range(spark: SparkSession, bronze_base: str) -> tuple[date, date] | None:
    """The first and last day with inventory events in Bronze, or None if there are none."""
    path = f"{bronze_base}/inventory.events"
    row = (
        spark.read.schema(INVENTORY_EVENT_SCHEMA).option("basePath", path).parquet(path)
        .agg(F.min(_event_date()).alias("first"), F.max(_event_date()).alias("last"))
        .first()
    )
    return (row["first"], row["last"]) if row and row["last"] else None


# ── Feature groups ─────────────────────────────────────────────────────────────

def _inventory_snapshot(inventory_30d: DataFrame) -> DataFrame:
    """Latest inventory event per pair and as-of date: current_qty, reorder settings, last event."""
    w = Window.partitionBy(*KEY).orderBy(_event_ts().desc())
    return (
        inventory_30d
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .select(
            *KEY,
            F.col("quantity_after").cast("double").alias("inventory__current_qty"),
            F.col("reorder_point").cast("double").alias("inventory__reorder_point"),
            F.col("safety_stock").cast("double").alias("inventory__safety_stock"),
            F.col("event_type").alias("inventory__last_event_type"),
            F.col("timestamp").alias("inventory__last_updated"),
        )
    )


def _velocity(inventory_30d: DataFrame) -> DataFrame:
    """
    Sales velocity from SALE postings over the last 7 and 30 days.

    Averages divide by the days actually observed (capped at the window),
    not by a fixed 7 or 30: during a pair's first week sales_7d == sales_30d,
    and fixed divisors would turn that into a velocity_trend of 30/7 instead
    of a flat 1.0. stddev_daily_7d is the spread of the daily totals.
    """
    sales = inventory_30d.filter(F.col("event_type") == "SALE")
    units = -F.col("quantity_delta")

    days_observed = (
        sales.groupBy(*KEY)
        .agg(F.min(_event_date()).alias("first_seen"))
        .withColumn("days_observed_30d", F.greatest(F.lit(1), F.least(
            F.lit(30), F.datediff(F.col("as_of"), F.col("first_seen")) + F.lit(1))))
        .withColumn("days_observed_7d", F.least(F.lit(7), F.col("days_observed_30d")))
        .select(*KEY, "days_observed_30d", "days_observed_7d")
    )
    sales_30d = sales.groupBy(*KEY).agg(F.sum(units).alias("velocity__sales_30d"))

    daily_7d = (
        sales.filter(_event_date() >= F.date_sub(F.col("as_of"), 6))
        .groupBy(*KEY, _event_date().alias("sale_date"))
        .agg(F.sum(units).alias("daily_units"))
    )
    sales_7d = daily_7d.groupBy(*KEY).agg(
        F.sum("daily_units").alias("velocity__sales_7d"),
        F.stddev_pop("daily_units").alias("velocity__stddev_daily_7d"),
    )

    return (
        sales_30d
        .join(sales_7d, KEY, "left")
        .join(days_observed, KEY, "left")
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
    counts = demand_24h.groupBy(*KEY).agg(
        F.count("*").cast("double").alias("demand__demand_signal_count_24h"))
    w = Window.partitionBy(*KEY).orderBy(_event_ts().desc())
    last = (
        demand_24h.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1)
        .select(*KEY, F.col("signal_type").alias("demand__last_signal_type"))
    )
    return counts.join(last, KEY, "left")


def _supplier(supplier_90d: DataFrame, po_90d: DataFrame, shipments_30d: DataFrame,
              pairs: DataFrame) -> DataFrame:
    """
    Lead time and reliability per warehouse; open order and in-transit quantity
    per pair. An order counts until its RECEIVED event and a shipment until its
    DELIVERED event (matched by order_id / shipment_id).
    """
    warehouse_key = ["warehouse_id", "as_of"]
    supplier_agg = (
        supplier_90d.join(pairs.select(*warehouse_key).distinct(), warehouse_key, "inner")
        .groupBy(*warehouse_key)
        .agg(
            F.avg("lead_time_days").alias("supplier__lead_time_days"),
            F.stddev_pop("lead_time_days").alias("supplier__lead_time_stddev"),
            # on_time_delivery_rate is 0/1 per completed order; its mean is the reliability
            F.avg("on_time_delivery_rate").alias("supplier__supplier_reliability"),
        )
    )
    received = po_90d.filter(F.col("event_type") == "RECEIVED").select("order_id", "as_of")
    active_po = (
        po_90d.filter(F.col("event_type") == "CREATED")
        .dropDuplicates(["order_id", "as_of"])
        .join(received, ["order_id", "as_of"], "left_anti")
        .groupBy(*KEY).agg(F.sum("quantity_ordered").cast("double").alias("supplier__active_po_qty"))
    )
    delivered = shipments_30d.filter(F.col("event_type") == "DELIVERED").select("shipment_id", "as_of")
    in_transit = (
        shipments_30d.filter(F.col("event_type") == "SHIPPED")
        .join(delivered, ["shipment_id", "as_of"], "left_anti")
        .groupBy(*KEY).agg(F.sum("quantity").cast("double").alias("supplier__in_transit_qty"))
    )
    # Anchored on all pairs: a pair with no open order keeps its supplier and
    # in-transit values.
    return (
        pairs
        .join(active_po, KEY, "left")
        .join(in_transit, KEY, "left")
        .join(supplier_agg, warehouse_key, "left")
    )


def _risk(df: DataFrame) -> DataFrame:
    """
    days_of_supply and reorder_urgency (0 OK, 1 WATCH, 2 REORDER, 3 CRITICAL).
    days_of_supply is the formula the model's input uses
    (ml_training/datasets/feature_engineer.py): with no sales it is the cap
    for an entity in stock and 0 for one that is out.
    """
    # greatest() skips a null average (a pair with no sales in 30 days).
    dos = F.least(
        F.col("inventory__current_qty")
        / F.greatest(F.col("velocity__avg_daily_sales_7d"), F.lit(DAYS_OF_SUPPLY_EPS)),
        F.lit(DAYS_OF_SUPPLY_CAP),
    )
    return (
        df.withColumn("inventory__days_of_supply", dos)
        .withColumn("risk__reorder_urgency",
            F.when((F.col("inventory__current_qty") == 0)
                   | (F.col("inventory__days_of_supply") <= STOCKOUT_CRITICAL_DAYS), F.lit(3))
            .when(F.col("inventory__days_of_supply") <= REORDER_CRITICAL_DAYS, F.lit(2))
            .when(F.col("inventory__days_of_supply") <= REORDER_WATCH_DAYS, F.lit(1))
            .otherwise(F.lit(0)))
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def compute_features(
    spark: SparkSession, bronze_base: str, first: date, last: date | None = None
) -> DataFrame:
    """
    Feature rows for the as-of dates `first`..`last` (one date when `last` is
    omitted): one row per as-of date and pair with inventory events in the
    30 days up to that date. The date is in the `as_of` column.
    """
    last = last or first
    inventory_30d = read_bronze(spark, bronze_base, "inventory.events", INVENTORY_EVENT_SCHEMA, first, last, 29)
    pairs = inventory_30d.select(*KEY).distinct()

    demand_24h    = read_bronze(spark, bronze_base, "demand.signals", DEMAND_SIGNAL_SCHEMA, first, last, 0)
    supplier_90d  = read_bronze(spark, bronze_base, "suppliers.events", SUPPLIER_EVENT_SCHEMA, first, last, 90)
    po_90d        = read_bronze(spark, bronze_base, "purchase.orders.events",
                                PURCHASE_ORDER_EVENT_SCHEMA, first, last, 90)
    shipments_30d = read_bronze(spark, bronze_base, "shipments.events", SHIPMENT_EVENT_SCHEMA, first, last, 30)

    features = (
        _inventory_snapshot(inventory_30d)
        .join(_velocity(inventory_30d), KEY, "left")
        .join(_demand(demand_24h.join(pairs, KEY, "inner")), KEY, "left")
        .join(_supplier(supplier_90d, po_90d.join(pairs, KEY, "inner"),
                        shipments_30d.join(pairs, KEY, "inner"), pairs), KEY, "left")
    )
    return _risk(features)
