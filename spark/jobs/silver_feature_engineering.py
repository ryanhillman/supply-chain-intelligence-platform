"""
Silver Feature Engineering — Spark Structured Streaming job.

Reads from Bronze GCS (inventory events as trigger stream) and computes
a complete 29-feature vector per (warehouse_id, sku_id) snapshot.

Architecture:
  Bronze GCS (inventory.events) → readStream trigger
  foreachBatch:
    ├── Batch read 30d inventory Bronze  → inventory snapshot + velocity
    ├── Batch read 24h demand Bronze     → demand features
    ├── Batch read 90d supplier Bronze   → lead time + reliability
    ├── Batch read active PO Bronze      → active_po_qty
    ├── Batch read in-transit Bronze     → in_transit_qty
    ├── Derive risk features             → urgency, days_until_stockout
    └── Add lineage columns              → epoch, timestamp, version
  Write Silver GCS (Parquet, partitioned by warehouse_id/year/month/day)
  Write Bigtable   (online feature store, <10ms reads for inference)

Dual-write guarantee:
  Silver GCS is written first (durable offline store).
  Bigtable write is attempted second; failure is caught and logged but does
  NOT fail the streaming query or roll back the Silver write. This gives us
  Silver as the source of truth and Bigtable as a best-effort online cache
  that can be rebuilt by replaying Silver at any time.

Note: Batch reads inside foreachBatch is a valid Lambda-hybrid pattern for
multi-source feature engineering. It avoids stream-stream join complexity
while still processing incrementally (only pairs active in the micro-batch
are recomputed).
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, "/opt/spark-app")

from config.bronze_config import (
    BIGTABLE_INSTANCE_ID,
    BIGTABLE_TABLE_ID,
    BRONZE_BASE_PATH,
    GCP_PROJECT_ID,
    SILVER_BASE_PATH,
    SILVER_CHECKPOINT_BASE_PATH,
    TRIGGER_INTERVAL,
)
from utils.bigtable_writer import write_to_bigtable
from config.feature_config import (
    PIPELINE_VERSION,
    REORDER_CRITICAL_DAYS,
    REORDER_WATCH_DAYS,
    ROW_KEY_SEPARATOR,
    STOCKOUT_CRITICAL_DAYS,
)
from jobs.schemas.inventory_schema import INVENTORY_EVENT_SCHEMA
from jobs.schemas.demand_signal_schema import DEMAND_SIGNAL_SCHEMA
from jobs.schemas.purchase_order_schema import PURCHASE_ORDER_EVENT_SCHEMA
from jobs.schemas.shipment_schema import SHIPMENT_EVENT_SCHEMA
from jobs.schemas.supplier_schema import SUPPLIER_EVENT_SCHEMA
from jobs.schemas.feature_schema import SILVER_DEDUP_COLS, SILVER_PARTITION_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("silver_feature_engineering")


# ── Spark session ──────────────────────────────────────────────────────────────

def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("scip-silver-feature-engineering")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.shuffle.partitions", "6")
        # GCS connector (JAR pre-installed in Dockerfile)
        .config("spark.hadoop.fs.gs.impl",
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl",
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        .getOrCreate()
    )


# ── Bronze batch readers ───────────────────────────────────────────────────────

def _read_bronze(spark: SparkSession, topic: str, schema, days_back: int) -> DataFrame:
    """
    Batch read of Bronze Parquet for the last `days_back` days.

    Uses partition-column predicates (year/month/day) to push GCS directory
    listing down to only the relevant date partitions. Without this, a full
    recursive listing of the topic root scans thousands of small files even
    when the logical dataset is small — causing driver OOM in the dev env.
    """
    from datetime import date, timedelta

    path = f"{BRONZE_BASE_PATH}/{topic}"
    cutoff_date = date.today() - timedelta(days=days_back)

    # Spark reads year/month/day/hour as partition columns from the directory
    # names. Adding a filter predicate on these columns enables partition
    # pruning so the driver only lists directories within the cutoff date
    # range. make_date() expresses this as a single date comparison instead
    # of a hand-rolled lexicographic (year, month, day) OR-chain.
    partition_filter = (
        F.make_date(
            F.col("year").cast("int"),
            F.col("month").cast("int"),
            F.col("day").cast("int"),
        ) >= F.lit(cutoff_date)
    )

    cutoff_ts = F.lit(cutoff_date.isoformat()).cast("timestamp")

    return (
        spark.read
        .schema(schema)
        .option("basePath", path)
        .parquet(path)
        .filter(partition_filter)
        .filter(F.col("timestamp").cast("timestamp") >= cutoff_ts)
    )


# ── Feature computation ────────────────────────────────────────────────────────

def _compute_inventory_snapshot(inventory_30d: DataFrame) -> DataFrame:
    """
    Latest inventory state per (warehouse_id, sku_id).

    Uses a window function to pick the most recent event, which gives us
    current_qty (quantity_after), reorder_point, and safety_stock.
    """
    w = Window.partitionBy("warehouse_id", "sku_id").orderBy(F.col("timestamp").desc())

    return (
        inventory_30d
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .select(
            "warehouse_id",
            "sku_id",
            F.col("quantity_after").cast("double").alias("inventory__current_qty"),
            F.col("reorder_point").cast("double").alias("inventory__reorder_point"),
            F.col("safety_stock").cast("double").alias("inventory__safety_stock"),
            F.col("event_type").alias("inventory__last_event_type"),
            F.col("timestamp").alias("inventory__last_updated"),
        )
    )


def _compute_velocity(inventory_30d: DataFrame) -> DataFrame:
    """
    Sales velocity features from the last 7 and 30 days of inventory events.

    SALE events have negative quantity_delta (stock decreases).
    We sum -quantity_delta to get units sold (positive number).

    avg_daily_sales_{7,30}d divide by the ACTUAL number of days a
    (warehouse, sku) pair has been observed selling, not hardcoded 7.0 /
    30.0. During a cold start (less than 7 days since the first sale),
    inventory_30d and the 7-day-filtered subset cover the exact same
    events, so sales_7d == sales_30d; dividing by mismatched fixed
    constants (7 vs 30) then manufactures a velocity_trend of exactly
    30/7 ≈ 4.29 regardless of the real trend (confirmed ~11 standard
    deviations outside the training distribution). Dividing both by the
    true elapsed days makes avg_daily_sales_7d == avg_daily_sales_30d in
    that case, so velocity_trend correctly lands at 1.0 (flat — there
    simply isn't enough history to say otherwise) instead of a numeric
    artifact.
    """
    cutoff_7d = F.date_sub(F.current_date(), 7).cast("timestamp")

    sales_events = inventory_30d.filter(F.col("event_type") == "SALE")

    # Actual days observed per (warehouse, sku), capped by the read window.
    # inventory_30d already only contains the last 30 days (see
    # _read_bronze), so first_seen can never be older than that.
    days_observed = (
        sales_events
        .groupBy("warehouse_id", "sku_id")
        .agg(F.min("timestamp").alias("first_seen"))
        .withColumn(
            "days_observed_30d",
            F.greatest(F.lit(1), F.least(
                F.lit(30),
                F.datediff(F.current_date(), F.to_date(F.col("first_seen"))) + F.lit(1),
            )),
        )
        .withColumn("days_observed_7d", F.least(F.lit(7), F.col("days_observed_30d")))
        .select("warehouse_id", "sku_id", "days_observed_30d", "days_observed_7d")
    )

    velocity_30d = (
        sales_events
        .groupBy("warehouse_id", "sku_id")
        .agg(F.sum(-F.col("quantity_delta")).alias("velocity__sales_30d"))
    )

    # stddev_daily_7d is the standard deviation ACROSS DAYS, not across
    # individual SALE transactions — so daily totals are aggregated first.
    # This matches ml_training/datasets/bronze_temporal_sampler.py and
    # synthetic_generator.py, both of which compute it the same way; using
    # stddev_pop directly over per-transaction quantities here would be a
    # different statistic from what the training pipeline produces for the
    # same feature name.
    daily_sales_7d = (
        sales_events
        .filter(F.col("timestamp").cast("timestamp") >= cutoff_7d)
        .groupBy("warehouse_id", "sku_id", F.to_date(F.col("timestamp")).alias("sale_date"))
        .agg(F.sum(-F.col("quantity_delta")).alias("daily_units_sold"))
    )

    velocity_7d = (
        daily_sales_7d
        .groupBy("warehouse_id", "sku_id")
        .agg(
            F.sum("daily_units_sold").alias("velocity__sales_7d"),
            F.stddev_pop("daily_units_sold").alias("velocity__stddev_daily_7d"),
        )
    )

    return (
        velocity_30d
        .join(velocity_7d,    ["warehouse_id", "sku_id"], "left")
        .join(days_observed,  ["warehouse_id", "sku_id"], "left")
        .withColumn(
            "velocity__avg_daily_sales_30d",
            F.col("velocity__sales_30d") / F.col("days_observed_30d").cast("double"),
        )
        .withColumn(
            "velocity__avg_daily_sales_7d",
            F.col("velocity__sales_7d") / F.col("days_observed_7d").cast("double"),
        )
        .withColumn(
            "velocity__velocity_trend",
            F.when(
                F.col("velocity__avg_daily_sales_30d") > 0,
                F.col("velocity__avg_daily_sales_7d") / F.col("velocity__avg_daily_sales_30d"),
            ).otherwise(F.lit(None).cast("double")),
        )
        .drop("days_observed_30d", "days_observed_7d")
    )


def _compute_demand_features(spark: SparkSession, hot_pairs: DataFrame) -> DataFrame:
    """
    Demand signal aggregations for active (warehouse_id, sku_id) pairs.
    Counts signal bursts in the last 24h and captures the latest signal type.
    """
    demand_24h = (
        _read_bronze(spark, "demand.signals", DEMAND_SIGNAL_SCHEMA, days_back=1)
        .join(hot_pairs, ["warehouse_id", "sku_id"], "inner")
    )

    signal_count = (
        demand_24h
        .groupBy("warehouse_id", "sku_id")
        .agg(F.count("*").cast("double").alias("demand__demand_signal_count_24h"))
    )

    # Latest signal type per pair
    w = Window.partitionBy("warehouse_id", "sku_id").orderBy(F.col("timestamp").desc())
    last_signal = (
        demand_24h
        .withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .select("warehouse_id", "sku_id", F.col("signal_type").alias("demand__last_signal_type"))
    )

    return signal_count.join(last_signal, ["warehouse_id", "sku_id"], "left")


def _compute_supplier_features(spark: SparkSession, hot_pairs: DataFrame) -> DataFrame:
    """
    Supplier-side features: lead time, reliability, active PO qty, in-transit qty.
    """
    # Lead time and reliability from supplier events (last 90 days)
    supplier_df = (
        _read_bronze(spark, "suppliers.events", SUPPLIER_EVENT_SCHEMA, days_back=90)
        .join(hot_pairs.select("warehouse_id").distinct(), "warehouse_id", "inner")
    )

    supplier_agg = (
        supplier_df
        .groupBy("warehouse_id")
        .agg(
            F.avg("lead_time_days").alias("supplier__lead_time_days"),
            F.stddev_pop("lead_time_days").alias("supplier__lead_time_stddev"),
            # on_time_delivery_rate is 0/1 per event; avg = reliability score
            F.avg("on_time_delivery_rate").alias("supplier__supplier_reliability"),
        )
    )

    # Active PO quantity (CREATED or SUBMITTED status, last 90 days)
    po_df = (
        _read_bronze(spark, "purchase.orders.events", PURCHASE_ORDER_EVENT_SCHEMA, days_back=90)
        .join(hot_pairs, ["warehouse_id", "sku_id"], "inner")
        .filter(F.col("event_type").isin("CREATED", "SUBMITTED"))
    )

    active_po = (
        po_df
        .groupBy("warehouse_id", "sku_id")
        .agg(F.sum("quantity_ordered").cast("double").alias("supplier__active_po_qty"))
    )

    # In-transit qty (SHIPPED status, not yet DELIVERED, last 30 days)
    shipment_df = (
        _read_bronze(spark, "shipments.events", SHIPMENT_EVENT_SCHEMA, days_back=30)
        .join(hot_pairs, ["warehouse_id", "sku_id"], "inner")
        .filter(F.col("event_type") == "SHIPPED")
    )

    in_transit = (
        shipment_df
        .groupBy("warehouse_id", "sku_id")
        .agg(F.sum("quantity").cast("double").alias("supplier__in_transit_qty"))
    )

    # Anchored on hot_pairs rather than active_po: a pair with no currently
    # open PO must still keep its in_transit_qty and supplier_agg values —
    # those facts are independent of whether a PO happens to be open right now.
    return (
        hot_pairs
        .join(active_po,    ["warehouse_id", "sku_id"], "left")
        .join(in_transit,   ["warehouse_id", "sku_id"], "left")
        .join(supplier_agg, "warehouse_id", "left")
    )


def _derive_risk_features(df: DataFrame) -> DataFrame:
    """
    Compute risk tier features from inventory snapshot and velocity.

    reorder_urgency = 0 (OK) | 1 (WATCH) | 2 (REORDER) | 3 (CRITICAL)

    stockout_risk_score and shortage_prob_7d are reserved null columns;
    they are not populated by any current pipeline stage.
    """
    days_of_supply_expr = F.when(
        F.col("velocity__avg_daily_sales_7d") > 0,
        F.col("inventory__current_qty") / F.col("velocity__avg_daily_sales_7d"),
    ).otherwise(F.lit(999.0))  # 999 = effectively infinite supply

    df = df.withColumn("inventory__days_of_supply", days_of_supply_expr)

    df = df.withColumn(
        "risk__reorder_urgency",
        F.when(
            (F.col("inventory__current_qty") == 0) |
            (F.col("inventory__days_of_supply") <= STOCKOUT_CRITICAL_DAYS),
            F.lit(3),
        ).when(
            F.col("inventory__days_of_supply") <= REORDER_CRITICAL_DAYS,
            F.lit(2),
        ).when(
            F.col("inventory__days_of_supply") <= REORDER_WATCH_DAYS,
            F.lit(1),
        ).otherwise(F.lit(0)),
    )

    df = df.withColumn("risk__stockout_risk_score", F.lit(None).cast("double"))
    df = df.withColumn("risk__shortage_prob_7d", F.lit(None).cast("double"))

    return df


def _add_reserved_forecast_columns(df: DataFrame) -> DataFrame:
    """Add null placeholder columns for demand forecasts not yet populated."""
    return (
        df.withColumn("demand__demand_forecast_7d",  F.lit(None).cast("double"))
          .withColumn("demand__demand_forecast_30d", F.lit(None).cast("double"))
          .withColumn("demand__demand_multiplier",   F.lit(None).cast("double"))
    )


def _add_lineage(df: DataFrame, epoch_id: int, event_count: int) -> DataFrame:
    """
    Append lineage columns: pipeline provenance + audit trail.

    event_count is a run-level diagnostic, not a per-entity count — every
    row in `df` gets the same value. Its scope differs by caller: the
    streaming job (this module) passes the current micro-batch's event
    count, while silver_batch.py passes the count of ALL historical Bronze
    inventory events read in that one-shot backfill run. Both answer
    "how many source events did the run that last wrote this row process,"
    but the two numbers are not comparable in magnitude — do not use this
    column to infer per-entity event density.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    return (
        df
        .withColumn("lineage__last_bronze_epoch", F.lit(epoch_id).cast("long"))
        .withColumn("lineage__last_silver_write", F.lit(now_iso))
        .withColumn("lineage__source_event_count", F.lit(event_count).cast("long"))
        .withColumn("lineage__pipeline_version", F.lit(PIPELINE_VERSION))
    )


def _add_partition_and_row_key(df: DataFrame) -> DataFrame:
    """
    Add year/month/day partition columns and pre-compute Bigtable row key.
    """
    now = datetime.now(timezone.utc)
    return (
        df
        .withColumn("row_key",
            F.concat_ws(ROW_KEY_SEPARATOR, F.col("warehouse_id"), F.col("sku_id")))
        .withColumn("window_end", F.current_timestamp())
        .withColumn("year",  F.lit(now.year))
        .withColumn("month", F.lit(now.month))
        .withColumn("day",   F.lit(now.day))
    )


# ── Silver writer ──────────────────────────────────────────────────────────────

def write_silver(feature_df: DataFrame, epoch_id: int) -> None:
    """
    Deduplicate and write the feature DataFrame to Silver GCS as Parquet.

    Partition layout:  silver/warehouse_id={id}/year={y}/month={m}/day={d}/
    Dedup key:         (warehouse_id, sku_id, window_end)
    """
    deduped = feature_df.dropDuplicates(SILVER_DEDUP_COLS)

    row_count = deduped.count()
    if row_count == 0:
        logger.info("write_silver: nothing to write for epoch=%d", epoch_id)
        return

    (
        deduped
        .write
        .mode("append")
        .partitionBy(*SILVER_PARTITION_COLS)
        .parquet(SILVER_BASE_PATH)
    )
    logger.info(
        "write_silver | epoch=%d rows=%d path=%s",
        epoch_id, row_count, SILVER_BASE_PATH,
    )


# ── foreachBatch processor ─────────────────────────────────────────────────────

def make_silver_processor(spark: SparkSession):
    """
    Factory that returns a foreachBatch function.

    Using a factory (not a lambda) avoids Python closure issues when `spark`
    is captured across multiple queries — the same pattern used in bronze_ingestion.py.
    """

    def process_silver_batch(batch_df: DataFrame, epoch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            logger.debug("silver batch empty | epoch=%d", epoch_id)
            return

        event_count = batch_df.count()
        logger.info("silver batch | epoch=%d events=%d", epoch_id, event_count)

        # ── Step 1: Identify active (warehouse_id, sku_id) pairs ──────────────
        # Only recompute features for pairs that received new events this batch.
        # No cache: hot_pairs is tiny (≤ 120 rows) and used in broadcast joins.
        hot_pairs = batch_df.select("warehouse_id", "sku_id").distinct()

        pair_count = hot_pairs.count()
        logger.info("hot pairs | epoch=%d count=%d", epoch_id, pair_count)

        # ── Step 2: Batch read 30d inventory Bronze ────────────────────────
        # No cache: inventory_30d is read twice (snapshot + velocity) but
        # each read is a GCS partition scan on a small dataset. Caching would
        # hold the full 30d scan in heap alongside the feature DataFrames,
        # which causes OOM in the low-memory dev environment.
        inventory_30d = (
            _read_bronze(spark, "inventory.events", INVENTORY_EVENT_SCHEMA, days_back=30)
            .join(hot_pairs, ["warehouse_id", "sku_id"], "inner")
        )

        # ── Step 3: Compute inventory and velocity features ────────────────
        snapshot_df = _compute_inventory_snapshot(inventory_30d)
        velocity_df = _compute_velocity(inventory_30d)

        # ── Step 4: Compute cross-source features ──────────────────────────
        demand_df   = _compute_demand_features(spark, hot_pairs)
        supplier_df = _compute_supplier_features(spark, hot_pairs)

        # ── Step 5: Join all feature groups ───────────────────────────────
        feature_df = (
            snapshot_df
            .join(velocity_df,  ["warehouse_id", "sku_id"], "left")
            .join(demand_df,    ["warehouse_id", "sku_id"], "left")
            .join(supplier_df,  ["warehouse_id", "sku_id"], "left")
        )

        # ── Step 6: Derive risk features ──────────────────────────────────
        feature_df = _derive_risk_features(feature_df)

        # ── Step 7: Reserved demand forecast columns ───────────────────────
        feature_df = _add_reserved_forecast_columns(feature_df)

        # ── Step 8: Lineage + partition + row key ─────────────────────────
        feature_df = _add_lineage(feature_df, epoch_id, event_count)
        feature_df = _add_partition_and_row_key(feature_df)

        # ── Step 9: Cache feature_df — read twice (Silver + Bigtable) ────────
        # Cache before the dual-write to avoid recomputing the full DAG
        # for the Bigtable write after the Silver write has already consumed it.
        feature_df = feature_df.cache()

        try:
            # ── Step 10: Silver GCS write (primary durable store) ──────────
            write_silver(feature_df, epoch_id)

            # ── Step 11: Bigtable dual-write (online feature store) ────────
            # Isolated in its own try/except: Bigtable failure must not roll
            # back or invalidate the Silver write. Silver is source of truth;
            # Bigtable can be rebuilt from Silver via a backfill job.
            try:
                write_to_bigtable(
                    df=feature_df,
                    project_id=GCP_PROJECT_ID,
                    instance_id=BIGTABLE_INSTANCE_ID,
                    table_id=BIGTABLE_TABLE_ID,
                    epoch_id=epoch_id,
                )
            except Exception as bt_exc:
                logger.error(
                    "Bigtable dual-write FAILED | epoch=%d error=%s | "
                    "Silver write succeeded — Bigtable can be rebuilt from Silver.",
                    epoch_id, bt_exc,
                )
        finally:
            feature_df.unpersist()

    return process_silver_batch


# ── Streaming query ────────────────────────────────────────────────────────────

def build_silver_query(spark: SparkSession):
    """
    Assemble the Silver streaming query.

    Source: inventory.events Bronze path (most frequent source — drives trigger cadence).
    Trigger: ProcessingTime(TRIGGER_INTERVAL) — same interval as Bronze job.
    Checkpoint: separate from Bronze checkpoints to allow independent restarts.
    """
    bronze_inventory_path = f"{BRONZE_BASE_PATH}/inventory.events"

    inventory_stream = (
        spark.readStream
        .schema(INVENTORY_EVENT_SCHEMA)
        .option("basePath", bronze_inventory_path)
        .option("maxFilesPerTrigger", 20)   # bound each micro-batch to 20 new files
        .parquet(bronze_inventory_path)
        # No withWatermark: watermarks only affect stateful operations (windowed
        # aggregations, stream-stream joins). This query uses foreachBatch with
        # no streaming state — all computation is batch reads inside each micro-batch.
        # Late-data tolerance is handled by the days_back cutoff in _read_bronze().
    )

    return (
        inventory_stream
        .writeStream
        .foreachBatch(make_silver_processor(spark))
        .trigger(processingTime=TRIGGER_INTERVAL)
        .option("checkpointLocation", SILVER_CHECKPOINT_BASE_PATH)
        .queryName("silver-feature-engineering")
        .start()
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info(
        "Silver feature engineering starting | bronze=%s silver=%s",
        BRONZE_BASE_PATH, SILVER_BASE_PATH,
    )

    query = build_silver_query(spark)

    try:
        while query.isActive:
            progress = query.lastProgress
            if progress:
                logger.info(
                    "silver progress | epoch=%s rows_per_sec=%.1f",
                    progress.get("batchId", "?"),
                    progress.get("processedRowsPerSecond", 0.0),
                )
            query.awaitTermination(timeout=60)
    except KeyboardInterrupt:
        logger.info("Silver job interrupted — stopping query")
        query.stop()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
