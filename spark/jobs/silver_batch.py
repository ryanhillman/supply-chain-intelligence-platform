"""
Silver job: one feature snapshot per (warehouse_id, sku_id) as of a date,
written to Silver Parquet and then to the Bigtable feature store.

The features describe the end of the as-of day; Bronze events after it are
never read (spark/jobs/silver_features.py). --as-of defaults to the latest
day with inventory events in Bronze, which after a replay is the last
simulated day.

Usage (inside the scip-spark container; see spark/docker-compose.spark.yml):
    spark-submit ... jobs/silver_batch.py [--as-of YYYY-MM-DD]
    # or: docker compose -f docker-compose.spark.yml run --rm \
    #         -e SILVER_AS_OF=2026-09-30 spark-silver-batch

Writes:
  Silver    {SILVER_BASE_PATH}/warehouse_id=.../year=/month=/day=  (as-of date)
            Rerunning the same as-of date replaces those partitions.
  Bigtable  one row per pair, best-effort after the Silver write: a failed
            Bigtable write leaves Silver intact, and rerunning the job
            repopulates Bigtable. lineage:silver_ts is the wall-clock write
            time, which the scorer's freshness check reads.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone

from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

sys.path.insert(0, "/opt/spark-app")

from config.bronze_config import (
    BIGTABLE_INSTANCE_ID,
    BIGTABLE_TABLE_ID,
    BRONZE_BASE_PATH,
    GCP_PROJECT_ID,
    SILVER_BASE_PATH,
)
from config.feature_config import PIPELINE_VERSION, ROW_KEY_SEPARATOR
from jobs.schemas.feature_schema import (
    SILVER_DEDUP_COLS,
    SILVER_PARTITION_COLS,
    conform_to_silver_schema,
)
from jobs.silver_features import compute_features, latest_event_date
from utils.bigtable_writer import write_to_bigtable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("silver_batch")

# Written to lineage:bronze_epoch. A batch run has no streaming epoch; the
# constant marks rows as written by this job.
BATCH_EPOCH_ID: int = 9000


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("scip-silver-batch")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.shuffle.partitions", "6")
        # Day boundaries in the feature windows are UTC midnights.
        .config("spark.sql.session.timeZone", "UTC")
        # mode("overwrite") replaces only the partitions being written.
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .config("spark.hadoop.fs.gs.impl",
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl",
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        .getOrCreate()
    )


def add_lineage(df: DataFrame, epoch_id: int, event_count: int) -> DataFrame:
    """
    Provenance columns. event_count is the number of inventory events this
    run read (the 30-day window), the same value on every row.
    """
    return (
        df
        .withColumn("lineage__last_bronze_epoch", F.lit(epoch_id).cast("long"))
        .withColumn("lineage__last_silver_write", F.lit(datetime.now(timezone.utc).isoformat()))
        .withColumn("lineage__source_event_count", F.lit(event_count).cast("long"))
        .withColumn("lineage__pipeline_version", F.lit(PIPELINE_VERSION))
    )


def add_row_key_and_partitions(df: DataFrame, as_of: date) -> DataFrame:
    """Bigtable row key, the snapshot time (end of the as-of day), and Silver partitions."""
    window_end = (as_of + timedelta(days=1)).isoformat()
    return (
        df
        .withColumn("row_key", F.concat_ws(ROW_KEY_SEPARATOR, F.col("warehouse_id"), F.col("sku_id")))
        .withColumn("window_end", F.lit(window_end).cast("timestamp"))
        .withColumn("year",  F.lit(as_of.year))
        .withColumn("month", F.lit(as_of.month))
        .withColumn("day",   F.lit(as_of.day))
    )


def write_silver(feature_df: DataFrame, epoch_id: int) -> int:
    """Write the snapshot to Silver Parquet, replacing the as-of date's partitions."""
    deduped = feature_df.dropDuplicates(SILVER_DEDUP_COLS)
    row_count = deduped.count()
    if row_count:
        deduped.write.mode("overwrite").partitionBy(*SILVER_PARTITION_COLS).parquet(SILVER_BASE_PATH)
    logger.info("write_silver | epoch=%d rows=%d path=%s", epoch_id, row_count, SILVER_BASE_PATH)
    return row_count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Silver features as of a date")
    p.add_argument(
        "--as-of",
        default=os.environ.get("SILVER_AS_OF") or None,
        help="Feature date YYYY-MM-DD (default: latest day with inventory events in Bronze)",
    )
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    as_of = date.fromisoformat(args.as_of) if args.as_of else latest_event_date(spark, BRONZE_BASE_PATH)
    if as_of is None:
        logger.error("No inventory events in %s — has the Bronze job run?", BRONZE_BASE_PATH)
        spark.stop()
        sys.exit(1)
    logger.info("Silver batch starting | as_of=%s bronze=%s silver=%s",
                as_of, BRONZE_BASE_PATH, SILVER_BASE_PATH)

    features, event_count = compute_features(spark, BRONZE_BASE_PATH, as_of)
    if event_count == 0:
        logger.error("No inventory events in the 30 days up to %s", as_of)
        spark.stop()
        sys.exit(1)

    feature_df = conform_to_silver_schema(add_row_key_and_partitions(
        add_lineage(features, BATCH_EPOCH_ID, event_count), as_of
    )).cache()
    try:
        write_silver(feature_df, BATCH_EPOCH_ID)
        try:
            write_to_bigtable(
                df=feature_df,
                project_id=GCP_PROJECT_ID,
                instance_id=BIGTABLE_INSTANCE_ID,
                table_id=BIGTABLE_TABLE_ID,
                epoch_id=BATCH_EPOCH_ID,
            )
        except Exception as bt_exc:
            logger.error(
                "Bigtable write FAILED | error=%s | Silver succeeded — rerun this job "
                "to repopulate Bigtable.",
                bt_exc,
            )
    finally:
        feature_df.unpersist()
        spark.stop()

    logger.info("Silver batch complete | as_of=%s", as_of)


if __name__ == "__main__":
    main()
