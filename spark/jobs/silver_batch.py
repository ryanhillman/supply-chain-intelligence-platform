"""
Silver job: one feature snapshot per (warehouse_id, sku_id) and as-of date,
written to Silver Parquet; the last date's snapshot is then written to the
Bigtable feature store.

The features describe the end of the as-of day; Bronze events after it are
never read (spark/jobs/silver_features.py). --as-of defaults to the latest
day with inventory events in Bronze, which after a replay is the last
simulated day. --from adds every earlier day back to that date, so one run
writes the history the training set is built from.

Usage (inside the scip-spark container; see spark/docker-compose.spark.yml):
    spark-submit ... jobs/silver_batch.py [--as-of YYYY-MM-DD] [--from YYYY-MM-DD]
    # or: docker compose -f docker-compose.spark.yml run --rm \
    #         -e SILVER_AS_OF=2026-09-30 spark-silver-batch

Writes:
  Silver    {SILVER_BASE_PATH}/warehouse_id=.../year=/month=/day=  (as-of date)
            Rerunning the same as-of date replaces those partitions.
  Bigtable  one row per pair for the last as-of date, after the Silver write.
            A failed Bigtable write leaves Silver intact and fails the job;
            rerunning it repopulates Bigtable. lineage:silver_ts is the
            wall-clock write time, which the scorer's freshness check reads.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timezone

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
    SILVER_PARTITION_COLS,
    conform_to_silver_schema,
)
from jobs.silver_features import bronze_date_range, compute_features
from utils.bigtable_writer import write_to_bigtable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("silver_batch")


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
        # GCS connector settings come from config/spark-gcs.conf (--properties-file).
        .getOrCreate()
    )


def add_lineage(df: DataFrame) -> DataFrame:
    """Provenance columns: the wall-clock write time and the pipeline version."""
    return (
        df
        .withColumn("lineage__last_silver_write", F.lit(datetime.now(timezone.utc).isoformat()))
        .withColumn("lineage__pipeline_version", F.lit(PIPELINE_VERSION))
    )


def add_row_key_and_partitions(df: DataFrame) -> DataFrame:
    """Bigtable row key, the snapshot time (end of the as-of day), and Silver partitions."""
    as_of = F.col("as_of")
    return (
        df
        .withColumn("row_key", F.concat_ws(ROW_KEY_SEPARATOR, F.col("warehouse_id"), F.col("sku_id")))
        .withColumn("window_end", F.date_add(as_of, 1).cast("timestamp"))
        .withColumn("year",  F.year(as_of))
        .withColumn("month", F.month(as_of))
        .withColumn("day",   F.dayofmonth(as_of))
    )


def on_date(d: date):
    """Rows of the Silver layout whose as-of date is `d`."""
    return (F.col("year") == d.year) & (F.col("month") == d.month) & (F.col("day") == d.day)


def write_silver(feature_df: DataFrame) -> int:
    """
    Write the snapshots to Silver Parquet, replacing the partitions of the
    as-of dates written. Rows are repartitioned by the partition columns
    first, so each directory gets one file.
    """
    row_count = feature_df.count()
    if row_count:
        (
            feature_df
            .repartition(*SILVER_PARTITION_COLS)
            .write.mode("overwrite").partitionBy(*SILVER_PARTITION_COLS).parquet(SILVER_BASE_PATH)
        )
    logger.info("write_silver | rows=%d path=%s", row_count, SILVER_BASE_PATH)
    return row_count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Silver features as of a date or a range of dates")
    p.add_argument(
        "--as-of",
        default=os.environ.get("SILVER_AS_OF") or None,
        help="Feature date YYYY-MM-DD (default: latest day with inventory events in Bronze)",
    )
    p.add_argument(
        "--from",
        dest="first",
        default=os.environ.get("SILVER_FROM") or None,
        help="Also write every day from this date YYYY-MM-DD up to --as-of (default: --as-of only)",
    )
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    bronze_days = None if args.as_of else bronze_date_range(spark, BRONZE_BASE_PATH)
    if not args.as_of and bronze_days is None:
        logger.error("No inventory events in %s — has the Bronze job run?", BRONZE_BASE_PATH)
        spark.stop()
        sys.exit(1)
    last  = date.fromisoformat(args.as_of) if args.as_of else bronze_days[1]
    first = date.fromisoformat(args.first) if args.first else last
    if first > last:
        logger.error("--from %s is after --as-of %s", first, last)
        spark.stop()
        sys.exit(1)
    logger.info("Silver batch starting | from=%s as_of=%s bronze=%s silver=%s",
                first, last, BRONZE_BASE_PATH, SILVER_BASE_PATH)

    # Cached before the schema check, whose null count is the first action on these
    # rows; the Silver and Bigtable writes then read the cache, not Bronze again.
    cached_df = add_row_key_and_partitions(
        add_lineage(compute_features(spark, BRONZE_BASE_PATH, first, last))
    ).cache()
    try:
        feature_df = conform_to_silver_schema(cached_df)
        row_count = write_silver(feature_df)
        if row_count == 0:
            logger.error("No inventory events in the 30 days up to %s", last)
            sys.exit(1)
        latest = feature_df if first == last else feature_df.filter(on_date(last))
        try:
            write_to_bigtable(
                df=latest,
                project_id=GCP_PROJECT_ID,
                instance_id=BIGTABLE_INSTANCE_ID,
                table_id=BIGTABLE_TABLE_ID,
                row_count=row_count if first == last else latest.count(),
            )
        except Exception as bt_exc:
            logger.error(
                "Bigtable write FAILED | error=%s | Silver succeeded — rerun this job "
                "to repopulate Bigtable.",
                bt_exc,
            )
            raise   # exit non-zero: the scorer would read an out-of-date store
    finally:
        cached_df.unpersist()
        spark.stop()

    logger.info("Silver batch complete | from=%s as_of=%s", first, last)


if __name__ == "__main__":
    main()
