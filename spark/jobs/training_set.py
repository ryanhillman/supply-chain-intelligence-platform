"""
Training set job: Silver's feature snapshots joined with the stockout label,
one row per (warehouse_id, sku_id, sample_date).

For a snapshot of day D the label is 1 when any inventory event of the pair
on days D+1..D+HORIZON_DAYS leaves the stock at zero. The label needs
HORIZON_DAYS of events after D, and the 30-day feature windows need
MIN_HISTORY_DAYS of events before it, so the sample dates run from the first
day in Bronze plus MIN_HISTORY_DAYS to the last day minus HORIZON_DAYS.

Silver must hold a snapshot for every sample date; a missing date fails the
job. Write them first with:
    spark-submit ... jobs/silver_batch.py --from <first sample date>

Usage (inside the scip-spark container; see spark/docker-compose.spark.yml):
    spark-submit ... jobs/training_set.py
    # or: docker compose -f docker-compose.spark.yml run --rm spark-training-set

Writes {TRAINING_BASE_PATH} as one Parquet file, replaced on each run: the
ids, sample_date, the model's stored features under the names in
feature_registry.RAW_FEATURE_COLS (null where Silver is null), and
stockout_label_7d.
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta

from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

sys.path.insert(0, "/opt/spark-app")

from config.bronze_config import BRONZE_BASE_PATH, SILVER_BASE_PATH, TRAINING_BASE_PATH
from config.feature_config import MIN_HISTORY_DAYS
from feature_registry import FEATURE_REGISTRY, HORIZON_DAYS, RAW_FEATURE_COLS
from jobs.schemas.feature_schema import SILVER_FEATURE_SCHEMA
from jobs.schemas.inventory_schema import INVENTORY_EVENT_SCHEMA
from jobs.silver_features import PAIR, bronze_date_range, read_events

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("training_set")

LABEL_COL = f"stockout_label_{HORIZON_DAYS}d"
ROW_KEY = [*PAIR, "sample_date"]

# Silver column of each stored model feature, e.g. velocity__sales_7d -> sales_7d.
_SILVER_COLUMN = {
    feature: f"{family}__{feature}"
    for family, features in FEATURE_REGISTRY.items()
    for feature in features
}


def compute_labels(spark: SparkSession, bronze_base: str, first: date, last: date) -> DataFrame:
    """
    stockout_label_7d for the sample dates `first`..`last`: 1 when an inventory
    event of the pair on days D+1..D+HORIZON_DAYS has quantity_after == 0.

    An event on day E falls in the label window of every sample date D with
    E - HORIZON_DAYS <= D <= E - 1, so it is counted once for each such D in
    the range. A pair and date with no event in its window has no row.
    """
    events = read_events(
        spark, bronze_base, "inventory.events", INVENTORY_EVENT_SCHEMA,
        first + timedelta(days=1), last + timedelta(days=HORIZON_DAYS + 1),
    )
    event_date = F.to_date(F.col("timestamp").cast("timestamp"))
    sample_dates = F.sequence(
        F.greatest(F.date_sub(event_date, HORIZON_DAYS), F.lit(first)),
        F.least(F.date_sub(event_date, 1), F.lit(last)),
    )
    return (
        events.withColumn("sample_date", F.explode(sample_dates))
        .groupBy(*ROW_KEY)
        .agg(F.max((F.col("quantity_after") == 0).cast("long")).alias(LABEL_COL))
    )


def training_rows(
    spark: SparkSession, silver_base: str, bronze_base: str, first: date, last: date
) -> DataFrame:
    """Silver's stored model features for the sample dates `first`..`last`, with the label."""
    sample_date = F.make_date(F.col("year"), F.col("month"), F.col("day"))
    features = (
        spark.read.schema(SILVER_FEATURE_SCHEMA).parquet(silver_base)
        .withColumn("sample_date", sample_date)
        .filter((F.col("sample_date") >= F.lit(first)) & (F.col("sample_date") <= F.lit(last)))
        .select(*ROW_KEY, *[F.col(_SILVER_COLUMN[name]).alias(name) for name in RAW_FEATURE_COLS])
    )
    labels = compute_labels(spark, bronze_base, first, last)
    return features.join(labels, ROW_KEY, "left").fillna(0, [LABEL_COL])


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("scip-training-set")
        .config("spark.sql.shuffle.partitions", "6")
        # Day boundaries in the label windows are UTC midnights.
        .config("spark.sql.session.timeZone", "UTC")
        # GCS connector settings come from config/spark-gcs.conf (--properties-file).
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        bronze_days = bronze_date_range(spark, BRONZE_BASE_PATH)
        if bronze_days is None:
            logger.error("No inventory events in %s — has the Bronze job run?", BRONZE_BASE_PATH)
            sys.exit(1)
        first = bronze_days[0] + timedelta(days=MIN_HISTORY_DAYS)
        last  = bronze_days[1] - timedelta(days=HORIZON_DAYS)
        n_dates = (last - first).days + 1
        if n_dates < 1:
            logger.error(
                "Bronze spans %s..%s; a sample date needs %d days before it and %d after",
                bronze_days[0], bronze_days[1], MIN_HISTORY_DAYS, HORIZON_DAYS,
            )
            sys.exit(1)
        logger.info("Training set starting | sample dates %s..%s silver=%s", first, last, SILVER_BASE_PATH)

        rows = training_rows(spark, SILVER_BASE_PATH, BRONZE_BASE_PATH, first, last).cache()
        dates_found = rows.select("sample_date").distinct().count()
        if dates_found != n_dates:
            logger.error(
                "Silver holds %d of the %d sample dates %s..%s — run silver_batch.py --from %s first",
                dates_found, n_dates, first, last, first,
            )
            sys.exit(1)

        rows.coalesce(1).write.mode("overwrite").parquet(TRAINING_BASE_PATH)
        logger.info(
            "Training set complete | rows=%d positive=%d path=%s",
            rows.count(), rows.filter(F.col(LABEL_COL) == 1).count(), TRAINING_BASE_PATH,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
