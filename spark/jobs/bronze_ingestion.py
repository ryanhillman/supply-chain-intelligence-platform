"""
Bronze Ingestion — Spark Structured Streaming job.

Reads from Kafka (5 topics) and writes raw events to GCS Bronze as Parquet.

Architecture:
  Kafka topics → readStream (one query per topic)
  foreachBatch:
    ├── parse_and_validate()    — JSON parse + required-field check
    ├── valid records           → Bronze GCS Parquet (partitioned)
    └── invalid records         → DLQ (newline-delimited JSON under DLQ_BASE_PATH)

Bronze contract:
  - Raw and immutable — no field transformations beyond type casting
  - May contain duplicates — event_id is the dedup key for Silver
  - Partitioned by event_type / year / month / day (event time; events are day-level)
  - demand.signals: signal_type normalised → event_type column for
    consistent partitioning across all topics
"""
from __future__ import annotations

import logging
import sys
import time

from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as F

sys.path.insert(0, "/opt/spark-app")

from config.bronze_config import (
    BRONZE_BASE_PATH,
    CHECKPOINT_BASE_PATH,
    DLQ_BASE_PATH,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_GROUP_ID,
    STARTING_OFFSETS,
    SOURCE_TOPICS,
    TRIGGER_INTERVAL,
)
from jobs.schemas.inventory_schema import INVENTORY_EVENT_SCHEMA
from jobs.schemas.shipment_schema import SHIPMENT_EVENT_SCHEMA
from jobs.schemas.purchase_order_schema import PURCHASE_ORDER_EVENT_SCHEMA
from jobs.schemas.demand_signal_schema import DEMAND_SIGNAL_SCHEMA
from jobs.schemas.supplier_schema import SUPPLIER_EVENT_SCHEMA
from utils.schema_validator import parse_and_validate
from utils.dlq_handler import write_dlq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("bronze_ingestion")

# ── Topic → schema mapping ─────────────────────────────────────────────────────
TOPIC_SCHEMA_MAP = {
    "inventory.events":       INVENTORY_EVENT_SCHEMA,
    "shipments.events":       SHIPMENT_EVENT_SCHEMA,
    "purchase.orders.events": PURCHASE_ORDER_EVENT_SCHEMA,
    "demand.signals":         DEMAND_SIGNAL_SCHEMA,
    "suppliers.events":       SUPPLIER_EVENT_SCHEMA,
}

# Topics that use `signal_type` instead of `event_type`.
# Normalised to a standard `event_type` column in Bronze for consistent partitioning.
SIGNAL_TYPE_TOPICS: frozenset[str] = frozenset({"demand.signals"})


# ── Spark session ──────────────────────────────────────────────────────────────

def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("scip-bronze-ingestion")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.shuffle.partitions", "6")
        .config("spark.kafka.consumer.cache.enabled", "false")
        # GCS connector (JAR pre-installed in Dockerfile)
        .config("spark.hadoop.fs.gs.impl",
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl",
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        .getOrCreate()
    )


# ── Bronze preparation ─────────────────────────────────────────────────────────

def prepare_bronze_df(validated_df: DataFrame, topic: str) -> DataFrame:
    """
    Prepare a validated DataFrame for Bronze Parquet write.

    Steps:
      1. Normalise event_type — demand.signals uses signal_type; rename to event_type
         so all Bronze topics partition on the same column name.
      2. Extract year/month/day partition columns from timestamp.
      3. Drop internal validation columns (_is_valid, _error_*, _corrupt_record).
      4. Retain kafka_* metadata columns for lineage and DLQ replay.
    """
    df = validated_df

    # Normalise event_type column
    if topic in SIGNAL_TYPE_TOPICS:
        df = df.withColumn("event_type", F.col("signal_type"))

    # Extract time partition columns from ISO-8601 timestamp string
    ts_col = F.to_timestamp(F.col("timestamp"))
    df = (
        df
        .withColumn("year",  F.year(ts_col))
        .withColumn("month", F.month(ts_col))
        .withColumn("day",   F.dayofmonth(ts_col))
    )

    # Drop validation internals — Bronze stores clean parsed records only
    drop_cols = ["_is_valid", "_error_reason", "_error_detail", "_corrupt_record"]
    existing_drop = [c for c in drop_cols if c in df.columns]
    return df.drop(*existing_drop)


# ── Bronze writer ──────────────────────────────────────────────────────────────

def write_bronze(df: DataFrame, topic: str, epoch_id: int) -> None:
    """
    Write a validated DataFrame to Bronze GCS as Parquet.

    Partition layout: {BRONZE_BASE_PATH}/{topic}/event_type={t}/year={y}/month={m}/day={d}/
    Mode: append — Bronze is immutable; duplicates are expected and tolerated.
    """
    output_path = f"{BRONZE_BASE_PATH}/{topic}"
    row_count = df.count()

    (
        df
        .write
        .mode("append")
        .partitionBy("event_type", "year", "month", "day")
        .parquet(output_path)
    )
    logger.info(
        "write_bronze | topic=%s epoch=%d rows=%d path=%s",
        topic, epoch_id, row_count, output_path,
    )


# ── Batch processor factory ────────────────────────────────────────────────────

def make_batch_processor(topic: str, schema):
    """
    Return a foreachBatch function for the given topic.

    Factory pattern (not lambda) ensures each topic correctly captures its own
    `topic` and `schema` variables. Using a lambda in a for-loop would capture
    the loop variable by reference, causing all processors to use the last topic.
    """

    def process_batch(raw_df: DataFrame, epoch_id: int) -> None:
        if raw_df.rdd.isEmpty():
            logger.debug("empty batch | topic=%s epoch=%d", topic, epoch_id)
            return

        # Parse and validate — cached to avoid re-executing the Kafka read DAG
        validated_df = parse_and_validate(raw_df, topic, schema).cache()

        try:
            valid_df   = validated_df.filter(F.col("_is_valid"))
            invalid_df = validated_df.filter(~F.col("_is_valid"))

            valid_count   = valid_df.count()
            invalid_count = invalid_df.count()

            logger.info(
                "batch | topic=%s epoch=%d valid=%d invalid=%d",
                topic, epoch_id, valid_count, invalid_count,
            )

            # Write valid records to Bronze
            if valid_count > 0:
                bronze_df = prepare_bronze_df(valid_df, topic)
                write_bronze(bronze_df, topic, epoch_id)

            # Route invalid records to DLQ
            if invalid_count > 0:
                write_dlq(invalid_df, topic, epoch_id, DLQ_BASE_PATH, KAFKA_GROUP_ID)

        finally:
            validated_df.unpersist()

    return process_batch


# ── Streaming queries ──────────────────────────────────────────────────────────

def build_streaming_queries(spark: SparkSession) -> list:
    """
    Build one streaming query per source topic.

    One query per topic allows independent checkpointing and failure isolation —
    a crash in one topic's query does not affect others.
    """
    queries = []

    for topic in SOURCE_TOPICS:
        schema = TOPIC_SCHEMA_MAP[topic]
        checkpoint_path = f"{CHECKPOINT_BASE_PATH}/{topic}"

        kafka_df = (
            spark.readStream
            .format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
            .option("subscribe", topic)
            .option("startingOffsets", STARTING_OFFSETS)
            .option("groupIdPrefix", KAFKA_GROUP_ID)
            .option("maxOffsetsPerTrigger", "10000")
            .option("failOnDataLoss", "false")
            .option("kafka.session.timeout.ms", "45000")
            .option("kafka.heartbeat.interval.ms", "15000")
            .load()
        )

        processor = make_batch_processor(topic, schema)

        query = (
            kafka_df
            .writeStream
            .foreachBatch(processor)
            .trigger(processingTime=TRIGGER_INTERVAL)
            .option("checkpointLocation", checkpoint_path)
            .queryName(f"bronze-{topic}")
            .start()
        )

        logger.info("Started streaming query | topic=%s checkpoint=%s", topic, checkpoint_path)
        queries.append(query)

    return queries


# ── Status logging ─────────────────────────────────────────────────────────────

def log_status(queries: list) -> None:
    """Log throughput metrics for all active queries."""
    for q in queries:
        if not q.isActive:
            continue
        progress = q.lastProgress
        if progress:
            logger.info(
                "query=%s epoch=%s rows/s=%.1f input_rows=%s",
                q.name,
                progress.get("batchId", "?"),
                progress.get("processedRowsPerSecond", 0.0),
                progress.get("numInputRows", "?"),
            )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info(
        "Bronze ingestion starting | topics=%s bootstrap=%s bronze=%s",
        SOURCE_TOPICS, KAFKA_BOOTSTRAP_SERVERS, BRONZE_BASE_PATH,
    )

    queries = build_streaming_queries(spark)

    try:
        while True:
            time.sleep(60)
            log_status(queries)

            failed = [q for q in queries if not q.isActive]
            if failed:
                for q in failed:
                    exc = q.exception()
                    logger.error("Query failed | name=%s error=%s", q.name, exc)
                raise RuntimeError(f"{len(failed)} streaming query/queries failed")

    except KeyboardInterrupt:
        logger.info("Bronze ingestion interrupted — stopping queries")
    finally:
        for q in queries:
            q.stop()
        spark.stop()


if __name__ == "__main__":
    main()
