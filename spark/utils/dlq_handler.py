"""
Dead-letter queue writer for the Bronze ingestion job.

Invalid records (JSON that fails parsing or is missing a required field) are
wrapped in a DLQ envelope and written as newline-delimited JSON under
DLQ_BASE_PATH/{topic}/year=/month=/day=/hour=/ — a local path in the local
stack, a gs:// prefix when running against GCS. The envelope keeps the
original Kafka coordinates and the raw payload (base64) so a record can be
inspected or replayed by hand.

Envelope fields:
  dlq_id, failed_at, consumer_group, original_topic, original_partition,
  original_offset, original_key, original_payload, error_reason,
  error_detail, epoch_id, replay_eligible

An earlier version also had a Kafka sink producing to *.dlq topics behind a
DLQ_SINK switch. It was never selected — the simulator always emits the
required fields, so the DLQ sees nothing in practice — and the extra topics,
handler classes and factory were removed.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

logger = logging.getLogger(__name__)

# demand.signals are time-bounded: replaying a stale signal would corrupt
# forecasts, so their envelopes are marked replay_eligible=False.
_NOT_REPLAYABLE: frozenset[str] = frozenset({"demand.signals"})


def write_dlq(
    df: DataFrame,
    topic: str,
    epoch_id: int,
    base_path: str,
    consumer_group: str,
) -> None:
    """
    Write the invalid records in `df` as DLQ envelopes under base_path/topic/.

    Args:
        df:             Invalid records from schema_validator (kafka_* columns,
                        raw_value, _error_reason, _error_detail).
        topic:          Source Kafka topic.
        epoch_id:       Spark micro-batch epoch, kept for idempotency tracking.
        base_path:      DLQ_BASE_PATH (local path or gs:// prefix).
        consumer_group: Kafka consumer group of the ingesting job.
    """
    if df.rdd.isEmpty():
        return

    now = datetime.now(timezone.utc)
    partition_path = (
        f"{base_path.rstrip('/')}/{topic}"
        f"/year={now.year}/month={now.month:02d}/day={now.day:02d}/hour={now.hour:02d}"
    )
    envelope_df = _build_envelope_df(df, topic, epoch_id, consumer_group)
    envelope_df.select("value").write.mode("append").text(partition_path)
    logger.info(
        "write_dlq | topic=%s epoch=%d records=%d path=%s",
        topic, epoch_id, df.count(), partition_path,
    )


def _build_envelope_df(df: DataFrame, topic: str, epoch_id: int, consumer_group: str) -> DataFrame:
    """
    Transform an invalid-records DataFrame into a DLQ envelope DataFrame with
    one JSON string column, `value`.

    Input df must have the columns produced by schema_validator.py:
      kafka_topic, kafka_partition, kafka_offset, kafka_key, raw_value,
      _error_reason, _error_detail
    """
    now_utc = datetime.now(timezone.utc).isoformat()
    return df.select(
        F.to_json(
            F.struct(
                # Native uuid() avoids a Python-worker round-trip per row.
                F.expr("uuid()").alias("dlq_id"),
                F.lit(now_utc).alias("failed_at"),
                F.lit(consumer_group).alias("consumer_group"),
                F.col("kafka_topic").alias("original_topic"),
                F.col("kafka_partition").alias("original_partition"),
                F.col("kafka_offset").alias("original_offset"),
                F.col("kafka_key").cast("string").alias("original_key"),
                F.base64(F.col("raw_value")).alias("original_payload"),
                F.col("_error_reason").alias("error_reason"),
                F.col("_error_detail").alias("error_detail"),
                F.lit(epoch_id).alias("epoch_id"),
                F.lit(topic not in _NOT_REPLAYABLE).alias("replay_eligible"),
            )
        ).alias("value"),
    )
