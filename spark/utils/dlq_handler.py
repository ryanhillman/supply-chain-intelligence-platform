"""
Dead-letter queue handlers for Spark Structured Streaming.

Two implementations share the same interface:
  FileDlqHandler  — Phase 2: writes DLQ records as JSON to GCS/local path.
  KafkaDlqHandler — Phase 3: produces DLQ records back into Kafka *.dlq topics.

Switch via DLQ_SINK env var ("file" | "kafka").

DLQ envelope fields:
  dlq_id, failed_at, consumer_group, original_topic, original_partition,
  original_offset, original_key, original_payload, error_reason,
  error_detail, epoch_id, replay_eligible
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from pyspark.sql import DataFrame
import pyspark.sql.functions as F

logger = logging.getLogger(__name__)


# ── DLQ topic routing ─────────────────────────────────────────────────────────
# Maps source topic → DLQ topic.
# None means "drop silently" — demand.signals are time-bounded; replaying
# a stale demand signal would corrupt forecasts, so no DLQ is appropriate.
DLQ_TOPIC_MAP: dict[str, str | None] = {
    "inventory.events":       "inventory.events.dlq",
    "purchase.orders.events": "purchase.orders.events.dlq",
    "shipments.events":       "shipments.events.dlq",
    "suppliers.events":       "suppliers.events.dlq",
    "demand.signals":         None,
}


def _build_envelope_df(df: DataFrame, topic: str, epoch_id: int, consumer_group: str) -> DataFrame:
    """
    Transform an invalid-records DataFrame into a DLQ envelope DataFrame.

    Input df must have columns produced by schema_validator.py:
      kafka_topic, kafka_partition, kafka_offset, kafka_key, raw_value,
      _error_reason, _error_detail

    Output df has exactly two columns for the Kafka/file writer:
      key   (binary) — original Kafka message key, preserves partition affinity
      value (string) — JSON-serialised DLQ envelope
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    envelope_df = df.select(
        # Kafka output key — keep original key to maintain topic partition affinity
        F.col("kafka_key").alias("key"),

        # Build envelope as a struct then serialise to JSON
        F.to_json(
            F.struct(
                # Native uuid() (Spark 2.3+) avoids the Python-worker
                # round-trip a UDF would incur for every row.
                F.expr("uuid()").alias("dlq_id"),
                F.lit(now_utc).alias("failed_at"),
                F.lit(consumer_group).alias("consumer_group"),
                F.col("kafka_topic").alias("original_topic"),
                F.col("kafka_partition").alias("original_partition"),
                F.col("kafka_offset").alias("original_offset"),
                # Store key as UTF-8 string; None if key was null
                F.col("kafka_key").cast("string").alias("original_key"),
                # raw_value is the original bytes from Kafka — base64 in JSON
                F.base64(F.col("raw_value")).alias("original_payload"),
                F.col("_error_reason").alias("error_reason"),
                F.col("_error_detail").alias("error_detail"),
                F.lit(epoch_id).alias("epoch_id"),
                # demand.signals are never replay-eligible (handled upstream by
                # dropping them, but included here for envelope completeness)
                F.lit(topic not in ("demand.signals",)).alias("replay_eligible"),
            )
        ).alias("value"),
    )
    return envelope_df


# ── Abstract base ─────────────────────────────────────────────────────────────

class DlqHandler(ABC):
    @abstractmethod
    def write(self, df: DataFrame, topic: str, epoch_id: int) -> None:
        """
        Write invalid records to the dead-letter sink.

        Args:
            df:       DataFrame of invalid records from schema_validator.
            topic:    Source Kafka topic (used for routing to correct DLQ).
            epoch_id: Spark micro-batch epoch (for idempotency tracking).
        """


# ── File DLQ (Phase 2) ────────────────────────────────────────────────────────

class FileDlqHandler(DlqHandler):
    """
    Writes DLQ envelopes as newline-delimited JSON files.

    Output path:
      {base_path}/{topic}/year=YYYY/month=MM/day=DD/hour=HH/

    Works with both local paths (dev) and gs:// URIs (Phase 3+).
    """

    def __init__(self, base_path: str, consumer_group: str) -> None:
        self.base_path = base_path.rstrip("/")
        self.consumer_group = consumer_group

    def write(self, df: DataFrame, topic: str, epoch_id: int) -> None:
        if df.rdd.isEmpty():
            return

        now = datetime.now(timezone.utc)
        partition_path = (
            f"{self.base_path}/{topic}"
            f"/year={now.year}"
            f"/month={now.month:02d}"
            f"/day={now.day:02d}"
            f"/hour={now.hour:02d}"
        )

        envelope_df = _build_envelope_df(df, topic, epoch_id, self.consumer_group)

        # Write value column only — key not meaningful for file sink
        (
            envelope_df
            .select("value")
            .write
            .mode("append")
            .text(partition_path)
        )
        logger.info(
            "FileDlqHandler wrote %d records | topic=%s epoch=%d path=%s",
            df.count(), topic, epoch_id, partition_path,
        )


# ── Kafka DLQ (Phase 3) ───────────────────────────────────────────────────────

class KafkaDlqHandler(DlqHandler):
    """
    Produces DLQ envelopes as JSON messages back into Kafka *.dlq topics.

    Routing (DLQ_TOPIC_MAP):
      inventory.events       → inventory.events.dlq
      purchase.orders.events → purchase.orders.events.dlq
      shipments.events       → shipments.events.dlq
      suppliers.events       → suppliers.events.dlq
      demand.signals         → None (dropped — time-bounded, replay harmful)

    Each DLQ message:
      key   = original Kafka message key (bytes)
      value = JSON-serialised DLQ envelope (UTF-8 string → Kafka bytes)

    The DLQ topics use Schema Registry NONE compatibility so the envelope
    schema can be amended during incidents without compatibility checks.
    """

    def __init__(self, bootstrap_servers: str, consumer_group: str) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.consumer_group = consumer_group

    def write(self, df: DataFrame, topic: str, epoch_id: int) -> None:
        dlq_topic = DLQ_TOPIC_MAP.get(topic)

        if dlq_topic is None:
            # demand.signals: log and drop — no replay value for stale signals
            count = df.count()
            if count > 0:
                logger.warning(
                    "KafkaDlqHandler dropping %d invalid records from %s "
                    "(topic has no DLQ — time-bounded signal)",
                    count, topic,
                )
            return

        if df.rdd.isEmpty():
            return

        envelope_df = _build_envelope_df(df, topic, epoch_id, self.consumer_group)

        try:
            (
                envelope_df
                .write
                .format("kafka")
                .option("kafka.bootstrap.servers", self.bootstrap_servers)
                .option("topic", dlq_topic)
                # Idempotent producer: exactly-once delivery to DLQ topic.
                # Requires broker acks=all and min.insync.replicas >= 1.
                .option("kafka.enable.idempotence", "true")
                .option("kafka.acks", "all")
                .option("kafka.retries", "3")
                .option("kafka.retry.backoff.ms", "500")
                .save()
            )
            logger.info(
                "KafkaDlqHandler produced records | source=%s dlq=%s epoch=%d",
                topic, dlq_topic, epoch_id,
            )
        except Exception as exc:
            # DLQ write failure must not crash the main streaming query.
            # Log with full context so ops can replay from Bronze if needed.
            logger.error(
                "KafkaDlqHandler FAILED to write to %s (epoch=%d): %s. "
                "Records lost — replay from Bronze path using epoch_id=%d.",
                dlq_topic, epoch_id, exc, epoch_id,
            )


# ── Factory ───────────────────────────────────────────────────────────────────

def get_dlq_handler(config) -> DlqHandler:
    """
    Return the appropriate DlqHandler based on DLQ_SINK config value.

    Args:
        config: bronze_config module (or any object with the required attrs).

    Returns:
        FileDlqHandler  if config.DLQ_SINK == "file"
        KafkaDlqHandler if config.DLQ_SINK == "kafka"

    Raises:
        ValueError: if DLQ_SINK is not "file" or "kafka".
    """
    sink = config.DLQ_SINK.lower()

    if sink == "file":
        return FileDlqHandler(
            base_path=config.DLQ_BASE_PATH,
            consumer_group=config.KAFKA_GROUP_ID,
        )

    if sink == "kafka":
        return KafkaDlqHandler(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            consumer_group=config.KAFKA_GROUP_ID,
        )

    raise ValueError(
        f"Unknown DLQ_SINK={config.DLQ_SINK!r}. Must be 'file' or 'kafka'."
    )
