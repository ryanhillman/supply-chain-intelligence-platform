"""
Schema validation for raw Kafka messages in the Bronze ingestion layer.

parse_and_validate() takes a raw Kafka DataFrame (value as binary) and returns
a DataFrame with:
  - All original Kafka metadata columns retained for DLQ/lineage
  - Parsed payload fields (from_json with PERMISSIVE mode)
  - Validation flags: _is_valid, _error_reason, _error_detail

Error reasons:
  UNPARSEABLE_JSON     — value bytes could not be parsed as JSON
  MISSING_REQUIRED_FIELD — parsed OK but a required field is null
"""
from __future__ import annotations

import logging
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, StructField, StructType
import pyspark.sql.functions as F

from config.bronze_config import TOPIC_REQUIRED_FIELDS

logger = logging.getLogger(__name__)


def parse_and_validate(
    raw_df: DataFrame,
    topic: str,
    payload_schema: StructType,
) -> DataFrame:
    """
    Parse raw Kafka bytes and validate required fields.

    Args:
        raw_df:         Raw Kafka DataFrame with binary `value` column.
        topic:          Source topic name (used to look up required fields).
        payload_schema: PySpark StructType for the topic's payload.

    Returns:
        DataFrame with columns:
          kafka_topic, kafka_partition, kafka_offset, kafka_timestamp,
          kafka_key, raw_value,
          <all payload fields>,
          _corrupt_record, _is_valid, _error_reason, _error_detail
    """
    required_fields = TOPIC_REQUIRED_FIELDS.get(topic, [])

    # Extend schema with _corrupt_record for PERMISSIVE parse mode.
    # PERMISSIVE sets all fields to null and populates _corrupt_record
    # with the raw string when JSON is malformed.
    extended_schema = StructType(
        payload_schema.fields + [StructField("_corrupt_record", StringType(), nullable=True)]
    )

    # Retain Kafka metadata for DLQ envelopes and lineage.
    # raw_value kept as bytes so DLQ can base64-encode the original payload.
    with_meta = raw_df.select(
        F.col("topic").alias("kafka_topic"),
        F.col("partition").alias("kafka_partition"),
        F.col("offset").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_timestamp"),
        F.col("key").alias("kafka_key"),
        F.col("value").alias("raw_value"),
        F.col("value").cast("string").alias("_raw_json"),
    )

    # Parse JSON payload using PERMISSIVE mode.
    parsed = with_meta.withColumn(
        "_parsed",
        F.from_json(F.col("_raw_json"), extended_schema, {"mode": "PERMISSIVE"}),
    )

    # Expand struct fields to top-level columns.
    payload_cols = [F.col(f"_parsed.{f.name}") for f in payload_schema.fields]
    expanded = parsed.select(
        "kafka_topic",
        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp",
        "kafka_key",
        "raw_value",
        *payload_cols,
        F.col("_parsed._corrupt_record").alias("_corrupt_record"),
    )

    # ── Validation ────────────────────────────────────────────────────────────

    # UNPARSEABLE_JSON: _corrupt_record is non-null (PERMISSIVE mode populated it)
    is_parse_error = F.col("_corrupt_record").isNotNull()

    # MISSING_REQUIRED_FIELD: any required field is null after a successful parse
    if required_fields:
        first_null_field = F.coalesce(
            *[F.when(F.col(f).isNull(), F.lit(f)) for f in required_fields]
        )
    else:
        first_null_field = F.lit(None).cast("string")

    validated = expanded.withColumn(
        "_error_reason",
        F.when(is_parse_error, F.lit("UNPARSEABLE_JSON"))
         .when(~is_parse_error & first_null_field.isNotNull(), F.lit("MISSING_REQUIRED_FIELD"))
         .otherwise(F.lit(None).cast("string")),
    ).withColumn(
        "_error_detail",
        F.when(is_parse_error, F.col("_corrupt_record"))
         .when(~is_parse_error & first_null_field.isNotNull(),
               F.concat(F.lit("null field: "), first_null_field))
         .otherwise(F.lit(None).cast("string")),
    ).withColumn(
        "_is_valid",
        F.col("_error_reason").isNull(),
    )

    return validated
