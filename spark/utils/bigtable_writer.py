"""
Bigtable feature store writer for the Silver layer dual-write.

Writes Silver feature DataFrames to Bigtable using the Google Cloud Bigtable
Python client's mutate_rows batch API.

Design:
  - Called from foreachBatch inside silver_feature_engineering.py
  - Uses foreachPartition so each Spark executor opens its own Bigtable client
    (avoids serialising the client object across the network)
  - Values encoded as big-endian double (numeric) or UTF-8 bytes (string)
  - Cell timestamp = current wall-clock time so GC max_age policies work
    correctly against write time, not window_end time
  - Batch size controlled by BIGTABLE_BATCH_SIZE (default 500)

Row key format: {warehouse_id}#{sku_id}
  Key design documented in docs/adr/ADR-003-bigtable-feature-store.md.
  Bigtable splits tablets automatically; the table has no pre-split points.
"""
from __future__ import annotations

import logging
import struct
from datetime import datetime, timezone
from typing import Iterator

from pyspark.sql import DataFrame
from pyspark.sql.types import Row

from config.feature_config import (
    BIGTABLE_BATCH_SIZE,
    FEATURE_REGISTRY,
    NULLABLE_FEATURES,
    STRING_FEATURES,
)

logger = logging.getLogger(__name__)

# ── Column → (family, qualifier) lookup ───────────────────────────────────────
# Built once at import time from FEATURE_REGISTRY.
# Key: Silver DataFrame column name  e.g. "velocity__sales_7d"
# Val: (family, qualifier)           e.g. ("velocity", "sales_7d")
_COL_TO_FAMILY_QUALIFIER: dict[str, tuple[str, str]] = {
    f"{family}__{attr}": (family, qualifier)
    for family, qualifiers in FEATURE_REGISTRY.items()
    for attr, qualifier in qualifiers.items()
}


# ── Value encoding ────────────────────────────────────────────────────────────

def _encode_value(col_name: str, value) -> bytes | None:
    """
    Encode a feature value to bytes for Bigtable storage.

    Numeric features: big-endian IEEE 754 double (8 bytes).
    String features:  UTF-8 bytes.
    None/null:        returns None → caller skips the cell.
    """
    if value is None:
        return None

    if col_name in STRING_FEATURES:
        return str(value).encode("utf-8")

    try:
        return struct.pack(">d", float(value))
    except (TypeError, ValueError) as exc:
        logger.warning("Cannot encode %s=%r as double: %s — skipping cell", col_name, value, exc)
        return None


# ── Per-partition writer ───────────────────────────────────────────────────────

def _write_partition(
    rows_iter: Iterator[Row],
    project_id: str,
    instance_id: str,
    table_id: str,
) -> None:
    """
    Write one Spark partition to Bigtable.

    Instantiates a Bigtable client per partition (not per row) to amortise
    connection overhead. Closed in a finally block to prevent connection leaks.
    """
    # Import here — runs inside Spark executor, not the driver
    from google.cloud import bigtable  # type: ignore[import]

    client = bigtable.Client(project=project_id, admin=False)
    table = client.instance(instance_id).table(table_id)

    batch: list = []
    rows_written = 0
    rows_failed = 0

    try:
        for spark_row in rows_iter:
            row_key: str = spark_row["row_key"]
            bt_row = table.direct_row(row_key.encode("utf-8"))

            # Timestamp applied to every cell in this row snapshot.
            # Using wall-clock time so max_age GC policies trigger on write time.
            cell_ts = datetime.now(timezone.utc)

            for col_name, (family, qualifier) in _COL_TO_FAMILY_QUALIFIER.items():
                # Skip nullable features whose value is None
                if col_name in NULLABLE_FEATURES and spark_row[col_name] is None:
                    continue

                value_bytes = _encode_value(col_name, spark_row[col_name])
                if value_bytes is None:
                    continue

                bt_row.set_cell(
                    column_family_id=family,
                    column=qualifier.encode("utf-8"),
                    value=value_bytes,
                    timestamp=cell_ts,
                )

            batch.append(bt_row)

            if len(batch) >= BIGTABLE_BATCH_SIZE:
                rows_written, rows_failed = _flush_batch(table, batch, rows_written, rows_failed)
                batch = []

        # Flush remaining rows
        if batch:
            rows_written, rows_failed = _flush_batch(table, batch, rows_written, rows_failed)

    finally:
        client.close()

    logger.info(
        "Bigtable partition done | written=%d failed=%d instance=%s table=%s",
        rows_written, rows_failed, instance_id, table_id,
    )


def _flush_batch(table, batch: list, rows_written: int, rows_failed: int) -> tuple[int, int]:
    """
    Call mutate_rows and handle per-row status codes.

    mutate_rows returns a list of MutateRowsResponse entries — one per row.
    Each entry has a .code (gRPC status); 0 = OK.
    """
    try:
        responses = table.mutate_rows(batch)
    except Exception as exc:
        logger.error("mutate_rows RPC failed for batch of %d rows: %s", len(batch), exc)
        return rows_written, rows_failed + len(batch)

    for i, response in enumerate(responses):
        if response.code != 0:
            row_key = batch[i].row_key.decode("utf-8") if batch[i].row_key else "unknown"
            logger.error(
                "Bigtable row mutation failed | row_key=%s code=%d message=%s",
                row_key, response.code, response.message,
            )
            rows_failed += 1
        else:
            rows_written += 1

    return rows_written, rows_failed


# ── Public API ─────────────────────────────────────────────────────────────────

def write_to_bigtable(
    df: DataFrame,
    project_id: str,
    instance_id: str,
    table_id: str,
    epoch_id: int,
) -> None:
    """
    Write a Silver feature DataFrame to Bigtable.

    Called from foreachBatch in silver_feature_engineering.py.

    Args:
        df:          Silver feature DataFrame (conforms to SILVER_FEATURE_SCHEMA).
        project_id:  GCP project ID.
        instance_id: Bigtable instance ID (e.g. "scip-feature-store").
        table_id:    Bigtable table ID (e.g. "inventory_features").
        epoch_id:    Spark micro-batch epoch (logged for observability).

    The DataFrame must contain:
      - row_key  (string): pre-computed as "{warehouse_id}#{sku_id}"
      - All feature columns named {family}__{attr} per FEATURE_REGISTRY
    """
    if df.rdd.isEmpty():
        logger.debug("write_to_bigtable: empty DataFrame for epoch=%d — skipping", epoch_id)
        return

    row_count = df.count()
    logger.info(
        "write_to_bigtable | epoch=%d rows=%d instance=%s table=%s",
        epoch_id, row_count, instance_id, table_id,
    )

    df.foreachPartition(
        lambda partition: _write_partition(partition, project_id, instance_id, table_id)
    )
