"""
prediction_writer.py — consumes Pub/Sub prediction messages and writes to Bigtable.

Design
------
- Pulls from the "prediction-writer" subscription on the "prediction-results" topic.
- For each batch of messages:
    1. Parse JSON payload.
    2. Build Bigtable mutations for the `predictions` column family.
    3. Write all mutations in one mutate_rows() call.
    4. Acknowledge Pub/Sub messages ONLY after successful Bigtable write.
    5. Leave failed messages un-acknowledged so Pub/Sub redelivers them.
- After 5 delivery attempts, the Pub/Sub subscription's dead-letter policy
  forwards messages to "prediction-results-dead-letter" (configured in pubsub.tf).

Bigtable predictions column family (one row per entity, same key as feature rows)
----------------------------------------------------------------------------------
Row key   : {warehouse_id}#{sku_id}  (matches inventory_features key)
Family    : predictions
Columns:
  shortage_prob_7d      — UTF-8 string, e.g. "0.735123"
  stockout_risk_score   — UTF-8 string, e.g. "0.682411"
  model_version         — UTF-8 string, e.g. "v20260620_180918"
  prediction_timestamp  — RFC3339 UTC string, e.g. "2026-06-20T18:09:21Z"

GC policy (configured in infrastructure/gcp/main.tf):
  max_version { number = 3 }  — retains current + 2 prior rounds for rollback.

Modes
-----
- Batch mode (default): pull until the subscription is empty, then exit.
  Use for Cloud Run jobs: the job finishes naturally after scoring is done.
- Streaming mode (--streaming): run indefinitely using the Pub/Sub streaming pull.
  Use for a long-lived Cloud Run service or GKE workload.

Error handling
--------------
- Transient Bigtable errors: exponential backoff retry up to max_bt_retries=3.
- Partial mutate_rows() failures: acknowledge only the successfully written rows.
- Persistent failures: un-acknowledged messages are redelivered by Pub/Sub
  up to max_delivery_attempts=5 before landing on the DLQ.

Logging emitted
---------------
- messages_pulled       : total messages received per pull call
- bt_rows_written       : Bigtable rows written successfully per batch
- bt_rows_failed        : rows that failed Bigtable write
- messages_acknowledged : Pub/Sub ACKs sent
- messages_nacked       : messages left un-acknowledged (will be redelivered)

Required packages:
    google-cloud-bigtable>=2.22.0
    google-cloud-pubsub>=2.20.0
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_PREDICTIONS_FAMILY    = "predictions"
_PULL_TIMEOUT_SECONDS  = 30
_STREAMING_FLOW_CONTROL_MAX_MESSAGES = 200
_DEFAULT_MAX_MESSAGES  = 100
_BT_MAX_RETRIES        = 3
_BT_RETRY_BASE_SECONDS = 2


@dataclass
class WriterStats:
    """Running totals returned by PredictionWriter.run_batch()."""

    messages_pulled:      int = 0
    bt_rows_written:      int = 0
    bt_rows_failed:       int = 0
    messages_acknowledged: int = 0
    messages_nacked:      int = 0
    batches_processed:    int = 0
    elapsed_seconds:      float = 0.0


@dataclass
class _ParsedMessage:
    ack_id:               str
    row_key:              str
    warehouse_id:         str
    sku_id:               str
    shortage_prob_7d:     float
    stockout_risk_score:  float
    model_version:        str
    prediction_timestamp: str


class PredictionWriter:
    """
    Consumes prediction-results Pub/Sub messages and writes to Bigtable.

    Parameters
    ----------
    project_id:
        GCP project ID.
    subscription_id:
        Pub/Sub subscription ID (e.g. "prediction-writer").
    instance_id:
        Bigtable instance ID (e.g. "scip-feature-store").
    table_id:
        Bigtable table ID (default "inventory_features").
    credentials_path:
        Path to service account JSON key for scip-prediction-writer-sa.
        None = application default credentials.
    """

    def __init__(
        self,
        project_id: str,
        subscription_id: str,
        instance_id: str,
        table_id: str = "inventory_features",
        credentials_path: str | None = None,
    ) -> None:
        self.project_id      = project_id
        self.subscription_id = subscription_id
        self.instance_id     = instance_id
        self.table_id        = table_id

        if credentials_path:
            from google.oauth2 import service_account
            self._creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            self._creds = None

        from google.cloud import bigtable, pubsub_v1

        # Bigtable
        bt_kwargs: dict = {"project": project_id, "admin": False}
        if self._creds:
            bt_kwargs["credentials"] = self._creds
        self._bt_client = bigtable.Client(**bt_kwargs)
        self._table     = self._bt_client.instance(instance_id).table(table_id)

        # Pub/Sub subscriber
        sub_kwargs: dict = {}
        if self._creds:
            sub_kwargs["credentials"] = self._creds
        self._subscriber       = pubsub_v1.SubscriberClient(**sub_kwargs)
        self._subscription_path = self._subscriber.subscription_path(
            project_id, subscription_id
        )

        logger.info(
            "PredictionWriter ready: subscription=%s  bigtable=%s/%s",
            self._subscription_path, instance_id, table_id,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_batch(
        self,
        max_messages: int = _DEFAULT_MAX_MESSAGES,
        max_batches: int | None = None,
    ) -> WriterStats:
        """
        Pull messages in batches until the subscription is empty, then return.

        Suitable for Cloud Run jobs: the process exits naturally once the
        prediction-results topic has been drained.

        Parameters
        ----------
        max_messages:
            Maximum messages per pull RPC call. Default 100.
        max_batches:
            Stop after this many pull calls (None = unlimited).
            Useful for testing with a bounded number of iterations.

        Returns
        -------
        WriterStats with totals across all pull iterations.
        """
        stats   = WriterStats()
        t_start = time.perf_counter()

        logger.info(
            "Starting batch writer: max_messages=%d  max_batches=%s",
            max_messages, max_batches,
        )

        batch_count = 0
        while True:
            if max_batches is not None and batch_count >= max_batches:
                logger.info("max_batches=%d reached — stopping", max_batches)
                break

            try:
                response = self._subscriber.pull(
                    request={
                        "subscription": self._subscription_path,
                        "max_messages":  max_messages,
                    },
                    timeout=_PULL_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                logger.error("Pub/Sub pull failed: %s", exc)
                break

            received = response.received_messages
            if not received:
                logger.info("Subscription drained — no more messages")
                break

            stats.messages_pulled += len(received)
            batch_count           += 1

            parsed   = self._parse_messages(received)
            n_written, n_failed, ack_ids = self._write_to_bigtable(parsed)

            stats.bt_rows_written  += n_written
            stats.bt_rows_failed   += n_failed
            stats.batches_processed += 1

            # Acknowledge only the messages whose Bigtable write succeeded.
            if ack_ids:
                try:
                    self._subscriber.acknowledge(
                        request={
                            "subscription": self._subscription_path,
                            "ack_ids":       ack_ids,
                        }
                    )
                    stats.messages_acknowledged += len(ack_ids)
                except Exception as exc:
                    logger.error(
                        "acknowledge() failed for %d ack_ids: %s — "
                        "messages will be redelivered",
                        len(ack_ids), exc,
                    )

            nacked = len(received) - len(ack_ids)
            stats.messages_nacked += nacked

            logger.info(
                "batch %d: pulled=%d  bt_written=%d  bt_failed=%d  "
                "acked=%d  nacked=%d",
                batch_count, len(received), n_written, n_failed,
                len(ack_ids), nacked,
            )

        stats.elapsed_seconds = time.perf_counter() - t_start
        self._log_summary(stats)
        return stats

    def run_streaming(self, shutdown_event=None) -> None:
        """
        Pull messages indefinitely using Pub/Sub streaming pull.

        Runs until shutdown_event is set (threading.Event) or KeyboardInterrupt.
        Use for long-running daemon processes (Cloud Run service, GKE).

        Parameters
        ----------
        shutdown_event:
            Optional threading.Event. The streaming pull stops when it is set.
        """
        from google.cloud.pubsub_v1.types import FlowControl

        flow_control = FlowControl(
            max_messages=_STREAMING_FLOW_CONTROL_MAX_MESSAGES
        )

        def callback(message):
            parsed = self._parse_single_message(message.ack_id, message.data)
            if parsed is None:
                message.nack()
                return

            success = self._write_single_to_bigtable(parsed)
            if success:
                message.ack()
                logger.debug(
                    "Wrote %s  prob=%.4f  score=%.4f",
                    parsed.row_key,
                    parsed.shortage_prob_7d,
                    parsed.stockout_risk_score,
                )
            else:
                message.nack()
                logger.warning(
                    "Bigtable write failed for %s — nacking (will redeliver)",
                    parsed.row_key,
                )

        streaming_pull_future = self._subscriber.subscribe(
            self._subscription_path,
            callback=callback,
            flow_control=flow_control,
        )
        logger.info("Streaming pull active on %s", self._subscription_path)

        try:
            if shutdown_event is not None:
                while not shutdown_event.is_set():
                    time.sleep(1.0)
                streaming_pull_future.cancel()
            else:
                streaming_pull_future.result()  # blocks until exception
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            logger.info("Streaming pull stopped by KeyboardInterrupt")

    def close(self) -> None:
        self._bt_client.close()
        self._subscriber.close()

    def __enter__(self) -> "PredictionWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Bigtable write helpers ─────────────────────────────────────────────────

    def _write_to_bigtable(
        self,
        parsed: list[_ParsedMessage],
    ) -> tuple[int, int, list[str]]:
        """
        Write a batch of predictions to Bigtable in one mutate_rows() call.

        Returns (n_written, n_failed, successful_ack_ids).
        Retries the whole batch on transient errors with exponential back-off.
        On partial success, returns ack_ids only for rows that were written.
        """
        if not parsed:
            return 0, 0, []

        rows = [self._build_mutation(p) for p in parsed]

        for attempt in range(_BT_MAX_RETRIES):
            try:
                failed_statuses = self._table.mutate_rows(rows)
                break
            except Exception as exc:
                if attempt == _BT_MAX_RETRIES - 1:
                    logger.error(
                        "mutate_rows() failed after %d attempts: %s — "
                        "all %d messages will be nacked",
                        _BT_MAX_RETRIES, exc, len(parsed),
                    )
                    return 0, len(parsed), []
                wait = _BT_RETRY_BASE_SECONDS ** attempt
                logger.warning(
                    "mutate_rows() attempt %d/%d failed: %s. Retrying in %ds.",
                    attempt + 1, _BT_MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        # failed_statuses is a list of statuses for rows that failed.
        # Each entry has .index (position in `rows`) and .code (gRPC status code).
        failed_indices: set[int] = {
            s.index for s in failed_statuses if s.code != 0
        }

        n_written = 0
        n_failed  = 0
        successful_ack_ids: list[str] = []

        for i, p in enumerate(parsed):
            if i in failed_indices:
                n_failed += 1
                logger.warning(
                    "Bigtable write failed for %s (index %d, code %s) — "
                    "message will be redelivered",
                    p.row_key, i,
                    next(
                        (s.code for s in failed_statuses if s.index == i),
                        "unknown",
                    ),
                )
            else:
                n_written += 1
                successful_ack_ids.append(p.ack_id)
                logger.debug(
                    "Written %s  version=%s  prob=%.4f  score=%.4f",
                    p.row_key, p.model_version,
                    p.shortage_prob_7d, p.stockout_risk_score,
                )

        return n_written, n_failed, successful_ack_ids

    def _write_single_to_bigtable(self, parsed: _ParsedMessage) -> bool:
        """Write a single prediction row (used in streaming mode)."""
        row = self._build_mutation(parsed)
        for attempt in range(_BT_MAX_RETRIES):
            try:
                failed = self._table.mutate_rows([row])
                return not any(s.code != 0 for s in failed)
            except Exception as exc:
                if attempt == _BT_MAX_RETRIES - 1:
                    logger.error("Single mutate_rows failed: %s", exc)
                    return False
                time.sleep(_BT_RETRY_BASE_SECONDS ** attempt)
        return False

    def _build_mutation(self, p: _ParsedMessage):
        """Build a Bigtable DirectRow mutation for the predictions column family."""
        # Use the prediction timestamp as the Bigtable cell timestamp so that
        # each scoring round creates a new version within the max_version=3 GC policy.
        try:
            ts = datetime.fromisoformat(
                p.prediction_timestamp.replace("Z", "+00:00")
            )
        except ValueError:
            ts = datetime.now(timezone.utc)

        row = self._table.direct_row(p.row_key.encode("utf-8"))
        row.set_cell(_PREDICTIONS_FAMILY, b"shortage_prob_7d",
                     f"{p.shortage_prob_7d:.6f}".encode("utf-8"), timestamp=ts)
        row.set_cell(_PREDICTIONS_FAMILY, b"stockout_risk_score",
                     f"{p.stockout_risk_score:.6f}".encode("utf-8"), timestamp=ts)
        row.set_cell(_PREDICTIONS_FAMILY, b"model_version",
                     p.model_version.encode("utf-8"), timestamp=ts)
        row.set_cell(_PREDICTIONS_FAMILY, b"prediction_timestamp",
                     p.prediction_timestamp.encode("utf-8"), timestamp=ts)
        return row

    # ── Pub/Sub message parsing ────────────────────────────────────────────────

    def _parse_messages(
        self, received_messages: list
    ) -> list[_ParsedMessage]:
        """Parse a list of Pub/Sub ReceivedMessage objects."""
        parsed: list[_ParsedMessage] = []
        for msg in received_messages:
            result = self._parse_single_message(msg.ack_id, msg.message.data)
            if result is not None:
                parsed.append(result)
        return parsed

    def _parse_single_message(
        self, ack_id: str, data: bytes
    ) -> _ParsedMessage | None:
        """
        Parse a single raw message bytes into a _ParsedMessage.
        Returns None if the message is malformed (will be nacked by caller).
        """
        try:
            payload = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("Failed to decode message (ack_id=%.20s...): %s", ack_id, exc)
            return None

        required = {
            "warehouse_id", "sku_id", "shortage_prob_7d",
            "stockout_risk_score", "model_version", "prediction_timestamp",
        }
        missing = required - set(payload.keys())
        if missing:
            logger.error(
                "Message missing fields %s (ack_id=%.20s...) — discarding",
                missing, ack_id,
            )
            return None

        warehouse_id = payload["warehouse_id"]
        sku_id       = payload["sku_id"]
        row_key      = f"{warehouse_id}#{sku_id}"

        try:
            return _ParsedMessage(
                ack_id               = ack_id,
                row_key              = row_key,
                warehouse_id         = warehouse_id,
                sku_id               = sku_id,
                shortage_prob_7d     = float(payload["shortage_prob_7d"]),
                stockout_risk_score  = float(payload["stockout_risk_score"]),
                model_version        = str(payload["model_version"]),
                prediction_timestamp = str(payload["prediction_timestamp"]),
            )
        except (TypeError, ValueError) as exc:
            logger.error(
                "Failed to parse message fields (ack_id=%.20s...): %s", ack_id, exc
            )
            return None

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log_summary(self, stats: WriterStats) -> None:
        logger.info(
            "=== Prediction Writer Summary ===\n"
            "  batches_processed     : %d\n"
            "  messages_pulled       : %d\n"
            "  bt_rows_written       : %d\n"
            "  bt_rows_failed        : %d\n"
            "  messages_acknowledged : %d\n"
            "  messages_nacked       : %d\n"
            "  elapsed               : %.1f s",
            stats.batches_processed,
            stats.messages_pulled,
            stats.bt_rows_written,
            stats.bt_rows_failed,
            stats.messages_acknowledged,
            stats.messages_nacked,
            stats.elapsed_seconds,
        )
