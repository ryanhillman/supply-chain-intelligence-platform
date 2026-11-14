"""
batch_scorer.py — orchestrates the full batch scoring run.

Flow
----
1. FeatureClient.scan_all() — scan all rows from Bigtable inventory_features,
   then split_fresh() drops rows whose lineage:silver_ts is older than
   max_feature_age_hours (default 48 h): stale features are not scored.
2. RequestBuilder.build()   — fill defaults, engineer features, map vocab.
3. endpoint.predict()       — call the Vertex AI endpoint in batches of `batch_size`.
   The endpoint returns {"shortage_prob_7d": float} per instance (see
   model_serving/serving/container/main.py). stockout_risk_score is the
   closed-form Normal-approximation P(stockout ≤ 7d) that RequestBuilder
   computes from the raw features — the same formula as the training
   rows' stockout_risk_score column.
4. For each prediction: build a JSON Pub/Sub message and publish to the
   prediction-results topic.

Pub/Sub message schema (one per entity)
-----------------------------------------
{
  "warehouse_id":        "W-001",
  "sku_id":              "SKU-005",
  "shortage_prob_7d":    0.73,
  "stockout_risk_score": 0.68,
  "model_version":       "v20260620_180918",
  "prediction_timestamp": "2026-06-20T18:09:21Z"
}
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import numpy as np

from model_serving import cli
from model_serving.pipeline.constants import variance_collapsed
from model_serving.pipeline.feature_client import FeatureClient, split_fresh
from model_serving.pipeline.request_builder import RequestBuilder, ScoringRequest

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32
DEFAULT_TOPIC_ID   = "prediction-results"
DEFAULT_MAX_FEATURE_AGE_HOURS = 48.0


@dataclass
class ScoringStats:
    """Summary statistics returned by BatchScorer.run()."""

    rows_scanned:            int   = 0
    rows_stale:              int   = 0
    rows_scored:             int   = 0
    batches_sent:            int   = 0
    predictions_published:   int   = 0
    publish_failures:        int   = 0
    mean_endpoint_latency_ms: float = 0.0
    prob_mean:               float = 0.0
    prob_std:                float = 0.0
    score_mean:              float = 0.0
    score_std:               float = 0.0
    variance_collapsed:      bool  = False
    elapsed_seconds:         float = 0.0
    model_version:           str   = "unknown"
    all_latencies_ms:        list[float] = field(default_factory=list)


class BatchScorer:
    """
    Orchestrates batch inference: Bigtable -> model endpoint -> Pub/Sub.

    Parameters
    ----------
    project_id:
        GCP project ID.
    region:
        GCP region (e.g. "us-central1").
    feature_client:
        Initialised FeatureClient instance.
    request_builder:
        Initialised RequestBuilder instance.
    endpoint_resource_name:
        Full Vertex AI endpoint resource name from the Terraform output
        `vertex_ai_endpoint_resource_name`.
    topic_id:
        Pub/Sub topic ID for prediction messages (default "prediction-results").
    credentials_path:
        Path to service account JSON key. None = application default credentials.
    """

    def __init__(
        self,
        project_id: str,
        region: str,
        feature_client: FeatureClient,
        request_builder: RequestBuilder,
        *,
        endpoint_resource_name: str,
        topic_id: str = DEFAULT_TOPIC_ID,
        credentials_path: str | None = None,
    ) -> None:
        self.project_id             = project_id
        self.region                 = region
        self.endpoint_resource_name = endpoint_resource_name
        self.topic_id               = topic_id
        self._feature_client        = feature_client
        self._request_builder       = request_builder

        from google.cloud import aiplatform, pubsub_v1

        creds = cli.load_credentials(credentials_path)
        aiplatform.init(project=project_id, location=region, credentials=creds)
        self._endpoint = aiplatform.Endpoint(endpoint_name=endpoint_resource_name)

        self._publisher  = pubsub_v1.PublisherClient(credentials=creds)
        self._topic_path = self._publisher.topic_path(project_id, topic_id)

        logger.info(
            "BatchScorer ready: endpoint=%s  topic=%s",
            endpoint_resource_name, self._topic_path,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model_version: str = "unknown",
        max_feature_age_hours: float | None = DEFAULT_MAX_FEATURE_AGE_HOURS,
    ) -> ScoringStats:
        """
        Execute the full batch scoring pipeline.

        Parameters
        ----------
        batch_size:
            Number of instances per endpoint.predict() call. Default 32.
            Vertex AI online prediction accepts requests up to 1.5 MB;
            32 × 21 float32 features is well within that limit.
        model_version:
            Version string embedded in each Pub/Sub message so PredictionWriter
            can write it to the predictions:model_version Bigtable column.
            Obtain from registration_info.json or the --model-version CLI flag.
        max_feature_age_hours:
            Rows whose lineage:silver_ts is older than this (or missing) are
            skipped and counted in rows_stale. Default 48 h; None disables.

        Returns
        -------
        ScoringStats with all logged counters populated.
        """
        stats = ScoringStats(model_version=model_version)
        t_start = time.perf_counter()

        # Step 1: read all feature rows from Bigtable.
        logger.info("=== Batch scoring — start ===")
        feature_rows = self._feature_client.scan_all()
        stats.rows_scanned = len(feature_rows)
        logger.info("rows_scanned=%d", stats.rows_scanned)

        # Step 1b: refuse stale rows. Feature families keep only the latest cell
        # (no age-based GC), so after a quiet period the store still looks
        # complete; the Silver timestamp is what tells us it is out of date.
        feature_rows, stats.rows_stale = split_fresh(feature_rows, max_feature_age_hours)
        if stats.rows_stale:
            logger.warning(
                "rows_stale=%d skipped (lineage:silver_ts older than %s h or missing) "
                "— re-run the Silver job to refresh features",
                stats.rows_stale, max_feature_age_hours,
            )

        if not feature_rows:
            logger.warning("No fresh feature rows — aborting scoring run")
            return stats

        # Step 2: apply defaults, engineer features, vocab-map.
        requests = self._request_builder.build(feature_rows)
        stats.rows_scored = len(requests)
        # feature_rows holds only the fresh rows here, so stale rows are not counted.
        logger.info("rows_scored=%d  (skipped=%d vocab-unknown)",
                    stats.rows_scored, len(feature_rows) - stats.rows_scored)

        if not requests:
            logger.warning("No valid scoring requests — aborting")
            return stats

        # Step 3 + 4: endpoint.predict() in batches, publish each prediction.
        all_probs:  list[float] = []
        all_scores: list[float] = []

        for batch in _chunks(requests, batch_size):
            batch_probs, latency_ms = self._score_batch(batch)
            batch_scores = [req.stockout_risk_score for req in batch]
            all_probs.extend(batch_probs)
            all_scores.extend(batch_scores)
            stats.batches_sent          += 1
            stats.all_latencies_ms.append(latency_ms)

            n_published = self._publish_batch(batch, batch_probs, batch_scores, model_version)
            stats.predictions_published  += n_published
            stats.publish_failures       += len(batch) - n_published

            logger.info(
                "batch %d: size=%d  latency=%.1f ms  published=%d  "
                "prob_mean=%.4f  score_mean=%.4f",
                stats.batches_sent, len(batch), latency_ms, n_published,
                float(np.mean(batch_probs)), float(np.mean(batch_scores)),
            )

        # Summarise statistics.
        stats.elapsed_seconds         = time.perf_counter() - t_start
        stats.mean_endpoint_latency_ms = float(np.mean(stats.all_latencies_ms))
        stats.prob_mean  = float(np.mean(all_probs))
        stats.prob_std   = float(np.std(all_probs))
        stats.score_mean = float(np.mean(all_scores))
        stats.score_std  = float(np.std(all_scores))
        stats.variance_collapsed = variance_collapsed(stats.prob_std)

        return stats

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _score_batch(
        self,
        batch: list[ScoringRequest],
    ) -> tuple[list[float], float]:
        """
        Call endpoint.predict() for a single batch.

        Returns (shortage_probs, latency_ms). The endpoint routes the request
        to whichever deployed model has traffic.
        """
        instances = [req.instance for req in batch]

        t0       = time.perf_counter()
        response = self._endpoint.predict(instances=instances)
        latency  = (time.perf_counter() - t0) * 1000.0

        probs: list[float] = [float(pred["shortage_prob_7d"]) for pred in response.predictions]

        if len(probs) != len(batch):
            raise RuntimeError(
                f"Endpoint returned {len(probs)} predictions for {len(batch)} instances; "
                "cannot safely align results"
            )

        return probs, latency

    def _publish_batch(
        self,
        batch: list[ScoringRequest],
        probs: list[float],
        scores: list[float],
        model_version: str,
    ) -> int:
        """
        Publish one Pub/Sub message per entity.

        Returns the count of successfully published messages.
        Publish futures are awaited before returning so the caller's
        predictions_published count is accurate.
        """
        ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        futures = []

        for req, prob, score in zip(batch, probs, scores, strict=True):
            message = {
                "warehouse_id":         req.warehouse_id,
                "sku_id":               req.sku_id,
                "shortage_prob_7d":     round(prob,  6),
                "stockout_risk_score":  round(score, 6),
                "model_version":        model_version,
                "prediction_timestamp": ts_utc,
            }
            data = json.dumps(message).encode("utf-8")
            futures.append((req.row_key, self._publisher.publish(self._topic_path, data=data)))

        n_published = 0
        for row_key, future in futures:
            try:
                future.result(timeout=30.0)
                n_published += 1
            except Exception as exc:
                logger.error("Publish failed for %s: %s", row_key, exc)

        return n_published


# ── Utilities ──────────────────────────────────────────────────────────────────

def _chunks(items: list, size: int) -> Iterator[list]:
    """Yield successive non-overlapping chunks of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]
