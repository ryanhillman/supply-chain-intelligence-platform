"""
batch_scorer.py — orchestrates the full batch scoring run.

Flow
----
1. FeatureClient.scan_all() — scan all rows from Bigtable inventory_features.
2. RequestBuilder.build()   — fill defaults, engineer features, scale, map vocab.
3. endpoint.predict()       — call the Vertex AI endpoint in batches of `batch_size`.
   The endpoint returns {"shortage_prob_7d": float} per instance (see
   model_serving/serving/container/main.py). stockout_risk_score is the
   closed-form Normal-approximation P(stockout ≤ 7d) that RequestBuilder
   computes from the raw features — the same formula as the
   continuous_risk_score training column (ADR-004 Decision 6).
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

Logging emitted
---------------
- rows_scanned       : total FeatureRow objects returned by scan_all()
- rows_scored        : entities that passed vocab filter and were sent to endpoint
- batches_sent       : number of endpoint.predict() calls
- endpoint_latency_ms: per-batch and overall mean latency
- predictions_published: total Pub/Sub messages published successfully
- prediction variance  : std dev of shortage_prob_7d (collapse < 0.01 flagged)

Required packages:
    google-cloud-aiplatform>=1.38.0
    google-cloud-pubsub>=2.20.0
    numpy>=1.26.0
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

import numpy as np

from model_serving.pipeline.constants import VARIANCE_COLLAPSE_THRESHOLD as _VARIANCE_COLLAPSE_THRESHOLD
from model_serving.pipeline.feature_client import FeatureClient
from model_serving.pipeline.request_builder import RequestBuilder, ScoringRequest

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 32
_DEFAULT_TOPIC_ID   = "prediction-results"


@dataclass
class ScoringStats:
    """Summary statistics returned by BatchScorer.run()."""

    rows_scanned:            int   = 0
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
    Orchestrates batch inference: Bigtable -> Vertex AI -> Pub/Sub.

    Parameters
    ----------
    project_id:
        GCP project ID.
    region:
        GCP region (e.g. "us-central1").
    endpoint_resource_name:
        Full Vertex AI endpoint resource name from the Terraform output
        `vertex_ai_endpoint_resource_name`.
    topic_id:
        Pub/Sub topic ID for prediction messages (default "prediction-results").
    feature_client:
        Initialised FeatureClient instance.
    request_builder:
        Initialised RequestBuilder instance.
    credentials_path:
        Path to service account JSON key. None = application default credentials.
    """

    def __init__(
        self,
        project_id: str,
        region: str,
        endpoint_resource_name: str,
        feature_client: FeatureClient,
        request_builder: RequestBuilder,
        topic_id: str = _DEFAULT_TOPIC_ID,
        credentials_path: str | None = None,
    ) -> None:
        self.project_id             = project_id
        self.region                 = region
        self.endpoint_resource_name = endpoint_resource_name
        self.topic_id               = topic_id
        self._feature_client        = feature_client
        self._request_builder       = request_builder

        if credentials_path:
            from google.oauth2 import service_account
            self._creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        else:
            self._creds = None

        from google.cloud import aiplatform

        init_kwargs: dict = {"project": project_id, "location": region}
        if self._creds:
            init_kwargs["credentials"] = self._creds
        aiplatform.init(**init_kwargs)

        self._endpoint = aiplatform.Endpoint(endpoint_name=endpoint_resource_name)

        # Publisher client with batching to coalesce small messages.
        from google.cloud import pubsub_v1

        publisher_kwargs: dict = {}
        if self._creds:
            publisher_kwargs["credentials"] = self._creds
        self._publisher = pubsub_v1.PublisherClient(**publisher_kwargs)
        self._topic_path = self._publisher.topic_path(project_id, topic_id)

        logger.info(
            "BatchScorer ready: endpoint=%s  topic=%s",
            endpoint_resource_name, self._topic_path,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        model_version: str = "unknown",
    ) -> ScoringStats:
        """
        Execute the full batch scoring pipeline.

        Parameters
        ----------
        batch_size:
            Number of instances per endpoint.predict() call. Default 32.
            Vertex AI Online Prediction supports up to 10 MB per request;
            32 × 21 float32 features is well within that limit.
        model_version:
            Version string embedded in each Pub/Sub message so PredictionWriter
            can write it to the predictions:model_version Bigtable column.
            Obtain from registration_info.json or the --model-version CLI flag.

        Returns
        -------
        ScoringStats with all logged counters populated.
        """
        stats = ScoringStats(model_version=model_version)
        t_start = time.perf_counter()

        # Step 1: read all feature rows from Bigtable.
        logger.info("=== Phase 5.4 Batch Scoring — start ===")
        feature_rows = self._feature_client.scan_all()
        stats.rows_scanned = len(feature_rows)
        logger.info("rows_scanned=%d", stats.rows_scanned)

        if not feature_rows:
            logger.warning("No feature rows returned — aborting scoring run")
            return stats

        # Step 2: apply defaults, engineer features, scale, vocab-map.
        requests = self._request_builder.build(feature_rows)
        stats.rows_scored = len(requests)
        logger.info("rows_scored=%d  (skipped=%d vocab-unknown)",
                    stats.rows_scored, stats.rows_scanned - stats.rows_scored)

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
        stats.variance_collapsed = stats.prob_std < _VARIANCE_COLLAPSE_THRESHOLD

        self._log_summary(stats)
        return stats

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _score_batch(
        self,
        batch: list[ScoringRequest],
    ) -> tuple[list[float], float]:
        """
        Call endpoint.predict() for a single batch.

        Returns (shortage_probs, latency_ms). model_version is
        not needed here — the endpoint already routes to whichever model
        version currently has traffic; it's only needed by _publish_batch()
        to stamp the published record for lineage.
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

        for req, prob, score in zip(batch, probs, scores):
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

    def _log_summary(self, stats: ScoringStats) -> None:
        logger.info(
            "=== Batch Scoring Summary ===\n"
            "  model_version           : %s\n"
            "  rows_scanned            : %d\n"
            "  rows_scored             : %d\n"
            "  batches_sent            : %d\n"
            "  predictions_published   : %d\n"
            "  publish_failures        : %d\n"
            "  mean_endpoint_latency   : %.1f ms\n"
            "  shortage_prob_7d        : mean=%.4f  std=%.4f\n"
            "  stockout_risk_score     : mean=%.4f  std=%.4f\n"
            "  variance_collapsed      : %s\n"
            "  elapsed                 : %.1f s",
            stats.model_version,
            stats.rows_scanned,
            stats.rows_scored,
            stats.batches_sent,
            stats.predictions_published,
            stats.publish_failures,
            stats.mean_endpoint_latency_ms,
            stats.prob_mean, stats.prob_std,
            stats.score_mean, stats.score_std,
            stats.variance_collapsed,
            stats.elapsed_seconds,
        )

        if stats.variance_collapsed:
            logger.warning(
                "DISTRIBUTION SHIFT ALERT: shortage_prob_7d std=%.4f < %.2f. "
                "Model predictions have collapsed to a near-constant value. "
                "This is expected with the simulated-data model on real Silver data "
                "(ADR-005 Decision 7). "
                "Trigger Phase 5.5 drift monitoring and retraining gate.",
                stats.prob_std, _VARIANCE_COLLAPSE_THRESHOLD,
            )


# ── Utilities ──────────────────────────────────────────────────────────────────

def _chunks(items: list, size: int) -> Iterator[list]:
    """Yield successive non-overlapping chunks of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]
