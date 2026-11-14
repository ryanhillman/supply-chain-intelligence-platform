"""
score.py — CLI entry point for the batch scoring pipeline.

This script:
1. Scans all rows from Bigtable inventory_features.
2. Applies engineer_features() from ml_training: defaults, then derived
   features (raw units).
3. Calls the Vertex AI endpoint in batches to obtain predictions.
4. Publishes each prediction to the Pub/Sub prediction-results topic.

Optionally runs the PredictionWriter in the same process after publishing
(--write flag) to drain the topic into Bigtable predictions column family.
Alternatively, run prediction_writer.py separately.

Usage
-----
    # Score only (writer runs separately):
    python model_serving/scripts/score.py \\
        --project your-gcp-project-id \\
        --region  us-central1 \\
        --endpoint $VERTEX_AI_ENDPOINT_RESOURCE_NAME \\
        --bigtable-instance scip-feature-store \\
        --topic prediction-results \\
        --model-version v20260620_180918 \\
        --sa-key infrastructure/gcp/keys/scip-vertex-sa.json

    # Score + write in one process (the writer uses its own key):
    python model_serving/scripts/score.py ... --write \\
        --subscription prediction-writer \\
        --writer-sa-key infrastructure/gcp/keys/scip-prediction-writer-sa.json

    # Dry-run — reads Bigtable and builds requests but does NOT call endpoint:
    python model_serving/scripts/score.py ... --dry-run

Environment variables (alternatives to CLI flags):
    GCP_PROJECT_ID                   -- --project
    GCP_REGION                       -- --region
    VERTEX_AI_ENDPOINT_RESOURCE_NAME -- --endpoint
    BIGTABLE_INSTANCE_ID             -- --bigtable-instance
    PUBSUB_TOPIC_ID                  -- --topic
    PUBSUB_SUBSCRIPTION_ID           -- --subscription
    MODEL_VERSION                    -- --model-version
    GOOGLE_APPLICATION_CREDENTIALS   -- --sa-key  (scorer SA)
    WRITER_SA_KEY                    -- --writer-sa-key
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from model_serving                            import cli
from model_serving.pipeline.constants         import VARIANCE_COLLAPSE_THRESHOLD
from model_serving.pipeline.feature_client    import FeatureClient, split_fresh
from model_serving.pipeline.request_builder   import RequestBuilder
from model_serving.pipeline.batch_scorer      import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_FEATURE_AGE_HOURS,
    DEFAULT_TOPIC_ID,
    BatchScorer,
)
from model_serving.pipeline.prediction_writer import DEFAULT_MAX_MESSAGES, PredictionWriter

logger = logging.getLogger("score")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the SCIP batch scoring pipeline (Bigtable -> model endpoint -> Pub/Sub)"
    )

    cli.add_project_arg(p)
    cli.add_region_arg(p)
    cli.add_sa_key_arg(p, "Service account key for the batch scorer (scip-vertex-sa.json)")
    cli.add_endpoint_arg(p)
    cli.add_bigtable_args(p)

    # ── Pub/Sub ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--topic",
        default=os.environ.get("PUBSUB_TOPIC_ID", DEFAULT_TOPIC_ID),
        help=f"Pub/Sub topic ID for predictions (default: {DEFAULT_TOPIC_ID})",
    )
    p.add_argument(
        "--subscription",
        default=os.environ.get("PUBSUB_SUBSCRIPTION_ID", "prediction-writer"),
        help="Pub/Sub subscription ID, used only with --write (default: prediction-writer)",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--model-version",
        default=os.environ.get("MODEL_VERSION", ""),
        help=(
            "Model version string embedded in each Pub/Sub message "
            "(e.g. v20260620_180918). "
            "Defaults to the version in model_serving/artifacts/registration_info.json."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Instances per endpoint.predict() call (default: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--max-feature-age-hours",
        type=float,
        default=float(os.environ.get("MAX_FEATURE_AGE_HOURS", DEFAULT_MAX_FEATURE_AGE_HOURS)),
        help=(
            "Skip rows whose lineage:silver_ts is older than this many hours "
            f"(default: {DEFAULT_MAX_FEATURE_AGE_HOURS:g}). 0 disables the check."
        ),
    )

    # ── Local artifacts ───────────────────────────────────────────────────────
    p.add_argument(
        "--artifacts",
        default="ml_training/artifacts",
        help="Directory containing feature_metadata.json (default: ml_training/artifacts)",
    )

    # ── Writer (optional in-process) ──────────────────────────────────────────
    p.add_argument(
        "--write",
        action="store_true",
        help=(
            "After publishing, drain the Pub/Sub subscription and write "
            "predictions to Bigtable in the same process. "
            "If omitted, run prediction_writer.py separately."
        ),
    )
    p.add_argument(
        "--writer-sa-key",
        default=os.environ.get("WRITER_SA_KEY"),
        help=(
            "Service account key for the prediction writer (scip-prediction-writer-sa.json). "
            "Required when --write is set and differs from --sa-key."
        ),
    )
    p.add_argument(
        "--writer-max-messages",
        type=int,
        default=DEFAULT_MAX_MESSAGES,
        help=f"Max messages per Pub/Sub pull call in --write mode (default: {DEFAULT_MAX_MESSAGES})",
    )

    # ── Control ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Read Bigtable and build requests but do NOT call the endpoint "
            "or publish to Pub/Sub."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Set log level to DEBUG",
    )

    return p.parse_args()


def _resolve_model_version(args: argparse.Namespace) -> str:
    """Return the model version string from CLI flag or registration_info.json."""
    if args.model_version:
        return args.model_version

    if cli.REGISTRATION_INFO_PATH.exists():
        try:
            version = cli.read_json(cli.REGISTRATION_INFO_PATH).get("version", "")
            if version:
                logger.info("model_version resolved from registration_info.json: %s", version)
                return version
        except Exception as exc:
            logger.warning("Could not read registration_info.json: %s", exc)

    logger.warning(
        "model_version not provided and registration_info.json unavailable. "
        "Using 'unknown'. Pass --model-version explicitly for traceability."
    )
    return "unknown"


def main() -> None:
    args = parse_args()
    cli.setup_logging(args.verbose)

    # Validate required arguments.
    cli.require(args, "project",  "GCP_PROJECT_ID")
    if not args.dry_run:
        cli.require(args, "endpoint", "VERTEX_AI_ENDPOINT_RESOURCE_NAME")

    model_version = _resolve_model_version(args)

    logger.info(
        "Batch scoring pipeline starting:\n"
        "  project          : %s\n"
        "  region           : %s\n"
        "  endpoint         : %s\n"
        "  bigtable         : %s / %s\n"
        "  topic            : %s\n"
        "  model_version    : %s\n"
        "  batch_size       : %d\n"
        "  dry_run          : %s",
        args.project, args.region,
        args.endpoint or "(dry-run — not used)",
        args.bigtable_instance, args.bigtable_table,
        args.topic,
        model_version,
        args.batch_size,
        args.dry_run,
    )

    # ── Feature client ────────────────────────────────────────────────────────
    with FeatureClient(
        project_id       = args.project,
        instance_id      = args.bigtable_instance,
        table_id         = args.bigtable_table,
        credentials_path = args.sa_key,
    ) as feature_client:
        _run_scoring(args, feature_client, model_version)


def _run_scoring(
    args: argparse.Namespace,
    feature_client: FeatureClient,
    model_version: str,
) -> None:
    """The scoring body, run inside main()'s `with FeatureClient(...)` block
    so the Bigtable client is always closed, including on an exception or
    sys.exit() from any of the branches below."""
    # ── Request builder ───────────────────────────────────────────────────────
    request_builder = RequestBuilder(artifacts_dir=args.artifacts)

    # ── Dry-run: scan + build only ────────────────────────────────────────────
    if args.dry_run:
        logger.info("[dry-run] Scanning Bigtable...")
        feature_rows = feature_client.scan_all()
        feature_rows, n_stale = split_fresh(feature_rows, args.max_feature_age_hours)
        logger.info("[dry-run] rows_scanned=%d  rows_stale=%d", len(feature_rows) + n_stale, n_stale)

        requests = request_builder.build(feature_rows)
        logger.info("[dry-run] rows_scored=%d — NOT calling endpoint or Pub/Sub", len(requests))

        sys.exit(0)

    # ── Full scoring run ──────────────────────────────────────────────────────
    scorer = BatchScorer(
        project_id             = args.project,
        region                 = args.region,
        feature_client         = feature_client,
        request_builder        = request_builder,
        endpoint_resource_name = args.endpoint,
        topic_id               = args.topic,
        credentials_path       = args.sa_key,
    )

    stats = scorer.run(
        batch_size            = args.batch_size,
        model_version         = model_version,
        max_feature_age_hours = args.max_feature_age_hours or None,
    )

    print("\n" + "=" * 68)
    print("  Batch Scoring Complete")
    print("=" * 68)
    print(f"  rows_scanned            : {stats.rows_scanned}")
    print(f"  rows_stale              : {stats.rows_stale}")
    print(f"  rows_scored             : {stats.rows_scored}")
    print(f"  predictions_published   : {stats.predictions_published}")
    print(f"  publish_failures        : {stats.publish_failures}")
    print(f"  mean_endpoint_latency   : {stats.mean_endpoint_latency_ms:.1f} ms")
    print(f"  shortage_prob_7d        : mean={stats.prob_mean:.4f}  std={stats.prob_std:.4f}")
    print(f"  stockout_risk_score     : mean={stats.score_mean:.4f}  std={stats.score_std:.4f}")
    print(f"  variance_collapsed      : {stats.variance_collapsed}")
    print(f"  elapsed                 : {stats.elapsed_seconds:.1f} s")
    print("=" * 68)

    if stats.variance_collapsed:
        print(
            f"\n  [WARNING] prediction variance collapsed (std < {VARIANCE_COLLAPSE_THRESHOLD}).\n"
            "  Run model_serving/scripts/monitor.py to check feature drift\n"
            "  before using these predictions."
        )

    if stats.predictions_published == 0:
        if stats.rows_stale and not stats.rows_scored:
            logger.error(
                "No predictions were published: all %d rows were stale — "
                "re-run the Silver job, or raise --max-feature-age-hours",
                stats.rows_stale,
            )
        else:
            logger.error("No predictions were published — check the model endpoint and Pub/Sub config")
        sys.exit(1)

    # ── Optional in-process write ─────────────────────────────────────────────
    if args.write:
        logger.info("--write set: draining Pub/Sub subscription %s into Bigtable...", args.subscription)

        writer_key = args.writer_sa_key or args.sa_key
        if args.sa_key and not args.writer_sa_key:
            logger.warning(
                "No --writer-sa-key provided. Using --sa-key for the writer. "
                "In production, use scip-prediction-writer-sa.json (roles: "
                "pubsub.subscriber, bigtable.user)."
            )

        with PredictionWriter(
            project_id       = args.project,
            subscription_id  = args.subscription,
            instance_id      = args.bigtable_instance,
            table_id         = args.bigtable_table,
            credentials_path = writer_key,
        ) as writer:
            writer_stats = writer.run_batch(max_messages=args.writer_max_messages)

        print()
        print("  Prediction Writer Summary")
        print(f"  bigtable_rows_written : {writer_stats.bigtable_rows_written}")
        print(f"  bigtable_rows_failed  : {writer_stats.bigtable_rows_failed}")
        print(f"  messages_acknowledged : {writer_stats.messages_acknowledged}")
        print(f"  messages_nacked       : {writer_stats.messages_nacked}")
        print("=" * 68)

        if writer_stats.bigtable_rows_failed > 0:
            logger.warning(
                "%d Bigtable write failures — those messages remain on the subscription "
                "and will be redelivered (up to 5 attempts before DLQ).",
                writer_stats.bigtable_rows_failed,
            )
        if writer_stats.bigtable_rows_written == 0:
            logger.error("No Bigtable rows written — check Bigtable connectivity and IAM")
            sys.exit(1)

    else:
        print(
            "\n  Predictions are on the Pub/Sub topic.\n"
            "  To write them to Bigtable, run the prediction writer standalone\n"
            "  (this does NOT re-run scoring):\n"
            f"\n    python model_serving/pipeline/prediction_writer.py \\\n"
            f"        --project {args.project} \\\n"
            f"        --bigtable-instance {args.bigtable_instance} \\\n"
            f"        --subscription {args.subscription}\n"
            "\n  (add --sa-key infrastructure/gcp/keys/scip-prediction-writer-sa.json on GCP.)"
        )

    # ── Exit status ───────────────────────────────────────────────────────────
    # Checked after the write, so the predictions that were published reach Bigtable.
    incomplete = []
    if stats.publish_failures > 0:
        incomplete.append(
            f"{stats.publish_failures} of {stats.rows_scored} predictions were not published"
        )
    if args.write and writer_stats.bigtable_rows_written < stats.predictions_published:
        incomplete.append(
            f"{writer_stats.bigtable_rows_written} of {stats.predictions_published} "
            "published predictions were written to Bigtable"
        )
    if incomplete:
        logger.error("Scoring run incomplete: %s", "; ".join(incomplete))
        sys.exit(1)


if __name__ == "__main__":
    main()
