"""
build_dataset.py — CLI entry point for Step 4.1.

Reads Bronze Parquet -> engineers features -> generates labels ->
splits 70/15/15 by time -> writes train/val/test Parquet + metadata.

Usage
-----
    # Local Bronze written by the Spark job (the default local run):
    python ml_training/scripts/build_dataset.py \\
        --bronze-path spark/data/bronze \\
        --output-dir ml_training/artifacts/dataset

    # GCS Bronze (June 2026 deployment):
    python ml_training/scripts/build_dataset.py \\
        --bucket my-project-scip-lake \\
        --sa-key infrastructure/gcp/keys/scip-spark-sa.json \\
        --output-dir ml_training/artifacts/dataset

Environment variables (alternative to flags):
    GCS_BUCKET_NAME                 overrides --bucket
    GOOGLE_APPLICATION_CREDENTIALS  overrides --sa-key
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running as: python ml_training/scripts/build_dataset.py from repo root
sys.path.insert(0, str(Path(__file__).parents[2]))

from ml_training.datasets.dataset_builder import DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("build_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the training dataset from Bronze (local directory or GCS)"
    )

    # ── Bronze source ─────────────────────────────────────────────────────────
    p.add_argument(
        "--bronze-path",
        default=None,
        help=(
            "Local Bronze directory written by spark/jobs/bronze_ingestion.py "
            "(e.g. spark/data/bronze). Takes precedence over --bucket."
        ),
    )
    p.add_argument(
        "--bucket",
        default=os.environ.get("GCS_BUCKET_NAME", ""),
        help=(
            "GCS bucket name (e.g. my-project-scip-lake) for Bronze on GCS. "
            "One of --bronze-path / --bucket is required."
        ),
    )
    p.add_argument(
        "--sa-key",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Path to service account JSON key (default: Application Default Credentials)",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--output-dir",
        default="ml_training/artifacts/dataset",
        help="Local output directory for Parquet splits and metadata",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="Stockout prediction horizon in days (default: 7)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bronze_path and not args.bucket:
        logger.error(
            "A Bronze source is required: --bronze-path <dir> (local Spark output) "
            "or --bucket <name> / GCS_BUCKET_NAME (GCS)."
        )
        sys.exit(1)

    if args.bronze_path:
        logger.info("Bronze path : %s", args.bronze_path)
    else:
        logger.info("Bronze bucket : %s (SA key: %s)", args.bucket,
                    args.sa_key or "Application Default Credentials")
    logger.info("Output : %s | horizon : %d days", args.output_dir, args.horizon)

    manifest = DatasetBuilder(
        bucket           = None if args.bronze_path else args.bucket,
        bronze_path      = args.bronze_path,
        credentials_path = args.sa_key,
        output_dir       = args.output_dir,
        horizon_days     = args.horizon,
    ).build()

    print("\n" + "=" * 60)
    print("  Dataset Build Complete")
    print("=" * 60)
    print(f"  Train    : {manifest.n_train:>6} rows")
    print(f"  Val      : {manifest.n_val:>6} rows")
    print(f"  Test     : {manifest.n_test:>6} rows")
    print(f"  Features : {manifest.n_features:>6} numeric")
    print(f"  Date range: {manifest.date_range_start} -> {manifest.date_range_end}")
    print(f"  Sample dates: {manifest.n_sample_dates}")
    print(f"  Positive rate (train): {manifest.positive_rate_train:.3f}")
    print(f"  Positive rate (test) : {manifest.positive_rate_test:.3f}")

    bm = manifest.baseline_test_metrics
    if bm.get("auc_roc") is not None:
        print()
        print("  Baseline (Normal-approx formula) test metrics:")
        print(f"    AUC-ROC : {bm['auc_roc']:.4f}")
        print(f"    AUPRC   : {bm['auprc']:.4f}")
        print(f"    Brier   : {bm['brier']:.4f}")

    print()
    print("  Next step:")
    print("    python ml_training/scripts/train.py \\")
    print(f"      --dataset {args.output_dir} \\")
    print("      --artifacts ml_training/artifacts")
    print("=" * 60)


if __name__ == "__main__":
    main()
