"""
build_dataset.py — CLI entry point for Step 4.1.

Reads Bronze Parquet -> engineers features -> generates labels ->
splits 70/15/15 by time -> writes train/val/test Parquet + metadata.

Usage
-----
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

from ml_training.datasets.dataset_builder import DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("build_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the training dataset from Bronze on GCS"
    )

    # ── Bronze source ─────────────────────────────────────────────────────────
    p.add_argument(
        "--bucket",
        default=os.environ.get("GCS_BUCKET_NAME", ""),
        help="GCS bucket name holding Bronze (e.g. my-project-scip-lake). Required.",
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.bucket:
        logger.error("A Bronze bucket is required: --bucket <name> or GCS_BUCKET_NAME.")
        sys.exit(1)

    logger.info("Bronze bucket : %s (SA key: %s)", args.bucket,
                args.sa_key or "Application Default Credentials")
    logger.info("Output : %s", args.output_dir)

    manifest = DatasetBuilder(
        bucket           = args.bucket,
        credentials_path = args.sa_key,
        output_dir       = args.output_dir,
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
