"""
build_dataset.py — CLI entry point for Step 4.1.

Reads Bronze GCS Parquet -> engineers features -> generates labels ->
splits 70/15/15 by time -> writes train/val/test Parquet + metadata.

Usage
-----
    python phase4/scripts/build_dataset.py \\
        --bucket my-project-scip-lake \\
        --sa-key infrastructure/gcp/keys/scip-spark-sa.json \\
        --output phase4/artifacts/dataset

Environment variables (alternative to flags):
    GCS_BUCKET_NAME            overrides --bucket
    GOOGLE_APPLICATION_CREDENTIALS  overrides --sa-key
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running as: python phase4/scripts/build_dataset.py from repo root
sys.path.insert(0, str(Path(__file__).parents[2]))

from phase4.datasets.dataset_builder import DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("build_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Phase 4 training dataset from Bronze GCS")
    p.add_argument(
        "--bucket",
        default=os.environ.get("GCS_BUCKET_NAME", ""),
        help="GCS bucket name (e.g. my-project-scip-lake)",
    )
    p.add_argument(
        "--sa-key",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Path to service account JSON key (default: Application Default Credentials)",
    )
    p.add_argument(
        "--output",
        default="phase4/artifacts/dataset",
        help="Local output directory for Parquet splits and metadata",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="Stockout prediction horizon in days (default: 7)",
    )
    p.add_argument(
        "--simulate-days",
        type=int,
        default=60,
        help=(
            "Tile Bronze events across N simulated days when data spans < 38 days "
            "(default: 60; set 0 to disable and raise an error instead)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.bucket:
        logger.error(
            "GCS bucket name is required. Pass --bucket or set GCS_BUCKET_NAME."
        )
        sys.exit(1)

    logger.info("Starting dataset build")
    logger.info("  Bucket        : %s", args.bucket)
    logger.info("  SA key        : %s", args.sa_key or "Application Default Credentials")
    logger.info("  Output        : %s", args.output)
    logger.info("  Horizon       : %d days", args.horizon)
    logger.info("  Simulate days : %d", args.simulate_days)
    if args.simulate_days > 0:
        logger.warning(
            "Temporal simulation enabled (%d days). Bronze data will be tiled "
            "with ±10%% quantity noise. Disable with --simulate-days 0 once "
            "real multi-day Bronze data is available.", args.simulate_days
        )

    builder = DatasetBuilder(
        bucket           = args.bucket,
        credentials_path = args.sa_key,
        output_dir       = args.output,
        horizon_days     = args.horizon,
        simulate_days    = args.simulate_days,
    )
    manifest = builder.build()

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
        print("  Baseline (days_of_supply < 7) test metrics:")
        print(f"    AUC-ROC : {bm['auc_roc']:.4f}")
        print(f"    AUPRC   : {bm['auprc']:.4f}")
        print(f"    Brier   : {bm['brier']:.4f}")

    print()
    print("  Next step:")
    print("    python phase4/scripts/train.py \\")
    print(f"      --dataset {args.output} \\")
    print("      --artifacts phase4/artifacts")
    print("=" * 60)


if __name__ == "__main__":
    main()
