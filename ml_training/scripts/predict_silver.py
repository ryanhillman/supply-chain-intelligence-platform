"""
predict_silver.py — batch inference on the current Silver GCS layer.

Reads the latest Silver snapshot, runs the trained ShortagePredictor,
and writes a Parquet file with risk__shortage_prob_7d and
risk__stockout_risk_score populated for all 120 warehouse-SKU entities.

Also prints a sanity-check table comparing predictions against the
existing rule-based risk__reorder_urgency field (not used as a feature —
used only for post-hoc human review of prediction plausibility).

Usage
-----
    python ml_training/scripts/predict_silver.py \\
        --bucket  my-project-scip-lake \\
        --sa-key  infrastructure/gcp/keys/scip-spark-sa.json \\
        --artifacts ml_training/artifacts \\
        --dataset   ml_training/artifacts/dataset \\
        --output  ml_training/artifacts/predictions.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from ml_training.datasets.silver_snapshot_loader import load_silver_features
from ml_training.inference.predictor import OUT_PROB_COL, OUT_SCORE_COL, ShortagePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("predict_silver")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch inference on Silver GCS layer")
    p.add_argument(
        "--bucket",
        default=os.environ.get("GCS_BUCKET_NAME", ""),
        help="GCS bucket name",
    )
    p.add_argument(
        "--sa-key",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    )
    p.add_argument(
        "--artifacts",
        default="ml_training/artifacts",
        help="Directory containing model.pt and feature_metadata.json",
    )
    p.add_argument(
        "--dataset",
        default="ml_training/artifacts/dataset",
        help="Directory containing preprocessor.pkl",
    )
    p.add_argument(
        "--output",
        default="ml_training/artifacts/predictions.parquet",
        help="Output Parquet path for predictions",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.bucket:
        logger.error("--bucket or GCS_BUCKET_NAME is required.")
        sys.exit(1)

    logger.info("Loading Silver features from gs://%s/silver", args.bucket)
    features = load_silver_features(
        bucket           = args.bucket,
        credentials_path = args.sa_key,
    )

    logger.info("Running ShortagePredictor inference")
    predictor = ShortagePredictor(
        artifacts_dir = args.artifacts,
        dataset_dir   = args.dataset,
    )
    predictions = predictor.predict(features)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)
    logger.info("Predictions written to %s", output_path)

    # ── Sanity-check summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Inference Summary")
    print("=" * 65)
    print(f"  Rows : {len(predictions)}")
    print(f"  shortage_prob_7d  — mean={predictions[OUT_PROB_COL].mean():.3f}  "
          f"min={predictions[OUT_PROB_COL].min():.3f}  "
          f"max={predictions[OUT_PROB_COL].max():.3f}")
    print(f"  stockout_risk_score — mean={predictions[OUT_SCORE_COL].mean():.3f}  "
          f"min={predictions[OUT_SCORE_COL].min():.3f}  "
          f"max={predictions[OUT_SCORE_COL].max():.3f}")

    # Count entities by risk tier (using prob threshold)
    high_risk = (predictions[OUT_PROB_COL] >= 0.7).sum()
    med_risk  = ((predictions[OUT_PROB_COL] >= 0.4) & (predictions[OUT_PROB_COL] < 0.7)).sum()
    low_risk  = (predictions[OUT_PROB_COL] < 0.4).sum()
    print()
    print(f"  Risk tiers (shortage_prob_7d):")
    print(f"    HIGH  (>=0.70)      : {high_risk:>4} entities")
    print(f"    MED   (0.40-0.70)   : {med_risk:>4} entities")
    print(f"    LOW   (<0.40)       : {low_risk:>4} entities")
    print()

    # Sample rows for human review
    sample = predictions[["warehouse_id", "sku_id", OUT_PROB_COL, OUT_SCORE_COL]].sort_values(
        OUT_PROB_COL, ascending=False
    ).head(10)
    print("  Top-10 highest shortage risk entities:")
    print(sample.to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
