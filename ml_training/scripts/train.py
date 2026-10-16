"""
train.py — CLI entry point for Phase 4 model training.

Loads the dataset built by build_dataset.py, trains ShortagePredictor,
evaluates on the test split, exports to ONNX, and optionally uploads
all artifacts to GCS for Phase 5 readiness.

Usage
-----
    python ml_training/scripts/train.py \\
        --dataset ml_training/artifacts/dataset \\
        --artifacts ml_training/artifacts \\
        --epochs 100 \\
        --batch-size 64

Optional GCS upload:
    python ml_training/scripts/train.py ... --upload-bucket my-project-scip-lake
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from ml_training.evaluation.evaluate import run_evaluation
from ml_training.inference.onnx_export import export_to_onnx
from ml_training.training.config import TrainingConfig
from ml_training.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Phase 4 ShortagePredictor")
    p.add_argument(
        "--dataset",
        default="ml_training/artifacts/dataset",
        help="Dataset directory (output of build_dataset.py)",
    )
    p.add_argument(
        "--artifacts",
        default="ml_training/artifacts",
        help="Output directory for model.pt, ONNX, and metrics",
    )
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument(
        "--loss",
        choices=["weighted_bce", "focal"],
        default="weighted_bce",
    )
    p.add_argument(
        "--upload-bucket",
        default=None,
        help="GCS bucket to upload artifacts to (optional). "
             "Writes to gs://<bucket>/models/shortage-predictor/v<timestamp>/",
    )
    p.add_argument(
        "--sa-key",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    )
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    dataset = Path(args.dataset)
    output  = Path(args.artifacts)
    output.mkdir(parents=True, exist_ok=True)

    if not (dataset / "train.parquet").exists():
        logger.error(
            "train.parquet not found in %s. Run build_dataset.py first.", dataset
        )
        sys.exit(1)

    # Copy feature_metadata.json into artifacts dir so predictor.py finds it
    import shutil
    meta_src = dataset / "feature_metadata.json"
    meta_dst = output / "feature_metadata.json"
    if meta_src != meta_dst:
        shutil.copy2(meta_src, meta_dst)

    config = TrainingConfig(
        max_epochs = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        patience   = args.patience,
        loss_type  = args.loss,
        dataset_dir = str(dataset),
        output_dir  = str(output),
    )

    logger.info("Starting training — config: %s", config)
    trainer = Trainer(config=config, dataset_dir=dataset, output_dir=output)
    history = trainer.train()

    logger.info("Evaluating on test split")
    report = run_evaluation(
        artifacts_dir = output,
        dataset_dir   = dataset,
        split         = "test",
    )

    # Export ONNX
    logger.info("Exporting to ONNX")
    onnx_path = export_to_onnx(artifacts_dir=output)
    logger.info("ONNX written to %s", onnx_path)

    # Optional GCS upload
    if args.upload_bucket:
        _upload_to_gcs(output, args.upload_bucket, args.sa_key)

    ml_training_status = "PASSED" if report.get("ml_training_passed") else "FAILED"
    print(f"\nPhase 4 training complete — success criterion: {ml_training_status}")
    if not report.get("ml_training_passed"):
        print("  Model did not outperform baseline on all metrics.")
        print("  Consider: more data, longer training, hyperparameter tuning.")


def _upload_to_gcs(artifacts_dir: Path, bucket: str, sa_key: str | None) -> None:
    """Upload model artifacts to GCS at a timestamped version path."""
    from datetime import datetime, timezone
    import gcsfs

    version    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    gcs_prefix = f"{bucket}/models/shortage-predictor/v{version}"
    fs         = gcsfs.GCSFileSystem(token=sa_key if sa_key else "google_default")

    upload_files = [
        "model.pt",
        "model.onnx",
        "feature_metadata.json",
        "training_metrics.json",
        "evaluation_test.json",
    ]
    for fname in upload_files:
        src = artifacts_dir / fname
        if src.exists():
            dst = f"{gcs_prefix}/{fname}"
            fs.put(str(src), dst)
            logger.info("Uploaded %s → gs://%s", fname, dst)

    logger.info("Artifacts uploaded to gs://%s/", gcs_prefix)


if __name__ == "__main__":
    main()
