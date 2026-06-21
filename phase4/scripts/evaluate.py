"""
evaluate.py — CLI entry point for Phase 4 model evaluation.

Loads model artifacts, runs inference on a dataset split, computes
AUC-ROC / AUPRC / Brier for both model and baseline, and prints the
Phase 4 success verdict.

Usage
-----
    python phase4/scripts/evaluate.py \\
        --artifacts phase4/artifacts \\
        --dataset   phase4/artifacts/dataset \\
        --split     test
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from phase4.evaluation.evaluate import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Phase 4 ShortagePredictor vs baseline")
    p.add_argument("--artifacts", default="phase4/artifacts",         help="Directory with model.pt")
    p.add_argument("--dataset",   default="phase4/artifacts/dataset", help="Directory with split Parquet")
    p.add_argument("--split",     default="test", choices=["train", "val", "test"])
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    report = run_evaluation(
        artifacts_dir = args.artifacts,
        dataset_dir   = args.dataset,
        split         = args.split,
    )
    sys.exit(0 if report.get("phase4_passed") else 1)


if __name__ == "__main__":
    main()
