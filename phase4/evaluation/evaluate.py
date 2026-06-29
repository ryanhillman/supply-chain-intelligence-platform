"""
evaluate.py — standalone evaluation runner.

Loads model artifacts from output_dir, runs inference on a dataset split,
computes full metric comparison against the heuristic baseline, and saves
an evaluation report JSON.

Called by scripts/evaluate.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from phase4.datasets.feature_engineer import (
    BASELINE_COL,
    NUMERIC_FEATURE_COLS,
    TARGET_COL,
)
from phase4.evaluation.metrics import compare_to_baseline, per_warehouse_metrics
from phase4.training.config import TrainingConfig
from phase4.training.model import ShortagePredictor

logger = logging.getLogger(__name__)


def run_evaluation(
    artifacts_dir: str | Path,
    dataset_dir:   str | Path,
    split:         str = "test",
) -> dict:
    """
    Evaluate the trained model against the heuristic baseline.

    Parameters
    ----------
    artifacts_dir:
        Directory containing model.pt (trained ShortagePredictor state dict).
    dataset_dir:
        Directory containing {split}.parquet, feature_metadata.json,
        preprocessor.pkl, and baseline_metrics.json.
    split:
        "train", "val", or "test".

    Returns
    -------
    dict with keys: model, baseline, improvements, outperforms_baseline,
                    per_warehouse, phase4_passed.
    """
    artifacts_dir = Path(artifacts_dir)
    dataset_dir   = Path(dataset_dir)

    metadata   = _load_json(dataset_dir / "feature_metadata.json")
    df         = pd.read_parquet(dataset_dir / f"{split}.parquet")

    wh_vocab   = metadata["warehouse_vocab"]
    sku_vocab  = metadata["sku_vocab"]
    checkpoint = torch.load(artifacts_dir / "model.pt", map_location="cpu", weights_only=False)
    config     = checkpoint.get("config", TrainingConfig())

    model = ShortagePredictor(
        n_numeric    = metadata["n_numeric_features"],
        n_warehouses = metadata["n_warehouses"],
        n_skus       = metadata["n_skus"],
        config       = config,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Build tensors
    x_num   = torch.tensor(df[NUMERIC_FEATURE_COLS].values, dtype=torch.float32)
    wh_ids  = torch.tensor([wh_vocab[w] for w in df["warehouse_id"]], dtype=torch.long)
    sku_ids = torch.tensor([sku_vocab[s] for s in df["sku_id"]],       dtype=torch.long)

    with torch.no_grad():
        prob, score = model.predict(x_num, wh_ids, sku_ids)

    y_prob_model = prob.squeeze(1).numpy()
    y_true       = df[TARGET_COL].values.astype(float)
    y_baseline   = df[BASELINE_COL].values.astype(float)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        logger.warning("Split '%s' has only one class — metrics may be degenerate.", split)

    comparison   = compare_to_baseline(y_true, y_prob_model, y_baseline)
    wh_breakdown = per_warehouse_metrics(df, y_prob_model)

    report = {
        "split":                split,
        "n_examples":           len(df),
        "positive_rate":        float(y_true.mean()),
        "model":                comparison["model"],
        "baseline":             comparison["baseline"],
        "improvements":         comparison["improvements"],
        "outperforms_baseline": comparison["outperforms_baseline"],
        "per_warehouse":        wh_breakdown.to_dict(orient="records"),
        "phase4_passed":        comparison["outperforms_baseline"],
    }

    out_path = artifacts_dir / f"evaluation_{split}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Phase 4 Evaluation — {report['split'].upper()} split")
    print(sep)
    print(f"  Examples : {report['n_examples']}")
    print(f"  Pos rate : {report['positive_rate']:.3f}")
    print()
    print(f"  {'Metric':<12} {'Baseline':>10} {'Model':>10} {'Delta':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for metric in ["auc_roc", "auprc", "brier"]:
        base_v  = report["baseline"][metric]
        model_v = report["model"][metric]
        delta   = report["improvements"][metric]
        sign    = "+" if delta > 0 else ""
        print(f"  {metric:<12} {base_v:>10.4f} {model_v:>10.4f} {sign}{delta:>9.4f}")
    print()
    status = "PASSED" if report["phase4_passed"] else "FAILED"
    print(f"  Phase 4 success criterion: {status}")
    print(f"  (Model outperforms baseline on all three metrics)")
    print(sep)

    if report["per_warehouse"]:
        print("\n  Per-warehouse breakdown:")
        print(f"  {'Warehouse':<12} {'n':>5} {'Pos%':>6} {'Mdl AUC':>9} {'Base AUC':>9}")
        print(f"  {'-'*12} {'-'*5} {'-'*6} {'-'*9} {'-'*9}")
        for row in report["per_warehouse"]:
            print(
                f"  {row['warehouse_id']:<12} {row['n']:>5} "
                f"{row['positive_rate']:>5.1%} "
                f"{row['model_auc_roc']:>9.4f} {row['baseline_auc_roc']:>9.4f}"
            )
    print()


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
