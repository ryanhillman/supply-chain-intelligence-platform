"""
evaluate.py — standalone evaluation runner.

Loads model artifacts from output_dir, runs inference on a dataset split,
compares the model against the closed-form Normal-approximation baseline
(continuous_risk_score) with the days_of_supply < 7 rule reported for
context, and saves an evaluation report JSON.

Called by scripts/evaluate.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml_training.datasets.feature_engineer import (
    BASELINE_COL,
    NUMERIC_FEATURE_COLS,
    RISK_SCORE_COL,
    TARGET_COL,
)
from ml_training.evaluation.metrics import (
    compare_to_baseline,
    compute_metrics,
    per_warehouse_metrics,
)
from ml_training.training.config import TrainingConfig
from ml_training.training.model import ShortagePredictor

logger = logging.getLogger(__name__)


def run_evaluation(
    artifacts_dir: str | Path,
    dataset_dir:   str | Path,
    split:         str = "test",
) -> dict:
    """
    Evaluate the trained model against the closed-form formula baseline.

    Parameters
    ----------
    artifacts_dir:
        Directory containing model.pt (trained ShortagePredictor state dict).
    dataset_dir:
        Directory containing {split}.parquet, feature_metadata.json and
        baseline_metrics.json.
    split:
        "train", "val", or "test".

    Returns
    -------
    dict with keys: model, baseline, baseline_description, rule_baseline,
                    improvements, outperforms_baseline, in_stock, per_warehouse,
                    split_sha256 (hash of the parquet file the metrics came from;
                    promote_model.py uses it to tell whether two models' metrics
                    are comparable).
    """
    artifacts_dir = Path(artifacts_dir)
    dataset_dir   = Path(dataset_dir)

    metadata   = _load_json(dataset_dir / "feature_metadata.json")
    df         = pd.read_parquet(dataset_dir / f"{split}.parquet")

    wh_vocab   = metadata["warehouse_vocab"]
    sku_vocab  = metadata["sku_vocab"]
    checkpoint = torch.load(artifacts_dir / "model.pt", map_location="cpu", weights_only=False)
    config     = checkpoint.get("config", TrainingConfig())

    model = ShortagePredictor.from_metadata(metadata, config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Build tensors
    x_num   = torch.tensor(df[NUMERIC_FEATURE_COLS].values, dtype=torch.float32)
    wh_ids  = torch.tensor([wh_vocab[w] for w in df["warehouse_id"]], dtype=torch.long)
    sku_ids = torch.tensor([sku_vocab[s] for s in df["sku_id"]],       dtype=torch.long)

    with torch.no_grad():
        prob = model.predict(x_num, wh_ids, sku_ids)

    y_prob_model = prob.squeeze(1).numpy()
    y_true       = df[TARGET_COL].values.astype(float)
    y_baseline   = df[RISK_SCORE_COL].values.astype(float)   # the bar: closed-form formula
    y_rule       = df[BASELINE_COL].values.astype(float)     # context: 0/1 rule

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        logger.warning("Split '%s' has only one class — metrics may be degenerate.", split)

    comparison   = compare_to_baseline(y_true, y_prob_model, y_baseline)
    rule_metrics = compute_metrics(y_true, y_rule)
    wh_breakdown = per_warehouse_metrics(df, y_prob_model, baseline_col=RISK_SCORE_COL)

    # In-stock subset. An entity already at zero stays at zero until a PO
    # lands, so it is labelled positive almost by definition and flatters
    # every predictor on the full split. The operational question is "of
    # what is in stock today, what runs out within 7 days?" — report that
    # alongside. current_qty is in raw units in the parquet splits.
    in_stock_block = _subset_metrics(
        mask       = df["current_qty"].values > 0,
        y_true     = y_true,
        y_model    = y_prob_model,
        y_baseline = y_baseline,
        y_rule     = y_rule,
        label      = "current_qty > 0",
    )

    report = {
        "split":                split,
        "split_sha256":         _file_sha256(dataset_dir / f"{split}.parquet"),
        "n_examples":           len(df),
        "positive_rate":        float(y_true.mean()),
        "model":                comparison["model"],
        "baseline":             comparison["baseline"],
        "baseline_description": "Normal-approx P(stockout <= 7d)",
        "rule_baseline":        rule_metrics,   # days_of_supply < 7, for context
        "improvements":         comparison["improvements"],
        "outperforms_baseline": comparison["outperforms_baseline"],
        "in_stock":             in_stock_block,
        "per_warehouse":        wh_breakdown.to_dict(orient="records"),
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
    print(f"  {'Metric':<12} {'Rule':>10} {'Formula':>10} {'Model':>10} {'Delta':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for metric in ["auc_roc", "auprc", "brier"]:
        rule_v  = report["rule_baseline"][metric]
        base_v  = report["baseline"][metric]
        model_v = report["model"][metric]
        delta   = report["improvements"][metric]
        sign    = "+" if delta > 0 else ""
        print(
            f"  {metric:<12} {rule_v:>10.4f} {base_v:>10.4f} {model_v:>10.4f} "
            f"{sign}{delta:>9.4f}"
        )
    print()
    status = "PASSED" if report["outperforms_baseline"] else "FAILED"
    print(f"  Phase 4 success criterion: {status}")
    print("  (Model outperforms the Normal-approx formula on all three metrics;")
    print("   Rule = days_of_supply < 7 is shown for context and is not the bar)")

    sub = report.get("in_stock") or {}
    if sub.get("model"):
        zero_rate = sub.get("zero_stock_positive_rate")
        zero_txt  = f", pos rate {zero_rate:.3f}" if zero_rate is not None else ""
        print()
        print(
            f"  In-stock rows only ({sub['description']}): n={sub['n_examples']}  "
            f"pos rate={sub['positive_rate']:.3f}  "
            f"[excluded {sub['zero_stock_rows']} zero-stock rows{zero_txt}]"
        )
        print(f"  {'Metric':<12} {'Rule':>10} {'Formula':>10} {'Model':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        for metric in ["auc_roc", "auprc", "brier"]:
            print(
                f"  {metric:<12} {sub['rule_baseline'][metric]:>10.4f} "
                f"{sub['baseline'][metric]:>10.4f} {sub['model'][metric]:>10.4f}"
            )
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


def _subset_metrics(
    mask:       np.ndarray,
    y_true:     np.ndarray,
    y_model:    np.ndarray,
    y_baseline: np.ndarray,
    y_rule:     np.ndarray,
    label:      str,
) -> dict:
    """
    Model / formula / rule metrics on the rows selected by `mask`, plus the
    size and label rate of both the kept and the excluded rows. Metric
    blocks are None when the subset is empty or single-class.
    """
    n     = int(mask.sum())
    y_sub = y_true[mask]
    excluded = ~mask
    block: dict = {
        "description":              label,
        "n_examples":               n,
        "positive_rate":            float(y_sub.mean()) if n else None,
        "zero_stock_rows":          int(excluded.sum()),
        "zero_stock_positive_rate": float(y_true[excluded].mean()) if excluded.any() else None,
        "model":                    None,
        "baseline":                 None,
        "rule_baseline":            None,
    }
    if n == 0 or y_sub.sum() == 0 or y_sub.sum() == len(y_sub):
        logger.warning("In-stock subset is empty or single-class — subset metrics skipped.")
        return block

    block["model"]         = compute_metrics(y_sub, y_model[mask])
    block["baseline"]      = compute_metrics(y_sub, y_baseline[mask])
    block["rule_baseline"] = compute_metrics(y_sub, y_rule[mask])
    return block


def _file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
