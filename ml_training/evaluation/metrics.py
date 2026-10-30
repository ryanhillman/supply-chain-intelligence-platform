"""
Metric computation for the Phase 4 baseline comparison framework.

Phase 4 success criterion: PyTorch model outperforms the closed-form baseline
  continuous_risk_score = P(stockout ≤ 7d) under a Normal approximation
                        = Φ((7 − μ) / σ),  μ = qty / avg_7d,  σ = qty·std_7d / avg_7d²
on held-out test data across all three metrics:
  AUC-ROC  — higher is better
  AUPRC    — higher is better (primary: more informative under class imbalance)
  Brier    — lower is better

The 0/1 rule baseline_pred = float(days_of_supply < 7) is reported alongside
for context only. A binary predictor has structurally low AUC-ROC and AUPRC,
so beating it proves little.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """AUC-ROC, AUPRC, and Brier score for a single predictor."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auprc":   float(average_precision_score(y_true, y_prob)),
        "brier":   float(brier_score_loss(y_true, y_prob)),
    }


def compare_to_baseline(
    y_true:          np.ndarray,
    y_prob_model:    np.ndarray,
    y_prob_baseline: np.ndarray,
) -> dict:
    """
    Compare model against baseline on all three metrics.

    Returns a dict with keys:
      model      — {auc_roc, auprc, brier}
      baseline   — {auc_roc, auprc, brier}
      outperforms_baseline — True only when model beats baseline on ALL three
      improvements — per-metric delta (positive = better for model)
    """
    model_m    = compute_metrics(y_true, y_prob_model)
    baseline_m = compute_metrics(y_true, y_prob_baseline)

    improvements = {
        "auc_roc": model_m["auc_roc"] - baseline_m["auc_roc"],
        "auprc":   model_m["auprc"]   - baseline_m["auprc"],
        "brier":   baseline_m["brier"] - model_m["brier"],   # positive = model lower (better)
    }

    outperforms = all(v > 0.0 for v in improvements.values())

    return {
        "model":                model_m,
        "baseline":             baseline_m,
        "improvements":         improvements,
        "outperforms_baseline": outperforms,
    }


def per_warehouse_metrics(
    df:           pd.DataFrame,
    y_prob:       np.ndarray,
    label_col:    str = "stockout_label_7d",
    baseline_col: str = "continuous_risk_score",
) -> pd.DataFrame:
    """
    Compute model vs baseline metrics broken down by warehouse_id.

    Warehouses where the test split has only one class (all 0s or all 1s)
    cannot have AUC-ROC computed and are skipped.
    """
    df = df.copy()
    df["_model_prob"] = y_prob
    df["_baseline"]   = df[baseline_col].values.astype(float)

    rows = []
    for wh, grp in df.groupby("warehouse_id"):
        y_t   = grp[label_col].values.astype(float)
        y_m   = grp["_model_prob"].values
        y_b   = grp["_baseline"].values

        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            continue   # single-class partition: AUC-ROC undefined

        rows.append({
            "warehouse_id":     wh,
            "n":                len(grp),
            "positive_rate":    float(y_t.mean()),
            "model_auc_roc":    float(roc_auc_score(y_t, y_m)),
            "model_auprc":      float(average_precision_score(y_t, y_m)),
            "model_brier":      float(brier_score_loss(y_t, y_m)),
            "baseline_auc_roc": float(roc_auc_score(y_t, y_b)),
            "baseline_auprc":   float(average_precision_score(y_t, y_b)),
            "baseline_brier":   float(brier_score_loss(y_t, y_b)),
        })

    return pd.DataFrame(rows)
