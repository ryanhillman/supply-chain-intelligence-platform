"""
LabelGenerator — attaches training targets to each (warehouse, sku, sample_date)
feature row produced by BronzeTemporalSampler.

Two targets are produced:

  stockout_label_7d (int, 0 or 1)
      Ground truth binary label: did inventory quantity_after reach 0 at any
      point during the 7 full calendar days after sample_date — i.e. any
      time from the start of day+1 through the end of day+7?
      Derived from actual Bronze inventory events — no information from the
      feature snapshot leaks into this label.

  baseline_pred (float, 0.0 or 1.0)
      The heuristic comparator: 1.0 when days_of_supply < HORIZON_DAYS, else 0.0.
      This is the rule-based threshold the PyTorch model must outperform.
      days_of_supply = current_qty / max(avg_daily_sales_7d, EPS)

  continuous_risk_score (float, 0.0–1.0)
      Stochastic probability label used as the regression target for the
      risk_score output head. Computed via a Normal approximation:

          P(stockout ≤ horizon) ≈ Φ((horizon − μ) / σ)

      where:
          μ = current_qty / avg_daily_sales_7d          (mean days-to-zero)
          σ = current_qty × stddev_daily_7d / avg²      (std of days-to-zero)

      When σ is near zero (deterministic demand), this degenerates cleanly to
      the step function baseline_pred. The model learns to improve on both.
"""
from __future__ import annotations

import logging

import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

HORIZON_DAYS: int = 7
EPS: float = 1e-6   # prevents division by zero in days_of_supply and σ


def generate_labels(
    feature_rows: pd.DataFrame,
    inv_events: pd.DataFrame,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Attach stockout_label_7d, baseline_pred, and continuous_risk_score to
    each row in `feature_rows`.

    Parameters
    ----------
    feature_rows:
        Output of BronzeTemporalSampler.build(). Must contain columns:
        warehouse_id, sku_id, sample_date, current_qty, avg_daily_sales_7d,
        stddev_daily_7d.
    inv_events:
        Full inventory event log from BronzeTemporalSampler.build().
        Must contain: warehouse_id, sku_id, quantity_after, ts (datetime64[ns, UTC]).
    horizon_days:
        Prediction horizon in calendar days (default 7).

    Returns
    -------
    pd.DataFrame
        feature_rows with three new columns appended.
    """
    # Build a fast lookup: {(warehouse_id, sku_id): sorted events DataFrame}
    event_index: dict[tuple, pd.DataFrame] = {}
    for (wh, sku), grp in inv_events.groupby(["warehouse_id", "sku_id"]):
        event_index[(wh, sku)] = grp.sort_values("ts").reset_index(drop=True)

    labels = _compute_labels(feature_rows, event_index, horizon_days)

    result = feature_rows.copy()
    result["stockout_label_7d"]    = labels["stockout_label_7d"].values
    result["baseline_pred"]        = labels["baseline_pred"].values
    result["continuous_risk_score"] = labels["continuous_risk_score"].values

    pos_rate  = result["stockout_label_7d"].mean()
    logger.info(
        "Labels generated: %d rows | positive rate=%.3f (stockout_label_7d=1)",
        len(result), pos_rate,
    )
    logger.info(
        "Baseline pred positive rate=%.3f",
        result["baseline_pred"].mean(),
    )
    return result


def _compute_labels(
    feature_rows: pd.DataFrame,
    event_index: dict[tuple, pd.DataFrame],
    horizon_days: int,
) -> pd.DataFrame:
    """Core label computation — iterates feature rows once."""
    # Window offsets relative to sample_date: [day+1, day+horizon_days], i.e.
    # everything from the start of day+1 up to (but not including) the start
    # of day+horizon_days+1. Using sample_date + horizon_days as the upper
    # bound would land exactly at midnight of day+horizon_days, excluding
    # every event later that day.
    start_offset = pd.Timedelta(days=1)
    end_offset   = pd.Timedelta(days=horizon_days + 1)
    rows: list[dict] = []

    for row in feature_rows.itertuples(index=False):
        wh   = row.warehouse_id
        sku  = row.sku_id
        d    = row.sample_date  # datetime.date

        t_start = pd.Timestamp(d, tz="UTC") + start_offset
        t_end   = pd.Timestamp(d, tz="UTC") + end_offset

        # Ground-truth binary label from future Bronze events
        stockout = 0
        grp = event_index.get((wh, sku))
        if grp is not None:
            future = grp[(grp["ts"] >= t_start) & (grp["ts"] < t_end)]
            if not future.empty and future["quantity_after"].min() == 0:
                stockout = 1

        # Heuristic baseline: threshold on days-of-supply
        avg_d7    = float(getattr(row, "avg_daily_sales_7d", 0.0) or 0.0)
        qty       = float(getattr(row, "current_qty", 0.0) or 0.0)
        dos       = qty / max(avg_d7, EPS)
        baseline  = 1.0 if dos < horizon_days else 0.0

        # Stochastic continuous risk score
        continuous = _normal_stockout_prob(
            current_qty=qty,
            avg_daily=avg_d7,
            stddev_daily=float(getattr(row, "stddev_daily_7d", 0.0) or 0.0),
            horizon_days=horizon_days,
        )

        rows.append({
            "stockout_label_7d":     stockout,
            "baseline_pred":         baseline,
            "continuous_risk_score": continuous,
        })

    return pd.DataFrame(rows)


def _normal_stockout_prob(
    current_qty: float,
    avg_daily: float,
    stddev_daily: float,
    horizon_days: int,
) -> float:
    """
    P(days-to-zero ≤ horizon) under a Normal approximation.

    Let T = time-to-stockout = Q / D where D ~ N(μ_D, σ_D²) per day.
    By delta method: T ≈ N(Q/μ_D, (Q·σ_D/μ_D²)²) for μ_D > 0.

    When avg_daily ≤ EPS (no demand), supply is effectively infinite → return 0.0.
    When stddev_daily ≤ EPS (deterministic demand), degenerate to step function.
    """
    if avg_daily <= EPS:
        return 0.0

    mu_t  = current_qty / avg_daily
    sig_t = (current_qty * stddev_daily) / (avg_daily ** 2)

    if sig_t <= EPS:
        return 1.0 if mu_t < horizon_days else 0.0

    return float(norm.cdf(horizon_days, loc=mu_t, scale=sig_t))
