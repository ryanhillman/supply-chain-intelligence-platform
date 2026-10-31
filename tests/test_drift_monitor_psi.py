"""
Regression tests for model_serving/monitoring/drift_monitor.py's PSI computation.

Two bugs are pinned here:

1. Normal-quantile bins (the original implementation) assumed every monitored
   feature is Gaussian. For the skewed and clustered features monitored here
   that packed most training mass into one or two bins and reported PSI >> 0.20
   for the training data compared with itself. The fix was empirical decile
   bins with each bin's real training fraction as `expected`.
2. Units. That fix read its bins from train.parquet while the parquet splits
   were StandardScaler-transformed, so standardized training bins were
   compared with raw live values and PSI came out at 2-8 for train-vs-itself.
   The splits are now stored in raw units (DatasetBuilder no longer transforms
   them), so both sides of the comparison are produced by engineer_features()
   on raw features. There is no approximate fallback any more: train.parquet
   is required.

DriftMonitor.__init__ needs a live Bigtable client, so these tests build an
instance via __new__ and call the pure methods (_compute_psi,
_load_training_quantiles) directly; neither touches Bigtable.
"""
import numpy as np
import pandas as pd
import pytest

from ml_training.datasets.feature_engineer import engineer_features
from model_serving.monitoring import drift_monitor as dm

MONITORED = [name for _, name in dm._MONITORED_FEATURES]


def _monitor(dataset_dir=None) -> dm.DriftMonitor:
    m = dm.DriftMonitor.__new__(dm.DriftMonitor)
    if dataset_dir is not None:
        m._dataset_dir = dataset_dir
    return m


def _raw_training_frame(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Raw Silver-scale features with the shapes the real ones have:
    skewed sales, clustered supplier_reliability, some zero-stock rows."""
    rng  = np.random.default_rng(seed)
    avg7 = rng.exponential(35.0, n)
    df = pd.DataFrame({
        "current_qty":             np.where(rng.random(n) < 0.07, 0.0, rng.exponential(400.0, n)),
        "reorder_point":           rng.exponential(500.0, n),
        "safety_stock":            rng.exponential(200.0, n),
        "sales_7d":                avg7 * 7,
        "sales_30d":               avg7 * 30 * rng.uniform(0.8, 1.2, n),
        "avg_daily_sales_7d":      avg7,
        "avg_daily_sales_30d":     avg7 * rng.uniform(0.8, 1.2, n),
        "velocity_trend":          rng.uniform(0.8, 1.2, n),
        "stddev_daily_7d":         avg7 * 0.3,
        "lead_time_days":          rng.uniform(5, 20, n),
        "lead_time_stddev":        rng.uniform(0.5, 3, n),
        "supplier_reliability":    np.where(rng.random(n) < 0.5, 0.93, rng.uniform(0.7, 1.0, n)),
        "active_po_qty":           rng.exponential(2500.0, n),
        "in_transit_qty":          rng.exponential(800.0, n),
        "demand_signal_count_24h": rng.poisson(4, n).astype(float),
    })
    return engineer_features(df)


def _bins(values: np.ndarray):
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, dm._N_BINS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(values, bins=edges)
    return edges, counts.astype(float) / len(values)


def test_empirical_bins_give_zero_psi_for_skewed_data_against_itself():
    rng = np.random.default_rng(0)
    values = rng.exponential(scale=100.0, size=5000)
    edges, expected = _bins(values)
    assert _monitor()._compute_psi(values, edges, expected) < 0.01


def test_empirical_bins_still_detect_genuine_drift():
    rng = np.random.default_rng(0)
    values = rng.exponential(scale=100.0, size=5000)
    edges, expected = _bins(values)
    assert _monitor()._compute_psi(values * 2.0, edges, expected) >= 0.20


def test_training_rows_through_the_live_path_give_zero_psi(tmp_path):
    """Both sides in raw units: train.parquet as DatasetBuilder now writes it,
    live values as _collect_feature_values() derives them (engineer_features
    on raw Bigtable cells). Feeding the training rows back as 'live' must
    read as stable for every monitored feature."""
    train_df = _raw_training_frame()
    train_df.to_parquet(tmp_path / "train.parquet")

    monitor   = _monitor(tmp_path)
    quantiles = monitor._load_training_quantiles()
    assert set(quantiles) == set(MONITORED)

    for feature in MONITORED:
        edges, expected = quantiles[feature]
        assert pytest.approx(expected.sum(), abs=1e-9) == 1.0
        psi = monitor._compute_psi(train_df[feature].to_numpy(dtype=float), edges, expected)
        assert psi < 0.01, (feature, psi)


def test_standardized_training_rows_would_have_falsely_triggered(tmp_path):
    """The units bug, kept as a regression: if train.parquet held
    StandardScaler-transformed columns (as it used to), raw live values land
    almost entirely in one bin and PSI reports a TRIGGER for the training
    data compared with itself."""
    train_df = _raw_training_frame()
    scaled   = train_df.copy()
    for feature in MONITORED:
        col = train_df[feature].to_numpy(dtype=float)
        scaled[feature] = (col - col.mean()) / col.std()
    scaled.to_parquet(tmp_path / "train.parquet")

    monitor   = _monitor(tmp_path)
    quantiles = monitor._load_training_quantiles()
    for feature in MONITORED:
        edges, expected = quantiles[feature]
        psi = monitor._compute_psi(train_df[feature].to_numpy(dtype=float), edges, expected)
        assert psi >= dm._PSI_WARNING_THRESHOLD, (feature, psi)


def test_load_training_quantiles_handles_heavily_clustered_values(tmp_path):
    """supplier_reliability: one value accounts for ~half of all rows, which
    collapses several decile edges onto the same number. Deduped edges with
    each bin's real training fraction must still give ~0 PSI train-vs-itself."""
    rng = np.random.default_rng(0)
    clustered = np.concatenate([np.full(5000, 0.73), rng.uniform(0.0, 0.73, size=5000)])
    train_df = pd.DataFrame({
        "days_of_supply":       rng.normal(20, 5, size=10000),
        "sales_7d":             rng.exponential(500, size=10000),
        "avg_daily_sales_7d":   rng.exponential(70, size=10000),
        "supplier_reliability": clustered,
    })
    train_df.to_parquet(tmp_path / "train.parquet")

    monitor = _monitor(tmp_path)
    edges, expected = monitor._load_training_quantiles()["supplier_reliability"]
    assert len(edges) < dm._N_BINS + 1          # duplicate edges were deduped
    assert pytest.approx(expected.sum(), abs=1e-9) == 1.0
    assert monitor._compute_psi(clustered, edges, expected) < 0.01


def test_missing_train_parquet_is_an_error_not_a_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        _monitor(tmp_path)._load_training_quantiles()
