"""
Regression tests for the drift monitor's PSI computation and its reference.

Two bugs are pinned here:

1. Normal-quantile bins (the original implementation) assumed every monitored
   feature is Gaussian. For the skewed and clustered features monitored here
   that packed most training mass into one or two bins and reported PSI >> 0.20
   for the training data compared with itself. The fix was empirical decile
   bins with each bin's real training fraction as `expected`.
2. Units. That fix read its bins from train.parquet while the parquet splits
   were StandardScaler-transformed, so standardized training bins were
   compared with raw live values and PSI came out at 2-8 for train-vs-itself.
   The splits are now stored in raw units, so both sides of the comparison are
   produced by engineer_features() on raw features.

The reference itself now lives in drift_reference.json, written next to the
model by the trainer (ml_training/evaluation/drift_reference.py), so the
monitor's baseline travels with the model artifacts. There is no approximate
fallback: a missing reference is an error.

DriftMonitor.__init__ needs a live Bigtable client, so these tests build an
instance via __new__ and call the pure methods directly.
"""
import numpy as np
import pandas as pd
import pytest

from ml_training.datasets.feature_engineer import engineer_features
from ml_training.evaluation.drift_reference import (
    MONITORED_FEATURES,
    N_BINS,
    build_drift_reference,
    load_drift_reference,
    write_drift_reference,
)
from model_serving.monitoring import drift_monitor as dm

NAMES = [name for _, name in MONITORED_FEATURES]


def _monitor(artifacts_dir=None) -> dm.DriftMonitor:
    m = dm.DriftMonitor.__new__(dm.DriftMonitor)
    if artifacts_dir is not None:
        m._artifacts_dir = artifacts_dir
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


def _reference_via_file(df: pd.DataFrame, tmp_path):
    """Build the reference, write it as the trainer does, load it as the monitor does."""
    write_drift_reference(build_drift_reference(df), tmp_path / "drift_reference.json")
    return _monitor(tmp_path)._load_drift_reference()


def _bins(values: np.ndarray):
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, N_BINS + 1)))
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
    """Both sides in raw units: the reference from the raw training split as
    the trainer writes it, live values as _collect_feature_values() derives
    them (engineer_features on raw Bigtable cells). Feeding the training rows
    back as 'live' must read as stable for every monitored feature."""
    train_df = _raw_training_frame()
    bins, info = _reference_via_file(train_df, tmp_path)
    assert set(bins) == set(NAMES)
    assert info["n_rows"] == len(train_df) and info["created_at"]

    monitor = _monitor(tmp_path)
    for feature in NAMES:
        edges, expected = bins[feature]
        assert pytest.approx(expected.sum(), abs=1e-9) == 1.0
        psi = monitor._compute_psi(train_df[feature].to_numpy(dtype=float), edges, expected)
        assert psi < 0.01, (feature, psi)


def test_standardized_training_rows_would_have_falsely_triggered(tmp_path):
    """The units bug, kept as a regression: a reference built from
    StandardScaler-transformed columns (as the old train.parquet held them)
    against raw live values reports a TRIGGER for the training data compared
    with itself."""
    train_df = _raw_training_frame()
    scaled   = train_df.copy()
    for feature in NAMES:
        col = train_df[feature].to_numpy(dtype=float)
        scaled[feature] = (col - col.mean()) / col.std()
    bins, _ = _reference_via_file(scaled, tmp_path)

    monitor = _monitor(tmp_path)
    for feature in NAMES:
        edges, expected = bins[feature]
        psi = monitor._compute_psi(train_df[feature].to_numpy(dtype=float), edges, expected)
        assert psi >= dm._PSI_WARNING_THRESHOLD, (feature, psi)


def test_reference_handles_heavily_clustered_values(tmp_path):
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
    bins, _ = _reference_via_file(train_df, tmp_path)
    edges, expected = bins["supplier_reliability"]
    assert len(edges) < N_BINS + 1                 # duplicate edges were deduped
    assert pytest.approx(expected.sum(), abs=1e-9) == 1.0
    assert _monitor(tmp_path)._compute_psi(clustered, edges, expected) < 0.01


def test_reference_round_trip_preserves_bins(tmp_path):
    train_df = _raw_training_frame(n=500)
    ref  = build_drift_reference(train_df)
    path = write_drift_reference(ref, tmp_path / "drift_reference.json")
    bins, info = load_drift_reference(path)
    for name, spec in ref["features"].items():
        edges, expected = bins[name]
        assert edges[0] == -np.inf and edges[-1] == np.inf
        np.testing.assert_allclose(edges[1:-1], spec["interior_edges"])
        np.testing.assert_allclose(expected, spec["expected"])
    assert info["n_rows"] == 500 and info["n_bins"] == N_BINS


def test_missing_reference_is_an_error_not_a_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        _monitor(tmp_path)._load_drift_reference()
