"""
Regression tests for model_serving/monitoring/drift_monitor.py's PSI
computation, covering the false-trigger bug fixed in this branch:
_compute_psi assumed every monitored feature is normally distributed,
building bins from normal quantiles of (scaler.mean_, scaler.scale_).
For skewed or heavily-clustered features (sales, supplier_reliability),
this packed most of the training mass into one or two bins and made PSI
blow up (>> 0.20, a false "retraining required" trigger) even when fed
the training data against itself.

DriftMonitor.__init__ needs a live Bigtable client, so these tests
build an instance via __new__ and call the pure methods
(_compute_psi, _load_training_quantiles) directly -- neither touches
Bigtable; _load_training_quantiles only reads a local parquet path.
"""
import numpy as np
import pandas as pd
import pytest

from model_serving.monitoring import drift_monitor as dm


def _monitor() -> dm.DriftMonitor:
    return dm.DriftMonitor.__new__(dm.DriftMonitor)


def test_psi_is_zero_for_identical_distribution_normal_fallback():
    """Sanity check on the normal-quantile fallback path itself (used only
    when train.parquet is unavailable): a sample truly drawn from
    N(mean, std) should score as stable, not a trigger."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=50.0, scale=10.0, size=5000)
    psi = _monitor()._compute_psi(training_mean=50.0, training_std=10.0, current_values=values)
    assert psi < 0.10  # "stable" per ADR-005 thresholds


def test_psi_false_trigger_on_skewed_data_with_normal_bins():
    """Reproduces the bug directly: an exponential-like (right-skewed)
    distribution fed through the normal-quantile fallback bins falsely
    reads as a massive drift trigger even though it's compared against
    its own distribution's mean/std."""
    rng = np.random.default_rng(0)
    values = rng.exponential(scale=100.0, size=5000)  # skewed, like sales_7d
    mean, std = float(values.mean()), float(values.std())

    psi = _monitor()._compute_psi(training_mean=mean, training_std=std, current_values=values)
    assert psi > 0.20  # confirms the normal-bin approximation is unsound here


def test_empirical_bins_fix_the_false_trigger_on_skewed_data():
    """The actual fix: using the training data's own empirical quantile
    bins (and each bin's real training fraction as `expected`) must give
    ~0 PSI for the same skewed distribution compared against itself."""
    rng = np.random.default_rng(0)
    values = rng.exponential(scale=100.0, size=5000)
    mean, std = float(values.mean()), float(values.std())

    quantile_points = np.linspace(0.0, 1.0, dm._N_BINS + 1)
    edges = np.unique(np.quantile(values, quantile_points))
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(values, bins=edges)
    expected = counts.astype(float) / len(values)

    psi = _monitor()._compute_psi(mean, std, values, bin_edges=edges, expected=expected)
    assert psi < 0.01


def test_empirical_bins_still_detect_genuine_drift():
    """The fix must not make PSI blind -- a real distribution shift
    (values doubled) should still trigger."""
    rng = np.random.default_rng(0)
    values = rng.exponential(scale=100.0, size=5000)
    mean, std = float(values.mean()), float(values.std())

    quantile_points = np.linspace(0.0, 1.0, dm._N_BINS + 1)
    edges = np.unique(np.quantile(values, quantile_points))
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(values, bins=edges)
    expected = counts.astype(float) / len(values)

    shifted = values * 2.0
    psi = _monitor()._compute_psi(mean, std, shifted, bin_edges=edges, expected=expected)
    assert psi >= 0.20  # "trigger" per ADR-005 thresholds


def test_load_training_quantiles_handles_heavily_clustered_values(tmp_path):
    """Reproduces the supplier_reliability case: a single repeated value
    accounts for ~half of all rows, which degenerates naive 10-equal-bin
    quantile edges into duplicates. Deduping edges and using each bin's
    real training fraction as `expected` must still give ~0 PSI for
    train-vs-itself despite the clustering."""
    rng = np.random.default_rng(0)
    clustered = np.concatenate([
        np.full(5000, 0.73),                      # ~50% identical value
        rng.uniform(0.0, 0.73, size=5000),        # the rest spread out
    ])
    train_df = pd.DataFrame({
        "days_of_supply": rng.normal(20, 5, size=10000),
        "sales_7d": rng.exponential(500, size=10000),
        "avg_daily_sales_7d": rng.exponential(70, size=10000),
        "supplier_reliability": clustered,
    })
    train_path = tmp_path / "train.parquet"
    train_df.to_parquet(train_path)

    monitor = _monitor()
    monitor._dataset_dir = tmp_path
    quantiles = monitor._load_training_quantiles()

    edges, expected = quantiles["supplier_reliability"]
    assert len(edges) < dm._N_BINS + 1  # duplicate edges were deduped
    assert pytest.approx(expected.sum(), abs=1e-9) == 1.0

    psi = monitor._compute_psi(
        training_mean=float(clustered.mean()),
        training_std=float(clustered.std()),
        current_values=clustered,
        bin_edges=edges,
        expected=expected,
    )
    assert psi < 0.01
