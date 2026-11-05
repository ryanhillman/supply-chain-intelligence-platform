"""
The drift monitor's prediction check reads the model's probability only.

stockout_risk_score is a closed-form function of stock and sales, so its
spread reflects stock levels and retraining cannot change it. Only a
near-constant shortage_prob_7d counts as a collapse.

DriftMonitor.__init__ needs a live Bigtable client, so the monitor is built
via __new__ with a fake feature client.
"""
from types import SimpleNamespace

from model_serving.monitoring import drift_monitor as dm
from model_serving.pipeline.feature_client import PredictionRow


def _monitor(rows: list[PredictionRow]) -> dm.DriftMonitor:
    m = dm.DriftMonitor.__new__(dm.DriftMonitor)
    m._feature_client = SimpleNamespace(read_predictions=lambda: rows)
    return m


def _row(i: int, prob: float | None, score: float | None = None) -> PredictionRow:
    return PredictionRow(f"W-001#SKU-{i:03d}", "W-001", f"SKU-{i:03d}", prob, score)


def test_a_constant_risk_score_is_not_a_collapse():
    pm = _monitor([_row(1, 0.1, 0.0), _row(2, 0.9, 0.0)])._collect_prediction_values()
    assert pm.n_entities == 2
    assert not pm.prob_variance_collapsed


def test_near_constant_probabilities_are_a_collapse():
    pm = _monitor([_row(i, 0.5 + i * 1e-4) for i in range(5)])._collect_prediction_values()
    assert pm.prob_variance_collapsed


def test_rows_without_a_probability_are_ignored():
    assert _monitor([_row(1, None, 0.7)])._collect_prediction_values() is None
