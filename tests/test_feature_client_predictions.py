"""
Tests for FeatureClient's prediction-row parsing (pure; no Bigtable).

read_predictions() is the one reader of the predictions family (UTF-8 strings
written by prediction_writer.py) and risk:urgency (a big-endian double written
by Silver), shared by the drift monitor's variance-collapse and
baseline-divergence checks.
"""
import struct
from types import SimpleNamespace

from model_serving.pipeline.feature_client import PredictionRow, _parse_prediction_row


def _cells(pred: dict[str, str] | None = None, urgency: float | None = None) -> dict:
    cells: dict = {}
    if pred:
        cells["predictions"] = {
            k.encode(): [SimpleNamespace(value=v.encode())] for k, v in pred.items()
        }
    if urgency is not None:
        cells["risk"] = {b"urgency": [SimpleNamespace(value=struct.pack(">d", urgency))]}
    return cells


def test_parses_prediction_strings_and_urgency_double():
    row = _parse_prediction_row(
        "W-001#SKU-001", "W-001", "SKU-001",
        _cells({"shortage_prob_7d": "0.250000", "stockout_risk_score": "0.700000",
                "model_version": "v20260621_222846", "prediction_timestamp": "2026-10-30T00:00:00Z"},
               urgency=2.0),
    )
    assert row == PredictionRow(
        "W-001#SKU-001", "W-001", "SKU-001",
        0.25, 0.7, "v20260621_222846", "2026-10-30T00:00:00Z", 2.0,
    )


def test_urgency_only_row_is_kept_for_the_baseline_check():
    row = _parse_prediction_row("k", "W-001", "SKU-002", _cells(urgency=3.0))
    assert row is not None
    assert row.shortage_prob_7d is None and row.reorder_urgency == 3.0


def test_row_with_no_prediction_or_urgency_is_dropped():
    assert _parse_prediction_row("k", "W-001", "SKU-003", _cells()) is None
    assert _parse_prediction_row("k", "W-001", "SKU-003", _cells({"model_version": "v1"})) is None


def test_unparseable_probability_becomes_none_but_row_survives():
    row = _parse_prediction_row("k", "W-001", "SKU-004", _cells({"shortage_prob_7d": "n/a"}, urgency=1.0))
    assert row is not None
    assert row.shortage_prob_7d is None and row.reorder_urgency == 1.0
