"""
Regression tests for ml_training/datasets/label_generator.py.

Covers the 7-day stockout window boundary bug fixed in this branch: the
lookahead window was (sample_date+1d, sample_date+7d] in literal timestamp
terms, where sample_date+7d is exactly midnight of day+7 -- so a stockout
event any time later on day+7 was silently excluded from the label.
"""
import pandas as pd

from ml_training.datasets.label_generator import generate_labels

SAMPLE_DATE = pd.Timestamp("2026-01-01").date()


def _feature_row(current_qty=100.0, avg_daily_sales_7d=10.0, stddev_daily_7d=2.0):
    return pd.DataFrame([{
        "warehouse_id": "W-001",
        "sku_id": "SKU-001",
        "sample_date": SAMPLE_DATE,
        "current_qty": current_qty,
        "avg_daily_sales_7d": avg_daily_sales_7d,
        "stddev_daily_7d": stddev_daily_7d,
    }])


def _inv_event(ts: pd.Timestamp, quantity_after: float = 0):
    return pd.DataFrame([{
        "warehouse_id": "W-001",
        "sku_id": "SKU-001",
        "quantity_after": quantity_after,
        "ts": ts,
    }])


def test_stockout_late_on_day7_is_labeled_positive():
    """A stockout at day+7 18:00 must count -- this is exactly the boundary
    bug: the old window ended at day+7 00:00, excluding this event."""
    inv_events = _inv_event(pd.Timestamp("2026-01-08 18:00", tz="UTC"))
    result = generate_labels(_feature_row(), inv_events, horizon_days=7)
    assert result["stockout_label_7d"].iloc[0] == 1


def test_stockout_exactly_at_day7_midnight_is_labeled_positive():
    """The instant previously included by the old (buggy) window must still
    count under the fixed window."""
    inv_events = _inv_event(pd.Timestamp("2026-01-08 00:00", tz="UTC"))
    result = generate_labels(_feature_row(), inv_events, horizon_days=7)
    assert result["stockout_label_7d"].iloc[0] == 1


def test_stockout_on_day8_is_not_labeled():
    """An event the day after the 7-day horizon must NOT count."""
    inv_events = _inv_event(pd.Timestamp("2026-01-09 01:00", tz="UTC"))
    result = generate_labels(_feature_row(), inv_events, horizon_days=7)
    assert result["stockout_label_7d"].iloc[0] == 0


def test_stockout_on_sample_date_itself_is_not_labeled():
    """An event on the sample date itself is leakage, not a future
    stockout, and must not count."""
    inv_events = _inv_event(pd.Timestamp("2026-01-01 23:00", tz="UTC"))
    result = generate_labels(_feature_row(), inv_events, horizon_days=7)
    assert result["stockout_label_7d"].iloc[0] == 0


def test_no_future_events_is_not_labeled():
    inv_events = pd.DataFrame(columns=["warehouse_id", "sku_id", "quantity_after", "ts"])
    result = generate_labels(_feature_row(), inv_events, horizon_days=7)
    assert result["stockout_label_7d"].iloc[0] == 0
