"""
Tests for the scorer's feature-freshness guard (feature_client.split_fresh).

Feature families keep only the latest cell (no age-based GC), so a store
that has not been written for a while still looks complete. The guard uses
lineage:silver_ts to drop rows whose Silver write is older than the
threshold instead of scoring them on stale features.
"""
from datetime import datetime, timedelta, timezone

from model_serving.pipeline.feature_client import FeatureRow, _decode_timestamp, split_fresh

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def _row(key: str, age_hours: float | None) -> FeatureRow:
    ts = None if age_hours is None else NOW - timedelta(hours=age_hours)
    return FeatureRow(row_key=key, warehouse_id="W-001", sku_id=key, features={}, last_silver_write=ts)


def test_rows_within_threshold_are_fresh():
    fresh, n_stale = split_fresh([_row("a", 1), _row("b", 47.9)], max_age_hours=48, now=NOW)
    assert [r.row_key for r in fresh] == ["a", "b"]
    assert n_stale == 0


def test_old_and_missing_timestamps_are_stale():
    fresh, n_stale = split_fresh([_row("a", 1), _row("b", 49), _row("c", None)], max_age_hours=48, now=NOW)
    assert [r.row_key for r in fresh] == ["a"]
    assert n_stale == 2


def test_zero_or_none_threshold_disables_the_check():
    rows = [_row("a", 1000), _row("b", None)]
    assert split_fresh(rows, max_age_hours=None, now=NOW) == (rows, 0)
    assert split_fresh(rows, max_age_hours=0, now=NOW) == (rows, 0)


def test_decode_timestamp_handles_silver_format_and_naive_values():
    # Silver writes datetime.now(timezone.utc).isoformat().
    assert _decode_timestamp(b"2026-10-30T11:00:00.123456+00:00", "k") == datetime(
        2026, 10, 30, 11, 0, 0, 123456, tzinfo=timezone.utc
    )
    naive = _decode_timestamp(b"2026-10-30T11:00:00", "k")
    assert naive is not None and naive.tzinfo is timezone.utc
    assert _decode_timestamp(b"not-a-timestamp", "k") is None
