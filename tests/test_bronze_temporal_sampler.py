"""
Regression tests for ml_training/datasets/bronze_temporal_sampler.py's
_velocity_features, covering the velocity_trend cold-start artifact fixed
in this branch: avg_daily_sales_{7,30}d divided by hardcoded 7.0/30.0
regardless of how much real history existed, so a pair with only a few
days of sales got velocity_trend == 30/7 (~4.29) as a pure divisor
artifact instead of a real trend.

The velocity tests build an instance via __new__ and call
_velocity_features directly -- it only touches its DataFrame arguments.
The local-directory tests below construct the sampler normally with
bronze_path, which needs no GCS client.
"""
import pandas as pd
import pytest

from ml_training.datasets import bronze_temporal_sampler as bts
from ml_training.datasets.bronze_temporal_sampler import BronzeTemporalSampler


def _sampler() -> BronzeTemporalSampler:
    return BronzeTemporalSampler.__new__(BronzeTemporalSampler)


def _sales_events(base: pd.Timestamp, daily_units: list[float]) -> pd.DataFrame:
    """One SALE event per day, at noon, with the given quantity_delta magnitude."""
    return pd.DataFrame([
        {
            "warehouse_id": "W-001",
            "sku_id": "SKU-001",
            "event_type": "SALE",
            "quantity_delta": -units,
            "ts": base + pd.Timedelta(days=day_offset, hours=12),
        }
        for day_offset, units in enumerate(daily_units)
    ])


def test_cold_start_velocity_trend_is_flat_not_an_artifact():
    """With only 3 days of real history, sales_7d == sales_30d. Dividing
    both by the actual elapsed days (not hardcoded 7.0/30.0) must give
    velocity_trend == 1.0, not the 30/7 ~= 4.29 divisor artifact."""
    base = pd.Timestamp("2026-01-01", tz="UTC")
    inv = _sales_events(base, [10.0, 10.0, 10.0])  # 3 days, flat demand

    sample_dates = pd.DatetimeIndex([base + pd.Timedelta(days=2)])
    result = _sampler()._velocity_features(inv, sample_dates)

    row = result.iloc[0]
    assert row["sales_7d"] == row["sales_30d"] == 30.0
    assert row["avg_daily_sales_7d"] == row["avg_daily_sales_30d"] == 10.0
    assert row["velocity_trend"] == 1.0


def test_mature_history_denominators_saturate_at_window_length():
    """Once 30+ days of history exist, the actual-elapsed-days divisor
    must saturate at 30 (and 7), reproducing the original fixed-divisor
    behavior exactly."""
    base = pd.Timestamp("2026-01-01", tz="UTC")
    # 40 days: 10 units/day for the first 33 days, 20 units/day for the last 7.
    daily_units = [10.0] * 33 + [20.0] * 7
    inv = _sales_events(base, daily_units)

    sample_dates = pd.DatetimeIndex([base + pd.Timedelta(days=39)])
    result = _sampler()._velocity_features(inv, sample_dates)

    row = result.iloc[0]
    assert row["sales_7d"] == 140.0          # 7 days * 20
    assert row["avg_daily_sales_7d"] == 20.0  # 140 / 7, not 140 / (fewer days)
    assert row["sales_30d"] == sum(daily_units[-30:])
    assert row["avg_daily_sales_30d"] == row["sales_30d"] / 30.0


def test_real_trend_is_still_detected_once_mature():
    """The fix must not flatten real trends once there's enough history --
    only the cold-start artifact should disappear."""
    base = pd.Timestamp("2026-01-01", tz="UTC")
    daily_units = [10.0] * 33 + [20.0] * 7  # demand doubled in the last week
    inv = _sales_events(base, daily_units)

    sample_dates = pd.DatetimeIndex([base + pd.Timedelta(days=39)])
    result = _sampler()._velocity_features(inv, sample_dates)

    row = result.iloc[0]
    assert row["velocity_trend"] > 1.0  # genuine upward trend, correctly detected


# ── Local Bronze directory (--bronze-path) ────────────────────────────────────

def _inventory_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "warehouse_id":   ["W-001", "W-001", "W-002"],
        "sku_id":         ["SKU-001", "SKU-001", "SKU-002"],
        "event_type":     ["SALE", "RESTOCK", "SALE"],
        "quantity_after": [90.0, 140.0, 10.0],
        "quantity_delta": [-10.0, 50.0, -5.0],
        "reorder_point":  [50.0, 50.0, 20.0],
        "safety_stock":   [20.0, 20.0, 10.0],
        "timestamp":      ["2026-01-01T10:00:00+00:00"] * 3,
        "year": [2026] * 3, "month": [1] * 3, "day": [1] * 3, "hour": [10] * 3,
    })


def test_local_bronze_reads_hive_partitions_and_skips_uncommitted_output(tmp_path):
    """Lay the frame out the way bronze_ingestion.py does (hive partitions
    event_type=/year=/month=/day=/hour= under <root>/<topic>/), plus a file
    under _temporary/ the way Spark leaves in-flight task output. The
    partition columns must come back (event_type drives the SALE filter) and
    the uncommitted file must not."""
    df = _inventory_frame()
    df.to_parquet(tmp_path / "inventory.events",
                  partition_cols=["event_type", "year", "month", "day", "hour"])
    junk = tmp_path / "inventory.events" / "_temporary" / "0"
    junk.mkdir(parents=True)
    df.head(1).to_parquet(junk / "part-0.parquet")

    out = BronzeTemporalSampler(bronze_path=tmp_path)._read_topic(
        "inventory.events", bts._INVENTORY_COLS
    )

    assert len(out) == 3
    assert set(bts._INVENTORY_COLS) <= set(out.columns)
    assert sorted(out["event_type"].astype(str)) == ["RESTOCK", "SALE", "SALE"]
    assert out["quantity_after"].sum() == 240.0


def test_local_bronze_missing_topic_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="demand.signals"):
        BronzeTemporalSampler(bronze_path=tmp_path)._read_topic("demand.signals", bts._DEMAND_COLS)


def test_sampler_takes_exactly_one_bronze_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        BronzeTemporalSampler()
    with pytest.raises(ValueError, match="exactly one"):
        BronzeTemporalSampler(bucket="some-bucket", bronze_path=tmp_path)
