"""
Silver's feature windows (spark/jobs/silver_features.py), checked against
values computed directly from the replayed world's events
(tests/world_fixture.py).

The fixture world is written to disk in Bronze's layout by
bronze_ingestion.prepare_bronze_df, twice, the way a repeated replay lands,
so every check also covers dropping the duplicates. The training set is built
from these same Silver rows (spark/jobs/training_set.py).

Needs PySpark and Java 17. Marked slow: CI runs it in its own job
(.github/workflows/ci.yml); `pytest -m slow tests/test_silver_features.py`
runs it locally.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytestmark = pytest.mark.slow

from pyspark.sql import SparkSession  # noqa: E402

from feature_registry import FEATURE_REGISTRY, RAW_FEATURE_COLS  # noqa: E402
from jobs.bronze_ingestion import TOPIC_SCHEMA_MAP, prepare_bronze_df  # noqa: E402
from jobs.silver_features import _risk, compute_features  # noqa: E402
from ml_training.datasets.feature_engineer import engineer_features  # noqa: E402
from world_fixture import (  # noqa: E402
    BRONZE_PARTITIONS,
    DAYS,
    SKUS,
    START,
    WAREHOUSES,
    event_date,
    events_by_topic,
    world_events,
)

FIRST, LAST = START + timedelta(days=30), START + timedelta(days=DAYS - 1 - 7)
KEY = ["sample_date", "warehouse_id", "sku_id"]

# Silver column (family__attribute) -> the raw feature name the training code and scorer use.
_SILVER_TO_RAW = {f"{family}__{attr}": attr for family, attrs in FEATURE_REGISTRY.items() for attr in attrs}


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]").appName("scip-silver-features-test")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def _write_bronze(spark, root, events, copies: int = 1) -> None:
    for topic, payloads in events_by_topic(events).items():
        df = prepare_bronze_df(spark.createDataFrame(payloads, schema=TOPIC_SCHEMA_MAP[topic]), topic)
        for _ in range(copies):
            df.write.mode("append").partitionBy(*BRONZE_PARTITIONS).parquet(str(root / topic))


def _features(spark, bronze, first, last=None) -> pd.DataFrame:
    rows = compute_features(spark, str(bronze), first, last).collect()
    return (pd.DataFrame([r.asDict() for r in rows])
            .rename(columns={**_SILVER_TO_RAW, "as_of": "sample_date"})
            .set_index(KEY).sort_index())


@pytest.fixture(scope="module")
def bronze(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("bronze")
    _write_bronze(spark, root, world_events(), copies=2)   # a repeated replay
    return root


@pytest.fixture(scope="module")
def topics():
    return events_by_topic(world_events())


@pytest.fixture(scope="module")
def features(spark, bronze):
    """Silver's features for every sample date, computed in one pass."""
    return _features(spark, bronze, FIRST, LAST)


def _pair(p: dict) -> tuple[str, str]:
    return p["warehouse_id"], p["sku_id"]


def _cases(features):
    for d, wh, sku in features.index:
        yield d, (wh, sku)


def _value(features, d, pair, column) -> float:
    """A Silver value, with null (nothing in the window) read as 0."""
    v = features.loc[(d, *pair), column]
    return 0.0 if pd.isna(v) else float(v)


def test_one_row_per_pair_per_date(features):
    dates = [FIRST + timedelta(days=i) for i in range((LAST - FIRST).days + 1)]
    assert sorted(set(features.index.get_level_values("sample_date"))) == dates
    assert len(features) == len(WAREHOUSES) * len(SKUS) * len(dates)


def test_a_date_computed_alone_equals_its_rows_in_the_range(spark, bronze, features):
    one_date = FIRST + timedelta(days=4)
    cols = list(RAW_FEATURE_COLS)
    pd.testing.assert_frame_equal(_features(spark, bronze, one_date)[cols], features.loc[[one_date]][cols])


def test_current_qty_is_the_latest_inventory_event(topics, features):
    for d, pair in _cases(features):
        events = [p for p in topics["inventory.events"] if _pair(p) == pair and event_date(p) <= d]
        latest = max(events, key=lambda p: p["timestamp"])
        assert features.loc[(d, *pair), "current_qty"] == latest["quantity_after"]


def test_sales_windows_are_7_and_30_days_ending_on_the_date(topics, features):
    sales = [p for p in topics["inventory.events"] if p["event_type"] == "SALE"]
    for d, pair in _cases(features):
        units = {event_date(p): -p["quantity_delta"] for p in sales if _pair(p) == pair}
        week = [units.get(d - timedelta(days=i), 0) for i in range(7)]
        month = [units.get(d - timedelta(days=i), 0) for i in range(30)]
        row = features.loc[(d, *pair)]
        assert row.sales_7d == sum(week)
        assert row.sales_30d == sum(month)
        # Every pair has more than 30 days of history on the sample dates.
        np.testing.assert_allclose(row.avg_daily_sales_7d, sum(week) / 7)
        np.testing.assert_allclose(row.avg_daily_sales_30d, sum(month) / 30)
        np.testing.assert_allclose(row.stddev_daily_7d, np.std(week))
        np.testing.assert_allclose(row.velocity_trend, (sum(week) / 7) / (sum(month) / 30))


def test_demand_signals_count_the_date_only(topics, features):
    for d, pair in _cases(features):
        expected = sum(1 for p in topics["demand.signals"] if _pair(p) == pair and event_date(p) == d)
        assert _value(features, d, pair, "demand_signal_count_24h") == expected


def test_supplier_features_average_the_warehouse_over_90_days(topics, features):
    for d, (wh, sku) in _cases(features):
        window = [p for p in topics["suppliers.events"]
                  if p["warehouse_id"] == wh and d - timedelta(days=90) <= event_date(p) <= d]
        row = features.loc[(d, wh, sku)]
        np.testing.assert_allclose(row.lead_time_days, np.mean([p["lead_time_days"] for p in window]))
        np.testing.assert_allclose(row.lead_time_stddev, np.std([p["lead_time_days"] for p in window]))
        np.testing.assert_allclose(row.supplier_reliability, np.mean([p["on_time_delivery_rate"] for p in window]))


def _open_quantity(opened, closed_date_by_id, pair, d, window_days, id_field, qty_field):
    total = 0
    for p in opened:
        if _pair(p) != pair or not (d - timedelta(days=window_days) <= event_date(p) <= d):
            continue
        closed = closed_date_by_id.get(p[id_field])
        if closed is None or closed > d:
            total += p[qty_field]
    return total


def test_open_orders_and_shipments_net_out_closed_ones(topics, features):
    po, shp = topics["purchase.orders.events"], topics["shipments.events"]
    created   = [p for p in po if p["event_type"] == "CREATED"]
    received  = {p["order_id"]: event_date(p) for p in po if p["event_type"] == "RECEIVED"}
    shipped   = [p for p in shp if p["event_type"] == "SHIPPED"]
    delivered = {p["shipment_id"]: event_date(p) for p in shp if p["event_type"] == "DELIVERED"}

    open_orders = in_transit = closed = 0
    for d, pair in _cases(features):
        active_po = _value(features, d, pair, "active_po_qty")
        transit   = _value(features, d, pair, "in_transit_qty")
        assert active_po == _open_quantity(created, received, pair, d, 90, "order_id", "quantity_ordered")
        assert transit == _open_quantity(shipped, delivered, pair, d, 30, "shipment_id", "quantity")
        open_orders += active_po > 0
        in_transit  += transit > 0
        closed      += active_po == 0
    # The fixture must exercise both open and closed states to mean anything.
    assert open_orders and in_transit and closed


def _label_window_receipt(topics, dates):
    """A (pair, D) where an order was open at the end of D and received during D's 7-day label window."""
    po = topics["purchase.orders.events"]
    received = {p["order_id"]: event_date(p) for p in po if p["event_type"] == "RECEIVED"}
    for p in po:
        if p["event_type"] != "CREATED" or p["order_id"] not in received:
            continue
        for d in dates:
            if event_date(p) <= d < received[p["order_id"]] <= d + timedelta(days=7):
                return _pair(p), d, p["quantity_ordered"]
    return None


def test_features_on_a_date_use_no_event_after_it(spark, topics, features, tmp_path):
    """
    Leakage guard. Compute the features from only the events up to the end
    of D: every feature on D must be identical to the full-history value.
    D is chosen so an order placed before D is received inside D's label
    window; on D that order must still count as open.
    """
    found = _label_window_receipt(topics, sorted(set(features.index.get_level_values("sample_date"))))
    assert found, "fixture has no order received inside a label window"
    pair, d, qty = found

    _write_bronze(spark, tmp_path, [e for e in world_events() if event_date(e.payload) <= d])
    cols = list(RAW_FEATURE_COLS)
    pd.testing.assert_frame_equal(_features(spark, tmp_path, d)[cols], features.loc[[d]][cols])
    assert _value(features, d, pair, "active_po_qty") >= qty


# ── Velocity during a pair's first days ───────────────────────────────────────

def _sales_only(units_by_day: list[int]) -> list[dict]:
    """One SALE at noon per day from START for one pair, with the given units sold."""
    stock, payloads = 10_000, []
    for i, units in enumerate(units_by_day):
        day = START + timedelta(days=i)
        payloads.append({
            "event_id": f"sale-{i}", "schema_version": "1.0", "event_type": "SALE",
            "warehouse_id": "W-001", "sku_id": "SKU-001",
            "quantity_before": stock, "quantity_after": stock - units, "quantity_delta": -units,
            "unit_cost": None, "reorder_point": None, "safety_stock": None, "reference_id": None,
            "timestamp": f"{day.isoformat()}T12:00:00+00:00",
            "producer_id": None, "correlation_id": None, "metadata": None,
        })
        stock -= units
    return payloads


def _velocity(spark, tmp_path, units_by_day: list[int]) -> pd.Series:
    """
    Silver's row for W-001/SKU-001 on the last of these days. Its inventory
    events are these sales only; the other topics are the fixture world's,
    which holds no events for this pair.
    """
    _write_bronze(spark, tmp_path, [e for e in world_events() if e.topic != "inventory.events"])
    sales = prepare_bronze_df(
        spark.createDataFrame(_sales_only(units_by_day), schema=TOPIC_SCHEMA_MAP["inventory.events"]),
        "inventory.events")
    sales.write.partitionBy(*BRONZE_PARTITIONS).parquet(str(tmp_path / "inventory.events"))
    rows = _features(spark, tmp_path, START + timedelta(days=len(units_by_day) - 1))
    assert len(rows) == 1
    return rows.iloc[0]


def test_cold_start_velocity_trend_is_flat_not_an_artifact(spark, tmp_path):
    """With 3 days of history sales_7d == sales_30d; fixed 7/30 divisors would give a trend of 30/7."""
    row = _velocity(spark, tmp_path, [10, 10, 10])
    assert row.sales_7d == row.sales_30d == 30
    assert row.avg_daily_sales_7d == row.avg_daily_sales_30d == 10.0
    assert row.velocity_trend == 1.0


def test_mature_history_divides_by_the_window_length(spark, tmp_path):
    row = _velocity(spark, tmp_path, [10] * 40)
    assert row.avg_daily_sales_7d == 70 / 7
    assert row.avg_daily_sales_30d == 300 / 30
    assert row.velocity_trend == 1.0


def test_a_real_trend_is_detected_once_mature(spark, tmp_path):
    row = _velocity(spark, tmp_path, [10] * 33 + [20] * 7)
    np.testing.assert_allclose(row.velocity_trend, 20.0 / ((23 * 10 + 7 * 20) / 30))


# ── days_of_supply ────────────────────────────────────────────────────────────

def test_silver_days_of_supply_is_the_models_formula(spark):
    # current_qty, avg_daily_sales_7d: in range; out of stock with no sales;
    # in stock with no sales; no sales recorded at all; over the cap.
    cases = [(50.0, 10.0), (0.0, 0.0), (50.0, 0.0), (50.0, None), (5000.0, 1.0)]
    silver = _risk(spark.createDataFrame(
        [(i, qty, avg) for i, (qty, avg) in enumerate(cases)],
        "i int, inventory__current_qty double, velocity__avg_daily_sales_7d double",
    )).orderBy("i").collect()

    raw = pd.DataFrame(cases, columns=["current_qty", "avg_daily_sales_7d"])
    model = engineer_features(raw.reindex(columns=list(RAW_FEATURE_COLS)))["days_of_supply"].tolist()

    assert [r["inventory__days_of_supply"] for r in silver] == model == [5.0, 0.0, 365.0, 365.0, 365.0]
    assert [r["risk__reorder_urgency"] for r in silver] == [2, 3, 0, 0, 0]
