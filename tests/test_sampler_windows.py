"""
BronzeTemporalSampler's feature windows, checked against values computed
directly from the replayed world's events (tests/world_fixture.py).

These are the definitions Silver uses too (spark/jobs/silver_features.py);
tests/test_spark_parity.py checks that the two implementations agree.
"""
from datetime import timedelta

import pytest

from ml_training.datasets.bronze_temporal_sampler import BronzeTemporalSampler
from world_fixture import (
    DAYS,
    SKUS,
    START,
    WAREHOUSES,
    event_date,
    events_by_topic,
    world_events,
    write_bronze_pandas,
)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("bronze")
    events = world_events()
    write_bronze_pandas(root, events)
    features, _ = BronzeTemporalSampler(bronze_path=root).build()
    return events_by_topic(events), features


def _row(features, key, d):
    wh, sku = key.split("#")
    rows = features[(features.warehouse_id == wh) & (features.sku_id == sku) & (features.sample_date == d)]
    assert len(rows) == 1
    return rows.iloc[0]


def _cases(features):
    for r in features.itertuples(index=False):
        yield f"{r.warehouse_id}#{r.sku_id}", r.sample_date


def test_one_row_per_pair_per_usable_date(fixture):
    _, features = fixture
    first, last = START + timedelta(days=30), START + timedelta(days=DAYS - 1 - 7)
    assert sorted(features["sample_date"].unique()) == [
        first + timedelta(days=i) for i in range((last - first).days + 1)]
    assert len(features) == len(WAREHOUSES) * len(SKUS) * ((last - first).days + 1)


def test_sales_windows_are_7_and_30_days_ending_on_the_date(fixture):
    topics, features = fixture
    sales = [p for p in topics["inventory.events"] if p["event_type"] == "SALE"]
    for key, d in _cases(features):
        units = {}
        for p in sales:
            if f"{p['warehouse_id']}#{p['sku_id']}" == key:
                units[event_date(p)] = -p["quantity_delta"]
        row = _row(features, key, d)
        assert row.sales_7d == sum(units.get(d - timedelta(days=i), 0) for i in range(7))
        assert row.sales_30d == sum(units.get(d - timedelta(days=i), 0) for i in range(30))


def test_demand_signals_count_the_date_only(fixture):
    topics, features = fixture
    for key, d in _cases(features):
        expected = sum(1 for p in topics["demand.signals"]
                       if f"{p['warehouse_id']}#{p['sku_id']}" == key and event_date(p) == d)
        assert _row(features, key, d).demand_signal_count_24h == expected


def _open_quantity(opened, closed_ids_by_date, key, d, window_days, id_field, qty_field):
    total = 0
    for p in opened:
        if f"{p['warehouse_id']}#{p['sku_id']}" != key:
            continue
        if not (d - timedelta(days=window_days) <= event_date(p) <= d):
            continue
        closed = closed_ids_by_date.get(p[id_field])
        if closed is None or closed > d:
            total += p[qty_field]
    return total


def test_open_orders_and_shipments_net_out_closed_ones(fixture):
    topics, features = fixture
    po, shp = topics["purchase.orders.events"], topics["shipments.events"]
    created   = [p for p in po if p["event_type"] == "CREATED"]
    received  = {p["order_id"]: event_date(p) for p in po if p["event_type"] == "RECEIVED"}
    shipped   = [p for p in shp if p["event_type"] == "SHIPPED"]
    delivered = {p["shipment_id"]: event_date(p) for p in shp if p["event_type"] == "DELIVERED"}

    saw_open_order = saw_in_transit = False
    for key, d in _cases(features):
        row = _row(features, key, d)
        assert row.active_po_qty == _open_quantity(created, received, key, d, 90, "order_id", "quantity_ordered")
        assert row.in_transit_qty == _open_quantity(shipped, delivered, key, d, 30, "shipment_id", "quantity")
        saw_open_order |= row.active_po_qty > 0
        saw_in_transit |= row.in_transit_qty > 0
    # The fixture must exercise both open and closed states to mean anything.
    assert saw_open_order and saw_in_transit
    assert any(row.active_po_qty == 0 for _, row in features.iterrows())
