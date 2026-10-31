"""
simulation/world.py: the day-level world model the producer replays.

Covers reproducibility, stock accounting, the order lifecycle the Silver and
sampler netting relies on, and that every event carries the fields Bronze
requires for its topic.
"""
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pytest

from simulation.world import World

sys.path.insert(0, str(Path(__file__).parents[1] / "spark"))
from config.bronze_config import TOPIC_REQUIRED_FIELDS  # noqa: E402

START = date(2025, 10, 1)


def _run(world: World, days: int) -> list:
    events = []
    for _ in range(days):
        events += world.step()
    return events


@pytest.fixture(scope="module")
def year_events():
    return _run(World(START, seed=42), 365)


def test_same_seed_same_events_and_subset_matches_full_world():
    full   = _run(World(START, seed=7), 60)
    again  = _run(World(START, seed=7), 60)
    subset = _run(World(START, seed=7, warehouse_ids=["W-002"], sku_ids=["SKU-003", "SKU-012"]), 60)

    assert [e.payload for e in full] == [e.payload for e in again]
    keys = {"W-002#SKU-003", "W-002#SKU-012"}
    assert [e.payload for e in subset] == [e.payload for e in full if e.key in keys or
                                           (e.topic == "suppliers.events" and
                                            e.payload["warehouse_id"] == "W-002" and
                                            e.payload["affected_sku_ids"][0] in {"SKU-003", "SKU-012"})]
    assert [e.payload for e in _run(World(START, seed=8), 60)] != [e.payload for e in full]


def test_every_event_has_its_topic_s_required_fields(year_events):
    for e in year_events:
        missing = [f for f in TOPIC_REQUIRED_FIELDS[e.topic] if e.payload.get(f) is None]
        assert not missing, (e.topic, missing)
    assert len({e.payload["event_id"] for e in year_events}) == len(year_events)


def test_stock_accounting_and_one_sales_posting_per_pair_per_day(year_events):
    sales_per_pair_day = Counter()
    for e in year_events:
        if e.topic != "inventory.events":
            continue
        p = e.payload
        assert p["quantity_after"] == p["quantity_before"] + p["quantity_delta"]
        assert p["quantity_after"] >= 0
        if p["event_type"] == "SALE":
            assert p["quantity_delta"] <= 0
            sales_per_pair_day[(e.key, p["timestamp"][:10])] += 1
    assert len(sales_per_pair_day) == 120 * 365
    assert set(sales_per_pair_day.values()) == {1}


def test_order_lifecycle_is_consistent(year_events):
    by_order = defaultdict(dict)
    for e in year_events:
        p = e.payload
        if e.topic == "purchase.orders.events":
            by_order[p["order_id"]][p["event_type"]] = p
        elif e.topic == "shipments.events":
            by_order[p["shipment_id"].replace("SHP-", "PO-")][p["event_type"]] = p
        elif e.topic == "suppliers.events":
            by_order[p["correlation_id"]]["PERFORMANCE_UPDATE"] = p
        elif e.topic == "inventory.events" and p["event_type"] == "RECEIVE":
            by_order[p["reference_id"]]["RECEIVE"] = p

    completed = [o for o in by_order.values() if "RECEIVED" in o]
    assert len(completed) > 100
    for o in by_order.values():
        assert "CREATED" in o
        if "RECEIVED" not in o:
            continue
        created, received = o["CREATED"], o["RECEIVED"]
        assert o["SHIPPED"]["timestamp"] <= o["DELIVERED"]["timestamp"] == received["timestamp"]
        assert created["timestamp"] < o["SHIPPED"]["timestamp"]
        assert o["RECEIVE"]["quantity_delta"] == received["quantity_received"] == created["quantity_ordered"]
        perf = o["PERFORMANCE_UPDATE"]
        lead = (date.fromisoformat(received["timestamp"][:10]) - date.fromisoformat(created["timestamp"][:10])).days
        assert perf["lead_time_days"] == lead
        assert perf["on_time_delivery_rate"] in (0.0, 1.0)
        on_time = received["actual_delivery_date"] <= created["expected_delivery_date"]
        assert perf["on_time_delivery_rate"] == (1.0 if on_time else 0.0)


def test_stockouts_happen_and_recover(year_events):
    zero_days = {(e.key, e.payload["timestamp"][:10]) for e in year_events
                 if e.topic == "inventory.events" and e.payload["event_type"] == "SALE"
                 and e.payload["quantity_after"] == 0}
    assert 0.01 < len(zero_days) / (120 * 365) < 0.5
    restocked = {e.key for e in year_events
                 if e.topic == "inventory.events" and e.payload["event_type"] == "RECEIVE"
                 and e.payload["quantity_before"] == 0}
    assert restocked
