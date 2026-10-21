"""
Regression tests for kafka/producer/simulate.py's RECEIVE reorder-point
gating, fixed in this branch: RECEIVE events fired at a fixed 20% rate
regardless of current stock level, so restocking outpaced sales
unconditionally and inventory drifted to unbounded values, never
reaching zero (verified by simulation: mean item quantity grew from
~268 to ~17,100 over 24h with zero stockouts).

simulate.py imports confluent_kafka at module level, which isn't
installed in this environment (and isn't needed to test this pure
logic), so it's stubbed before loading the module by file path.
"""
import importlib.util
import random
import sys
import types
from pathlib import Path

import pytest

_SIMULATE_PATH = Path(__file__).parents[1] / "kafka" / "producer" / "simulate.py"


@pytest.fixture()
def simulate_module():
    if "confluent_kafka" not in sys.modules:
        fake = types.ModuleType("confluent_kafka")
        fake.Producer = object  # only referenced as a type hint at import time
        sys.modules["confluent_kafka"] = fake

    spec = importlib.util.spec_from_file_location("scip_simulate_under_test", _SIMULATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeProducer:
    """Captures the event dict instead of touching a real Kafka broker."""
    def __init__(self):
        self.last_event = None

    def produce(self, topic, key, value, callback):
        import json
        self.last_event = json.loads(value)


def test_receive_above_reorder_point_becomes_sale(simulate_module, monkeypatch):
    """Stock well above reorder_point: a drawn RECEIVE must be converted
    to a SALE instead of restocking an already-healthy item."""
    wh, sku = "W-001", "SKU-001"
    simulate_module.inventory[(wh, sku)] = 500
    simulate_module.REORDER_PARAMS[sku]["reorder_point"] = 30

    monkeypatch.setattr(random, "choice", lambda seq: wh if seq is simulate_module.WAREHOUSES else sku)
    monkeypatch.setattr(random, "choices", lambda *a, **k: ["RECEIVE"])

    producer = _FakeProducer()
    simulate_module.produce_inventory_event(producer)

    assert producer.last_event["event_type"] == "SALE"
    assert producer.last_event["quantity_after"] < producer.last_event["quantity_before"]


def test_receive_at_or_below_reorder_point_restocks(simulate_module, monkeypatch):
    """Stock at/below reorder_point: a drawn RECEIVE must restock
    normally -- the gate must not block legitimate replenishment."""
    wh, sku = "W-001", "SKU-001"
    simulate_module.inventory[(wh, sku)] = 10
    simulate_module.REORDER_PARAMS[sku]["reorder_point"] = 30

    monkeypatch.setattr(random, "choice", lambda seq: wh if seq is simulate_module.WAREHOUSES else sku)
    monkeypatch.setattr(random, "choices", lambda *a, **k: ["RECEIVE"])

    producer = _FakeProducer()
    simulate_module.produce_inventory_event(producer)

    assert producer.last_event["event_type"] == "RECEIVE"
    assert producer.last_event["quantity_after"] > producer.last_event["quantity_before"]


def test_non_receive_events_are_unaffected_by_the_gate(simulate_module, monkeypatch):
    """SALE/ADJUSTMENT/RETURN/TRANSFER draws must pass through unchanged
    regardless of stock level -- only RECEIVE is gated."""
    wh, sku = "W-001", "SKU-001"
    simulate_module.inventory[(wh, sku)] = 500
    simulate_module.REORDER_PARAMS[sku]["reorder_point"] = 30

    monkeypatch.setattr(random, "choice", lambda seq: wh if seq is simulate_module.WAREHOUSES else sku)
    monkeypatch.setattr(random, "choices", lambda *a, **k: ["RETURN"])

    producer = _FakeProducer()
    simulate_module.produce_inventory_event(producer)

    assert producer.last_event["event_type"] == "RETURN"
