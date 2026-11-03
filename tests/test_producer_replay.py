"""
kafka/producer/simulate.py: the producer sends every world-model event to its
topic, keyed by pair, with the simulated timestamp in the JSON payload.

confluent_kafka isn't needed to test this, so it is stubbed before the module
is loaded by file path.
"""
import importlib.util
import json
import sys
import types
from datetime import date
from pathlib import Path

import pytest

from simulation.world import World

_SIMULATE_PATH = Path(__file__).parents[1] / "kafka" / "producer" / "simulate.py"


def _load_simulate():
    if "confluent_kafka" not in sys.modules:
        fake = types.ModuleType("confluent_kafka")
        fake.Producer = object
        sys.modules["confluent_kafka"] = fake
    spec = importlib.util.spec_from_file_location("scip_simulate_under_test", _SIMULATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeProducer:
    def __init__(self, full_every: int = 0, reject_every: int = 0, undelivered: int = 0):
        self.sent, self.flushes, self._full_every, self._calls = [], 0, full_every, 0
        self._reject_every, self._undelivered = reject_every, undelivered

    def produce(self, topic, key, value, callback):
        self._calls += 1
        if self._full_every and self._calls % self._full_every == 0:
            raise BufferError("queue full")          # the producer must retry
        self.sent.append((topic, key, json.loads(value)))
        if self._reject_every and self._calls % self._reject_every == 0:
            callback("broker rejected", types.SimpleNamespace(topic=lambda: topic))

    def poll(self, timeout):
        return 0

    def flush(self):
        self.flushes += 1
        return self._undelivered


def test_replay_sends_every_event_in_order_with_simulated_timestamps():
    simulate = _load_simulate()
    expected = []
    reference = World(date(2025, 10, 1), seed=3, warehouse_ids=["W-001"], sku_ids=["SKU-001"])
    for _ in range(40):
        expected += reference.step()

    producer = _FakeProducer(full_every=7)
    world = World(date(2025, 10, 1), seed=3, warehouse_ids=["W-001"], sku_ids=["SKU-001"])
    total = simulate.replay(producer, world, days=40)

    assert total == len(expected) == len(producer.sent)
    assert [(t, k, v) for t, k, v in producer.sent] == [(e.topic, e.key, e.payload) for e in expected]
    assert producer.sent[0][2]["timestamp"].startswith("2025-10-01")
    assert producer.sent[-1][2]["timestamp"].startswith("2025-11-09")
    assert producer.flushes >= 2   # every 30 days and at the end


def _world():
    return World(date(2025, 10, 1), seed=3, warehouse_ids=["W-001"], sku_ids=["SKU-001"])


def test_replay_raises_when_kafka_rejects_an_event():
    simulate = _load_simulate()
    with pytest.raises(RuntimeError, match="Kafka delivery failed"):
        simulate.replay(_FakeProducer(reject_every=5), _world(), days=30)


def test_replay_raises_when_events_stay_queued():
    simulate = _load_simulate()
    with pytest.raises(RuntimeError, match="2 still queued"):
        simulate.replay(_FakeProducer(undelivered=2), _world(), days=30)
