"""
A small replayed world (4 pairs x 45 days), shared by the Spark tests of
Silver's features and the training set.

Two warehouses x two SKUs, so the per-warehouse supplier aggregation covers
more than one SKU. SKU-002 (4-day lead time) and SKU-016 (10-day lead time)
both reorder within the 45 days, so orders and shipments are open on some
sample dates.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime

from simulation.world import Event, World

START = date(2025, 10, 1)
DAYS = 45
WAREHOUSES = ["W-001", "W-006"]
SKUS = ["SKU-002", "SKU-016"]
BRONZE_PARTITIONS = ["event_type", "year", "month", "day"]


def world_events(days: int = DAYS) -> list[Event]:
    world = World(START, seed=42, warehouse_ids=WAREHOUSES, sku_ids=SKUS)
    events: list[Event] = []
    for _ in range(days):
        events += world.step()
    return events


def events_by_topic(events: list[Event]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        out[e.topic].append(e.payload)
    return out


def event_date(payload: dict) -> date:
    return datetime.fromisoformat(payload["timestamp"]).date()
