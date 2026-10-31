"""
A small replayed world (4 pairs x 45 days) in Bronze's on-disk layout, shared
by the sampler window tests and the Spark parity test.

Two warehouses x two SKUs, so the per-warehouse supplier aggregation covers
more than one SKU. SKU-002 (4-day lead time) and SKU-016 (10-day lead time)
both reorder within the 45 days, so orders and shipments are open on some
sample dates.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd

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


def write_bronze_pandas(root: Path, events: list[Event]) -> None:
    """
    Bronze layout without Spark: <root>/<topic>/event_type=/year=/month=/day=,
    with demand.signals' signal_type copied to event_type the way
    bronze_ingestion.prepare_bronze_df does.
    """
    for topic, payloads in events_by_topic(events).items():
        df = pd.DataFrame(payloads)
        if "event_type" not in df.columns:
            df["event_type"] = df["signal_type"]
        ts = pd.to_datetime(df["timestamp"], utc=True)
        df["year"], df["month"], df["day"] = ts.dt.year, ts.dt.month, ts.dt.day
        df.to_parquet(root / topic, partition_cols=BRONZE_PARTITIONS)


def event_date(payload: dict) -> date:
    return datetime.fromisoformat(payload["timestamp"]).date()
