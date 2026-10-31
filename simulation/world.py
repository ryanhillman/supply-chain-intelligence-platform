"""
Day-level supply-chain world model.

A World simulates 6 warehouses x 20 SKUs one day at a time. step() advances
one day and returns that day's events for the five Kafka topics, stamped with
simulated time. The Kafka producer (kafka/producer/simulate.py) replays it,
and Bronze, Silver, the training sampler and the scorer all read the result,
so training and scoring see one world.

Per (warehouse, SKU) pair and day, in timestamp order:

  07:00  shipments.events SHIPPED          the open order ships
  08:00  shipments.events DELIVERED        the open order arrives: stock goes up,
         purchase.orders.events RECEIVED   and the supplier's actual lead time and
         inventory.events RECEIVE          on-time result (1/0) are published
         suppliers.events PERFORMANCE_UPDATE
  12:00  demand.signals INQUIRY            Poisson count around expected demand
  18:00  inventory.events SALE             the day's sales; one posting per day,
                                           0 units when stock is out
  18:30  purchase.orders.events CREATED    stock below the reorder point and no
                                           order open

Daily demand is gamma-distributed around a per-SKU mean, scaled by warehouse,
seasonality and promotional spikes. Lead times vary around a per-SKU mean;
a late order arrives a Poisson(3)-day delay after its promised date.

Event IDs are derived from (seed, pair, day, sequence), so a replay with the
same seed reproduces the same IDs and downstream jobs can drop duplicates.

numpy only: the producer image installs nothing else.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Callable, NamedTuple

import numpy as np

SCHEMA_VERSION = "1.0"
PRODUCER_ID    = "scip-world"

TRANSIT_DAYS = 3      # an order ships this many days before it arrives
SIGNAL_UNITS = 25.0   # one demand inquiry per this many units of expected demand
PROMO_RATE   = 0.03   # chance per day that a promotional spike starts (~11 a year)
LATE_DELAY_MEAN = 3.0 # mean extra days for a late order (Poisson)

_ID_NAMESPACE = uuid.UUID("6f1d3c2e-5b7a-4c1e-9a0d-3e2f1b4c5d6e")


class WarehouseProfile(NamedTuple):
    wh_id:        str
    demand_mult:  float   # demand volume relative to the baseline warehouse
    reorder_aggr: float   # 0..1, higher reorders earlier (larger reorder point)
    rel_modifier: float   # multiplies the SKU supplier's on-time rate


class SKUProfile(NamedTuple):
    sku_id:      str
    base_demand: float    # mean units/day at the baseline warehouse
    demand_cv:   float    # coefficient of variation of daily demand
    base_lt:     int      # mean supplier lead time, days
    lt_cv:       float    # coefficient of variation of lead time
    base_rel:    float    # supplier on-time rate
    season:      str      # seasonality pattern
    category:    str


WAREHOUSES: list[WarehouseProfile] = [
    WarehouseProfile("W-001", 2.0, 0.85, 1.02),  # large DC, well-managed
    WarehouseProfile("W-002", 1.0, 0.65, 1.00),  # baseline regional
    WarehouseProfile("W-003", 0.5, 0.40, 0.98),  # small depot, passive reordering
    WarehouseProfile("W-004", 1.5, 0.90, 1.03),  # high-velocity, aggressive reorder
    WarehouseProfile("W-005", 0.3, 0.50, 0.99),  # slow specialty
    WarehouseProfile("W-006", 0.8, 0.30, 0.95),  # surge overflow, lowest discipline
]

SKUS: list[SKUProfile] = [
    # FMCG: short lead times, high reliability, moderate demand variability
    SKUProfile("SKU-001", 150.0, 0.25,  5, 0.15, 0.95, "holiday", "FMCG"),
    SKUProfile("SKU-002", 300.0, 0.20,  4, 0.12, 0.97, "flat",    "FMCG"),
    SKUProfile("SKU-003",  80.0, 0.30,  6, 0.18, 0.93, "summer",  "FMCG"),
    SKUProfile("SKU-004", 200.0, 0.22,  5, 0.14, 0.96, "flat",    "FMCG"),
    SKUProfile("SKU-005", 120.0, 0.28,  7, 0.20, 0.94, "holiday", "FMCG"),
    # Electronics: medium lead times, lower reliability, higher variability
    SKUProfile("SKU-006",  25.0, 0.40, 18, 0.25, 0.79, "holiday", "Electronics"),
    SKUProfile("SKU-007",  15.0, 0.35, 21, 0.22, 0.77, "flat",    "Electronics"),
    SKUProfile("SKU-008",  40.0, 0.32, 14, 0.20, 0.82, "holiday", "Electronics"),
    SKUProfile("SKU-009",  10.0, 0.50, 21, 0.30, 0.75, "flat",    "Electronics"),
    SKUProfile("SKU-010",  30.0, 0.38, 18, 0.28, 0.80, "holiday", "Electronics"),
    # Industrial parts: long lead times, lowest reliability
    SKUProfile("SKU-011",   8.0, 0.20, 30, 0.18, 0.70, "q2",      "Industrial"),
    SKUProfile("SKU-012",   5.0, 0.25, 45, 0.22, 0.68, "flat",    "Industrial"),
    SKUProfile("SKU-013",  15.0, 0.18, 25, 0.15, 0.73, "q2",      "Industrial"),
    SKUProfile("SKU-014",   3.0, 0.30, 45, 0.25, 0.67, "flat",    "Industrial"),
    SKUProfile("SKU-015",  20.0, 0.22, 30, 0.20, 0.72, "q2",      "Industrial"),
    # Seasonal: high demand variability, medium lead times
    SKUProfile("SKU-016",  60.0, 0.60, 10, 0.28, 0.82, "summer",  "Seasonal"),
    SKUProfile("SKU-017",  90.0, 0.55, 12, 0.32, 0.84, "winter",  "Seasonal"),
    SKUProfile("SKU-018",  45.0, 0.65, 10, 0.30, 0.80, "holiday", "Seasonal"),
    SKUProfile("SKU-019",  75.0, 0.50, 12, 0.28, 0.85, "summer",  "Seasonal"),
    SKUProfile("SKU-020",  55.0, 0.70, 14, 0.35, 0.81, "winter",  "Seasonal"),
]

SKU_CATEGORIES: dict[str, str] = {s.sku_id: s.category for s in SKUS}

# One supplier per product category.
CATEGORY_SUPPLIER: dict[str, str] = {
    "FMCG": "SUP-001", "Electronics": "SUP-002", "Industrial": "SUP-003", "Seasonal": "SUP-004",
}


@dataclass(frozen=True)
class Event:
    topic:   str
    key:     str
    payload: dict


@dataclass
class _Order:
    order_id:      str
    shipment_id:   str
    qty:           int
    placed_day:    int
    ship_day:      int
    arrival_day:   int
    promised_date: date
    on_time:       bool


def seasonal_multiplier(day_of_year: int, pattern: str) -> float:
    """Demand multiplier for a day of year (1..366) and a seasonality pattern."""
    t = (day_of_year - 1) / 364.0
    if pattern == "holiday":   # sharp peak around Dec 1
        return 1.0 + 0.80 * math.exp(-0.5 * ((t - 0.92) / 0.07) ** 2)
    if pattern == "summer":    # peak around Jul 20
        return 1.0 + 0.40 * math.exp(-0.5 * ((t - 0.55) / 0.12) ** 2)
    if pattern == "winter":    # peaks at Jan 1 and late December
        return 1.0 + 0.60 * max(math.exp(-0.5 * (t / 0.06) ** 2),
                                math.exp(-0.5 * ((t - 1.0) / 0.06) ** 2))
    if pattern == "q2":        # plateau around May 15
        return 1.0 + 0.25 * math.exp(-0.5 * ((t - 0.37) / 0.12) ** 2)
    return 1.0


def _stamp(d: date, hour: int, minute: int = 0) -> str:
    return datetime.combine(d, time(hour, minute), tzinfo=timezone.utc).isoformat()


class _Pair:
    """Inventory, open order and promotion state for one (warehouse, SKU) pair."""

    def __init__(self, wh: WarehouseProfile, sku: SKUProfile, rng: np.random.Generator) -> None:
        self.wh, self.sku, self.rng = wh, sku, rng
        self.key      = f"{wh.wh_id}#{sku.sku_id}"
        self.supplier = CATEGORY_SUPPLIER[sku.category]

        self.daily_mean  = sku.base_demand * wh.demand_mult
        self.gamma_shape = max(1.0 / max(sku.demand_cv ** 2, 1e-6), 0.5)
        self.on_time_rate = float(np.clip(sku.base_rel * wh.rel_modifier, 0.50, 0.999))

        cover = self.daily_mean * sku.base_lt
        self.reorder_point = int(round(cover * (0.5 + wh.reorder_aggr * 1.5)))
        self.safety_stock  = int(round(cover * 0.5))
        self.target_qty    = int(round(cover * 3.0))

        self.qty: int = self.target_qty           # start well stocked
        self.order: _Order | None = None
        self.promo_days_left = 0
        self.promo_mult      = 1.0

    def step(self, day: int, d: date, new_id: Callable[[], str]) -> list[Event]:
        wh_id, sku_id = self.wh.wh_id, self.sku.sku_id
        base = {"schema_version": SCHEMA_VERSION, "warehouse_id": wh_id, "sku_id": sku_id,
                "producer_id": PRODUCER_ID}
        events: list[Event] = []
        o = self.order

        # 07:00 — the open order ships
        if o is not None and o.ship_day == day:
            events.append(Event("shipments.events", self.key, {
                **base, "event_id": new_id(), "event_type": "SHIPPED", "timestamp": _stamp(d, 7),
                **self._shipment_fields(o), "actual_arrival": None,
            }))

        # 08:00 — the open order arrives
        if o is not None and o.arrival_day == day:
            before = self.qty
            self.qty += o.qty
            ts = _stamp(d, 8)
            events += [
                Event("shipments.events", self.key, {
                    **base, "event_id": new_id(), "event_type": "DELIVERED", "timestamp": ts,
                    **self._shipment_fields(o), "actual_arrival": d.isoformat(),
                }),
                Event("purchase.orders.events", self.key, {
                    **base, "event_id": new_id(), "event_type": "RECEIVED", "timestamp": ts,
                    **self._order_fields(o), "quantity_received": o.qty,
                    "actual_delivery_date": d.isoformat(),
                }),
                Event("inventory.events", self.key, self._inventory_payload(
                    base, new_id(), "RECEIVE", ts, before, o.order_id)),
                Event("suppliers.events", self.supplier, {
                    "event_id": new_id(), "schema_version": SCHEMA_VERSION,
                    "event_type": "PERFORMANCE_UPDATE", "supplier_id": self.supplier,
                    "warehouse_id": wh_id,
                    "lead_time_days": float(day - o.placed_day),
                    "on_time_delivery_rate": 1.0 if o.on_time else 0.0,
                    "reliability_score": None, "affected_sku_ids": [sku_id],
                    "timestamp": ts, "producer_id": PRODUCER_ID,
                    "correlation_id": o.order_id, "metadata": None,
                }),
            ]
            self.order = None

        # Expected demand today: base x seasonality x promotion
        if self.promo_days_left == 0 and self.rng.random() < PROMO_RATE:
            self.promo_days_left = int(self.rng.integers(1, 4))      # 1-3 days
            self.promo_mult      = float(self.rng.uniform(1.8, 4.5))
        promo = self.promo_mult if self.promo_days_left > 0 else 1.0
        self.promo_days_left = max(0, self.promo_days_left - 1)
        expected = self.daily_mean * seasonal_multiplier(d.timetuple().tm_yday, self.sku.season) * promo

        # 12:00 — demand inquiries, a noisy count around expected demand
        for _ in range(int(self.rng.poisson(expected / SIGNAL_UNITS))):
            events.append(Event("demand.signals", self.key, {
                **base, "event_id": new_id(), "signal_type": "INQUIRY", "timestamp": _stamp(d, 12),
                "demand_quantity": None, "confidence_score": None,
                "signal_source": "SALES_CHANNEL", "forecast_horizon_days": None,
                "correlation_id": None, "metadata": None,
            }))

        # 18:00 — the day's sales posting (0 units when out of stock)
        demand = int(round(float(self.rng.gamma(self.gamma_shape, expected / self.gamma_shape))))
        before = self.qty
        self.qty -= min(demand, self.qty)
        events.append(Event("inventory.events", self.key, self._inventory_payload(
            base, new_id(), "SALE", _stamp(d, 18), before, None)))

        # 18:30 — reorder below the reorder point, one open order at a time
        if self.qty < self.reorder_point and self.order is None:
            self.order = self._place_order(day, d)
            events.append(Event("purchase.orders.events", self.key, {
                **base, "event_id": new_id(), "event_type": "CREATED", "timestamp": _stamp(d, 18, 30),
                **self._order_fields(self.order), "quantity_received": None,
                "actual_delivery_date": None,
            }))
        return events

    def _place_order(self, day: int, d: date) -> _Order:
        sku, rng = self.sku, self.rng
        promised = float(np.clip(rng.normal(sku.base_lt, sku.base_lt * sku.lt_cv), 1.0, sku.base_lt * 4.0))
        promised_days = max(1, math.ceil(promised))
        delay   = 0 if rng.random() < self.on_time_rate else int(rng.poisson(LATE_DELAY_MEAN))
        arrival = day + promised_days + delay
        qty = int(math.ceil(max(self.target_qty - self.qty, self.daily_mean * sku.base_lt)))
        suffix = f"{self.wh.wh_id}-{sku.sku_id}-{day:05d}"
        return _Order(
            order_id=f"PO-{suffix}", shipment_id=f"SHP-{suffix}", qty=qty,
            placed_day=day, ship_day=max(day + 1, arrival - TRANSIT_DAYS), arrival_day=arrival,
            promised_date=d + timedelta(days=promised_days), on_time=delay == 0,
        )

    def _inventory_payload(self, base: dict, event_id: str, event_type: str, ts: str,
                           before: int, reference_id: str | None) -> dict:
        return {
            **base, "event_id": event_id, "event_type": event_type, "timestamp": ts,
            "quantity_before": before, "quantity_after": self.qty,
            "quantity_delta": self.qty - before, "unit_cost": None,
            "reorder_point": self.reorder_point, "safety_stock": self.safety_stock,
            "reference_id": reference_id, "correlation_id": None, "metadata": None,
        }

    def _order_fields(self, o: _Order) -> dict:
        return {
            "order_id": o.order_id, "supplier_id": self.supplier, "quantity_ordered": o.qty,
            "unit_cost": None, "total_cost": None,
            "expected_delivery_date": o.promised_date.isoformat(), "correlation_id": o.order_id,
        }

    def _shipment_fields(self, o: _Order) -> dict:
        return {
            "shipment_id": o.shipment_id, "supplier_id": self.supplier, "quantity": o.qty,
            "carrier": None, "tracking_number": None,
            "estimated_arrival": o.promised_date.isoformat(),
            "correlation_id": o.order_id, "metadata": None,
        }


class World:
    """
    The simulated supply chain.

    Parameters
    ----------
    start:
        Date of the first simulated day.
    seed:
        Random seed. A pair's random stream depends only on the seed and the
        pair's position in WAREHOUSES / SKUS, so a World restricted to a few
        pairs produces exactly the events the full World produces for them.
    warehouse_ids, sku_ids:
        Optional subsets (used by tests).
    """

    def __init__(self, start: date, seed: int = 42,
                 warehouse_ids: list[str] | None = None, sku_ids: list[str] | None = None) -> None:
        self.start = start
        self.seed  = seed
        self.day   = 0   # index of the next day step() simulates
        self._pairs = [
            _Pair(wh, sku, np.random.default_rng([seed, wi, si]))
            for wi, wh in enumerate(WAREHOUSES) if warehouse_ids is None or wh.wh_id in warehouse_ids
            for si, sku in enumerate(SKUS) if sku_ids is None or sku.sku_id in sku_ids
        ]

    @property
    def next_date(self) -> date:
        return self.start + timedelta(days=self.day)

    def step(self) -> list[Event]:
        """Simulate the next day; return its events in timestamp order."""
        d = self.next_date
        events: list[Event] = []
        for pair in self._pairs:
            seq = iter(range(1_000_000))
            def new_id(_p: _Pair = pair, _seq=seq) -> str:
                return str(uuid.uuid5(_ID_NAMESPACE, f"{self.seed}/{_p.key}/{self.day}/{next(_seq)}"))
            events += pair.step(self.day, d, new_id)
        self.day += 1
        events.sort(key=lambda e: e.payload["timestamp"])
        return events
