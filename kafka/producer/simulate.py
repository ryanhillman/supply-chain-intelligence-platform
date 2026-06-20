"""
Supply Chain Event Simulator — Phase 1/2

Produces realistic inventory events to all 5 Kafka topics continuously.

Topology:
  6 warehouses  (W-001 .. W-006)
  20 SKUs       (SKU-001 .. SKU-020)
  4 suppliers   (SUP-001 .. SUP-004)

Per 2-second cycle:
  inventory.events        — 10 events (mix of SALE, RECEIVE, ADJUSTMENT, RETURN, TRANSFER)
  demand.signals          —  3 events (FORECAST, SURGE, TREND, SEASONAL)
  purchase.orders.events  —  2 events (CREATED, SUBMITTED, APPROVED, RECEIVED)
  shipments.events        —  2 events (CREATED, SHIPPED, DELIVERED)
  suppliers.events        —  1 event  (PERFORMANCE_UPDATE, every 5th cycle only)

All fields match the PySpark Bronze schemas in spark/jobs/schemas/.
Required fields guaranteed present and non-null on every event.

Configuration via environment variables:
  KAFKA_BOOTSTRAP_SERVERS  default: kafka:29092
  EVENTS_PER_BATCH         default: 17
  BATCH_SLEEP_SECONDS      default: 2
  LOG_LEVEL                default: INFO
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone

from confluent_kafka import Producer

# ── Configuration ──────────────────────────────────────────────────────────────

BOOTSTRAP_SERVERS: str = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
BATCH_SLEEP: float      = float(os.environ.get("BATCH_SLEEP_SECONDS", "2"))
LOG_LEVEL: str          = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("simulate")

# ── Domain constants ───────────────────────────────────────────────────────────

WAREHOUSES: list[str] = [f"W-{i:03d}" for i in range(1, 7)]
SKUS: list[str]       = [f"SKU-{i:03d}" for i in range(1, 21)]
SUPPLIERS: list[str]  = [f"SUP-{i:03d}" for i in range(1, 5)]

SCHEMA_VERSION = "1.0"
PRODUCER_ID    = "scip-simulator"

# One primary supplier per warehouse (round-robin assignment)
WAREHOUSE_SUPPLIER: dict[str, str] = {
    wh: SUPPLIERS[i % len(SUPPLIERS)]
    for i, wh in enumerate(WAREHOUSES)
}

# Reorder parameters per SKU (static, simulating catalogue data)
REORDER_PARAMS: dict[str, dict] = {
    sku: {
        "reorder_point": random.randint(20, 60),
        "safety_stock":  random.randint(10, 25),
        "unit_cost":     round(random.uniform(5.0, 500.0), 2),
    }
    for sku in SKUS
}

# ── Inventory state ────────────────────────────────────────────────────────────
# Tracks current quantity per (warehouse, sku) pair.
# Initialised with random starting inventory.

inventory: dict[tuple[str, str], int] = {
    (wh, sku): random.randint(50, 500)
    for wh in WAREHOUSES
    for sku in SKUS
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def new_id() -> str:
    return str(uuid.uuid4())

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def delivery_callback(err, msg) -> None:
    if err:
        logger.error("Delivery failed | topic=%s partition=%d error=%s",
                     msg.topic(), msg.partition(), err)


# ── Event producers ────────────────────────────────────────────────────────────

def produce_inventory_event(p: Producer) -> None:
    """
    Produces one inventory.events message.

    Event types weighted to reflect real warehouse operations:
      SALE        60% — most common; decreases stock
      RECEIVE     20% — restocking; increases stock
      ADJUSTMENT  10% — cycle count correction; small delta either direction
      RETURN       5% — customer return; increases stock
      TRANSFER     5% — inter-warehouse; decreases stock at source
    """
    wh  = random.choice(WAREHOUSES)
    sku = random.choice(SKUS)
    key = (wh, sku)

    qty_before = inventory[key]
    params     = REORDER_PARAMS[sku]

    event_type = random.choices(
        ["SALE", "RECEIVE", "ADJUSTMENT", "RETURN", "TRANSFER"],
        weights=[60, 20, 10, 5, 5],
    )[0]

    if event_type == "SALE":
        # Sell between 1 and 20 units; can't sell more than in stock
        delta = -random.randint(1, max(1, min(20, qty_before)))
    elif event_type == "RECEIVE":
        # Restocking order arrival
        delta = random.randint(10, 100)
    elif event_type == "ADJUSTMENT":
        # Small positive or negative correction
        delta = random.randint(-10, 10)
    elif event_type == "RETURN":
        # Customer return adds back stock
        delta = random.randint(1, 5)
    else:
        # TRANSFER — removes stock (destination warehouse gets RECEIVE separately)
        delta = -random.randint(1, max(1, min(10, qty_before)))

    qty_after = max(0, qty_before + delta)
    inventory[key] = qty_after
    actual_delta   = qty_after - qty_before

    event = {
        "event_id":        new_id(),
        "schema_version":  SCHEMA_VERSION,
        "event_type":      event_type,
        "warehouse_id":    wh,
        "sku_id":          sku,
        "quantity_before": qty_before,
        "quantity_after":  qty_after,
        "quantity_delta":  actual_delta,
        "unit_cost":       params["unit_cost"],
        "reorder_point":   params["reorder_point"],
        "safety_stock":    params["safety_stock"],
        "reference_id":    None,
        "timestamp":       now_iso(),
        "producer_id":     PRODUCER_ID,
        "correlation_id":  new_id(),
        "metadata":        None,
    }

    p.produce(
        "inventory.events",
        key=f"{wh}#{sku}",
        value=json.dumps(event),
        callback=delivery_callback,
    )


def produce_demand_signal(p: Producer) -> None:
    """
    Produces one demand.signals message.

    Note: demand.signals uses signal_type (not event_type) — matches
    DEMAND_SIGNAL_SCHEMA and the Bronze normalisation in bronze_ingestion.py.
    """
    wh  = random.choice(WAREHOUSES)
    sku = random.choice(SKUS)

    signal = {
        "event_id":             new_id(),
        "schema_version":       SCHEMA_VERSION,
        "signal_type":          random.choice(["FORECAST", "SURGE", "TREND", "SEASONAL"]),
        "warehouse_id":         wh,
        "sku_id":               sku,
        "demand_quantity":      round(random.uniform(10.0, 200.0), 1),
        "confidence_score":     round(random.uniform(0.5, 1.0), 2),
        "signal_source":        random.choice(["ML_MODEL", "HISTORICAL", "MANUAL"]),
        "forecast_horizon_days": random.choice([7, 14, 30]),
        "timestamp":            now_iso(),
        "producer_id":          PRODUCER_ID,
        "correlation_id":       new_id(),
        "metadata":             None,
    }

    p.produce(
        "demand.signals",
        key=f"{wh}#{sku}",
        value=json.dumps(signal),
        callback=delivery_callback,
    )


def produce_purchase_order(p: Producer) -> None:
    """
    Produces one purchase.orders.events message.

    Status distribution:
      CREATED    40% — PO raised but not yet submitted
      SUBMITTED  30% — PO sent to supplier (Silver: counted in active_po_qty)
      APPROVED   20% — Supplier confirmed
      RECEIVED   10% — Goods arrived (quantity_received populated)
    """
    wh       = random.choice(WAREHOUSES)
    sku      = random.choice(SKUS)
    supplier = WAREHOUSE_SUPPLIER[wh]

    event_type    = random.choices(
        ["CREATED", "SUBMITTED", "APPROVED", "RECEIVED"],
        weights=[40, 30, 20, 10],
    )[0]

    qty_ordered = random.randint(50, 500)
    unit_cost   = REORDER_PARAMS[sku]["unit_cost"]

    po = {
        "event_id":             new_id(),
        "schema_version":       SCHEMA_VERSION,
        "event_type":           event_type,
        "warehouse_id":         wh,
        "sku_id":               sku,
        "order_id":             f"PO-{new_id()[:8].upper()}",
        "supplier_id":          supplier,
        "quantity_ordered":     qty_ordered,
        "quantity_received":    qty_ordered if event_type == "RECEIVED" else None,
        "unit_cost":            unit_cost,
        "total_cost":           round(qty_ordered * unit_cost, 2),
        "expected_delivery_date": now_iso(),
        "actual_delivery_date": now_iso() if event_type == "RECEIVED" else None,
        "timestamp":            now_iso(),
        "producer_id":          PRODUCER_ID,
        "correlation_id":       new_id(),
    }

    p.produce(
        "purchase.orders.events",
        key=f"{wh}#{sku}",
        value=json.dumps(po),
        callback=delivery_callback,
    )


def produce_shipment(p: Producer) -> None:
    """
    Produces one shipments.events message.

    Status distribution:
      CREATED    30% — shipment initiated
      SHIPPED    50% — in transit (Silver: counted in in_transit_qty)
      DELIVERED  20% — arrived; actual_arrival populated
    """
    wh       = random.choice(WAREHOUSES)
    sku      = random.choice(SKUS)
    supplier = WAREHOUSE_SUPPLIER[wh]

    event_type = random.choices(
        ["CREATED", "SHIPPED", "DELIVERED"],
        weights=[30, 50, 20],
    )[0]

    shipment = {
        "event_id":         new_id(),
        "schema_version":   SCHEMA_VERSION,
        "event_type":       event_type,
        "warehouse_id":     wh,
        "sku_id":           sku,
        "shipment_id":      f"SHP-{new_id()[:8].upper()}",
        "supplier_id":      supplier,
        "quantity":         random.randint(10, 200),
        "carrier":          random.choice(["FEDEX", "UPS", "DHL", "USPS"]),
        "tracking_number":  f"TRK{random.randint(100000, 999999)}",
        "estimated_arrival": now_iso(),
        "actual_arrival":   now_iso() if event_type == "DELIVERED" else None,
        "timestamp":        now_iso(),
        "producer_id":      PRODUCER_ID,
        "correlation_id":   new_id(),
        "metadata":         None,
    }

    p.produce(
        "shipments.events",
        key=f"{wh}#{sku}",
        value=json.dumps(shipment),
        callback=delivery_callback,
    )


def produce_supplier_event(p: Producer) -> None:
    """
    Produces one suppliers.events message.

    Supplier events are infrequent (once per 5 cycles = ~every 10s).
    They carry lead_time_days and on_time_delivery_rate, which Silver
    aggregates per warehouse for supplier feature computation.

    affected_sku_ids is a JSON array of strings, matching ArrayType(StringType())
    in SUPPLIER_EVENT_SCHEMA.
    """
    wh       = random.choice(WAREHOUSES)
    supplier = WAREHOUSE_SUPPLIER[wh]

    event = {
        "event_id":             new_id(),
        "schema_version":       SCHEMA_VERSION,
        "event_type":           "PERFORMANCE_UPDATE",
        "supplier_id":          supplier,
        "warehouse_id":         wh,
        "lead_time_days":       random.randint(3, 21),
        "on_time_delivery_rate": round(random.uniform(0.70, 1.0), 2),
        "reliability_score":    round(random.uniform(0.60, 1.0), 2),
        "affected_sku_ids":     random.sample(SKUS, random.randint(1, 5)),
        "timestamp":            now_iso(),
        "producer_id":          PRODUCER_ID,
        "correlation_id":       new_id(),
        "metadata":             None,
    }

    p.produce(
        "suppliers.events",
        key=supplier,
        value=json.dumps(event),
        callback=delivery_callback,
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

def make_producer() -> Producer:
    return Producer({
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "client.id":         PRODUCER_ID,
        "acks":              "all",
        "retries":           3,
        "retry.backoff.ms":  500,
    })


def main() -> None:
    logger.info(
        "Supply chain simulator starting | bootstrap=%s warehouses=%d skus=%d suppliers=%d",
        BOOTSTRAP_SERVERS, len(WAREHOUSES), len(SKUS), len(SUPPLIERS),
    )

    # Retry connection to Kafka — broker may not be ready immediately
    p: Producer | None = None
    for attempt in range(1, 11):
        try:
            p = make_producer()
            logger.info("Kafka producer connected | attempt=%d", attempt)
            break
        except Exception as exc:
            logger.warning("Producer connection attempt %d failed: %s — retrying in 5s", attempt, exc)
            time.sleep(5)

    if p is None:
        logger.error("Could not connect to Kafka after 10 attempts — exiting")
        return

    batch = 0
    total_events = 0

    while True:
        batch += 1

        # ── inventory.events (most frequent — drives Silver trigger cadence) ──
        for _ in range(10):
            produce_inventory_event(p)

        # ── demand.signals ────────────────────────────────────────────────────
        for _ in range(3):
            produce_demand_signal(p)

        # ── purchase.orders.events ────────────────────────────────────────────
        for _ in range(2):
            produce_purchase_order(p)

        # ── shipments.events ──────────────────────────────────────────────────
        for _ in range(2):
            produce_shipment(p)

        # ── suppliers.events (infrequent) ─────────────────────────────────────
        supplier_count = 0
        if batch % 5 == 0:
            produce_supplier_event(p)
            supplier_count = 1

        # Flush triggers delivery callbacks and ensures all messages are sent
        p.poll(0)
        p.flush()

        batch_total = 10 + 3 + 2 + 2 + supplier_count
        total_events += batch_total

        logger.info(
            "Batch %d | inventory=10 demand=3 po=2 shipment=2 supplier=%d | "
            "batch_total=%d cumulative=%d",
            batch, supplier_count, batch_total, total_events,
        )

        time.sleep(BATCH_SLEEP)


if __name__ == "__main__":
    main()
