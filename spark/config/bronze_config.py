"""
Bronze ingestion configuration.
All values read from environment variables; .env loaded by Docker Compose.
"""
from __future__ import annotations

import os

# ── Kafka ─────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS: str = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_GROUP_ID: str = os.environ.get("KAFKA_GROUP_ID", "scip-spark-bronze")
# earliest: the producer replays a year of simulated history and exits, so Bronze
# must read the topics from the start. Only the first run uses this; after that
# Spark resumes from the offsets stored in each query's checkpoint.
STARTING_OFFSETS: str = os.environ.get("KAFKA_STARTING_OFFSETS", "earliest")

# ── Spark ─────────────────────────────────────────────────────────────────────
TRIGGER_INTERVAL: str = os.environ.get("TRIGGER_INTERVAL", "30 seconds")

# ── GCP ───────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID: str = os.environ.get("GCP_PROJECT_ID", "")

# ── Storage paths ─────────────────────────────────────────────────────────────
# Local filesystem by default (spark/data/ through the /opt/spark-app mount;
# see .env.example). The June 2026 GCP deployment set these to gs:// URIs.
BRONZE_BASE_PATH: str = os.environ.get("BRONZE_BASE_PATH", "data/bronze")
SILVER_BASE_PATH: str = os.environ.get("SILVER_BASE_PATH", "data/silver")
CHECKPOINT_BASE_PATH: str = os.environ.get(
    "CHECKPOINT_BASE_PATH", "data/checkpoints/scip-spark-bronze"
)
DLQ_BASE_PATH: str = os.environ.get("DLQ_BASE_PATH", "data/dlq")

# ── Bigtable ──────────────────────────────────────────────────────────────────
BIGTABLE_INSTANCE_ID: str = os.environ.get("BIGTABLE_INSTANCE_ID", "scip-feature-store")
BIGTABLE_TABLE_ID: str = os.environ.get("BIGTABLE_TABLE_ID", "inventory_features")

# ── Topic configuration ───────────────────────────────────────────────────────
SOURCE_TOPICS: list[str] = [
    "inventory.events",
    "shipments.events",
    "purchase.orders.events",
    "demand.signals",
    "suppliers.events",
]

# Required fields per topic — missing any field routes the record to DLQ.
TOPIC_REQUIRED_FIELDS: dict[str, list[str]] = {
    "inventory.events":       ["event_id", "event_type", "warehouse_id", "sku_id", "timestamp"],
    "shipments.events":       ["event_id", "event_type", "shipment_id", "timestamp"],
    "purchase.orders.events": ["event_id", "event_type", "order_id", "timestamp"],
    "demand.signals":         ["event_id", "signal_type", "sku_id", "timestamp"],
    "suppliers.events":       ["event_id", "event_type", "supplier_id", "timestamp"],
}

