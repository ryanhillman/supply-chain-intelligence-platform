"""
Bronze ingestion configuration.
All values read from environment variables; .env loaded by Docker Compose.
"""
from __future__ import annotations

import os
from starlette.config import Config

config = Config(environ=os.environ)

# ── Kafka ─────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS: str = config("KAFKA_BOOTSTRAP_SERVERS", default="kafka:29092")
KAFKA_GROUP_ID: str = config("KAFKA_GROUP_ID", default="scip-spark-bronze")
STARTING_OFFSETS: str = config("STARTING_OFFSETS", default="latest")

# ── Spark ─────────────────────────────────────────────────────────────────────
SPARK_MASTER: str = config("SPARK_MASTER", default="local[*]")
TRIGGER_INTERVAL: str = config("TRIGGER_INTERVAL", default="30 seconds")

# ── GCP ───────────────────────────────────────────────────────────────────────
GCP_PROJECT_ID: str = config("GCP_PROJECT_ID", default="")
GCS_BUCKET_NAME: str = config("GCS_BUCKET_NAME", default="")
GOOGLE_APPLICATION_CREDENTIALS: str = config(
    "GOOGLE_APPLICATION_CREDENTIALS",
    default="/opt/spark-app/keys/scip-spark-sa.json",
)

# ── Storage paths ─────────────────────────────────────────────────────────────
# Phase 2+: gs://... paths written to GCS.
# Phase 1 fallback: local data/ paths (for unit tests / no-GCP dev).
BRONZE_BASE_PATH: str = config("BRONZE_BASE_PATH", default="data/bronze")
CHECKPOINT_BASE_PATH: str = config(
    "CHECKPOINT_BASE_PATH", default="data/checkpoints/scip-spark-bronze"
)
DLQ_BASE_PATH: str = config("DLQ_BASE_PATH", default="data/dlq")

# ── DLQ ───────────────────────────────────────────────────────────────────────
# "file"  → FileDlqHandler  (Phase 2)
# "kafka" → KafkaDlqHandler (Phase 3 Step 3)
DLQ_SINK: str = config("DLQ_SINK", default="file")

# ── Bigtable (Phase 3 Steps 5-7) ─────────────────────────────────────────────
BIGTABLE_INSTANCE_ID: str = config("BIGTABLE_INSTANCE_ID", default="scip-feature-store")
BIGTABLE_TABLE_ID: str = config("BIGTABLE_TABLE_ID", default="inventory_features")

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

# ── Partitioning ──────────────────────────────────────────────────────────────
PARTITION_COLUMNS: list[str] = ["year", "month", "day", "hour"]
