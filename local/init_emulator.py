"""
Create, in the local emulators, what Terraform created in GCP.

  Bigtable  instance scip-feature-store, table inventory_features, one column
            family per feature group plus `predictions`; GC keeps 1 version per
            feature family and 3 for predictions (infrastructure/gcp/main.tf).
  Pub/Sub   topic prediction-results and pull subscription prediction-writer
            (infrastructure/gcp/pubsub.tf).

Idempotent: resources that already exist are left alone. The emulators keep
everything in memory, so rerun this after restarting them.

Usage
-----
    docker compose -f local/docker-compose.emulators.yml up -d
    python local/init_emulator.py

Environment (defaults are used when unset; the script prints what it used):
    BIGTABLE_EMULATOR_HOST   default localhost:8086
    PUBSUB_EMULATOR_HOST     default localhost:8085
    GCP_PROJECT_ID           default local-scip  (any string; emulators do not check it)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "spark"))   # config.feature_config — the family list

DEFAULT_BIGTABLE_HOST = "localhost:8086"
DEFAULT_PUBSUB_HOST   = "localhost:8085"
DEFAULT_PROJECT       = "local-scip"

INSTANCE_ID     = "scip-feature-store"
TABLE_ID        = "inventory_features"
TOPIC_ID        = "prediction-results"
SUBSCRIPTION_ID = "prediction-writer"

PREDICTIONS_FAMILY   = "predictions"
PREDICTIONS_VERSIONS = 3     # current + 2 prior rounds (ADR-005 Decision 2)
ACK_DEADLINE_SECONDS = 60

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("init_emulator")


def init_bigtable(project: str) -> None:
    from google.cloud import bigtable
    from google.cloud.bigtable import column_family

    from config.feature_config import FEATURE_REGISTRY

    client   = bigtable.Client(project=project, admin=True)
    table    = client.instance(INSTANCE_ID).table(TABLE_ID)
    if table.exists():
        logger.info("Bigtable: table %s/%s already exists", INSTANCE_ID, TABLE_ID)
        return

    families = {fam: column_family.MaxVersionsGCRule(1) for fam in FEATURE_REGISTRY}
    families[PREDICTIONS_FAMILY] = column_family.MaxVersionsGCRule(PREDICTIONS_VERSIONS)
    table.create(column_families=families)
    logger.info(
        "Bigtable: created %s/%s with families %s",
        INSTANCE_ID, TABLE_ID, ", ".join(sorted(families)),
    )


def init_pubsub(project: str) -> None:
    from google.api_core.exceptions import AlreadyExists
    from google.cloud import pubsub_v1

    publisher  = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
    topic_path = publisher.topic_path(project, TOPIC_ID)
    sub_path   = subscriber.subscription_path(project, SUBSCRIPTION_ID)

    try:
        publisher.create_topic(request={"name": topic_path})
        logger.info("Pub/Sub: created topic %s", topic_path)
    except AlreadyExists:
        logger.info("Pub/Sub: topic %s already exists", topic_path)

    try:
        subscriber.create_subscription(request={
            "name": sub_path,
            "topic": topic_path,
            "ack_deadline_seconds": ACK_DEADLINE_SECONDS,
        })
        logger.info("Pub/Sub: created subscription %s", sub_path)
    except AlreadyExists:
        logger.info("Pub/Sub: subscription %s already exists", sub_path)
    finally:
        subscriber.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Create the local emulator resources the pipeline expects")
    p.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID", DEFAULT_PROJECT))
    args = p.parse_args()

    os.environ.setdefault("BIGTABLE_EMULATOR_HOST", DEFAULT_BIGTABLE_HOST)
    os.environ.setdefault("PUBSUB_EMULATOR_HOST", DEFAULT_PUBSUB_HOST)

    init_bigtable(args.project)
    init_pubsub(args.project)

    print(
        "\nEmulator resources ready. For host-side tools (score.py, monitor.py, dashboards):\n"
        f"  PowerShell: $env:BIGTABLE_EMULATOR_HOST=\"{os.environ['BIGTABLE_EMULATOR_HOST']}\"; "
        f"$env:PUBSUB_EMULATOR_HOST=\"{os.environ['PUBSUB_EMULATOR_HOST']}\"; "
        f"$env:GCP_PROJECT_ID=\"{args.project}\"\n"
        f"  bash:       export BIGTABLE_EMULATOR_HOST={os.environ['BIGTABLE_EMULATOR_HOST']} "
        f"PUBSUB_EMULATOR_HOST={os.environ['PUBSUB_EMULATOR_HOST']} GCP_PROJECT_ID={args.project}\n"
        "Spark containers reach the Bigtable emulator as bigtable-emulator:8086 (see spark/.env.example)."
    )


if __name__ == "__main__":
    main()
