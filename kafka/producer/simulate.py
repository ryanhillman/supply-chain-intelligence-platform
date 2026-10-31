"""
Kafka producer: replays the world model (simulation/world.py) onto the five
source topics.

By default it simulates REPLAY_DAYS days from REPLAY_START as fast as Kafka
accepts them, flushes, and exits. Every event's `timestamp` field is its
simulated time, so Bronze partitions the history under the right dates. The
Kafka record timestamp stays wall-clock, so topic retention counts from when
the replay ran.

TICK_SECONDS_PER_DAY > 0 keeps producing after the replay, one simulated day
every that many seconds, for live demos. It is off by default: at a few
seconds per day, continuous ticking adds decades of simulated history per
wall-clock day.

The same seed reproduces the same events and event IDs, so running the
replay twice duplicates records harmlessly (Silver and the training sampler
drop duplicate event IDs).

Environment:
  KAFKA_BOOTSTRAP_SERVERS  default kafka:29092
  REPLAY_START             default 2025-10-01
  REPLAY_DAYS              default 365
  SEED                     default 42
  TICK_SECONDS_PER_DAY     default 0 (stop after the replay)
  LOG_LEVEL                default INFO
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from confluent_kafka import Producer

# In the image, simulation/ sits next to this file; in the repo it is two levels up.
for _root in Path(__file__).resolve().parents:
    if (_root / "simulation").is_dir():
        sys.path.insert(0, str(_root))
        break

from simulation.world import Event, World  # noqa: E402

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
REPLAY_START      = date.fromisoformat(os.environ.get("REPLAY_START", "2025-10-01"))
REPLAY_DAYS       = int(os.environ.get("REPLAY_DAYS", "365"))
SEED              = int(os.environ.get("SEED", "42"))
TICK_SECONDS      = float(os.environ.get("TICK_SECONDS_PER_DAY", "0"))
LOG_LEVEL         = os.environ.get("LOG_LEVEL", "INFO")

PRODUCER_ID = "scip-simulator"
LOG_EVERY_DAYS = 30

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("simulate")


def _delivery_callback(err, msg) -> None:
    if err:
        logger.error("Delivery failed | topic=%s error=%s", msg.topic(), err)


def produce_events(p: Producer, events: list[Event]) -> None:
    """Send one day's events; wait for queue space when the local buffer is full."""
    for e in events:
        value = json.dumps(e.payload)
        while True:
            try:
                p.produce(e.topic, key=e.key, value=value, callback=_delivery_callback)
                break
            except BufferError:
                p.poll(1.0)
        p.poll(0)


def make_producer() -> Producer:
    return Producer({
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "client.id":         PRODUCER_ID,
        "acks":              "all",
        "retries":           3,
        "retry.backoff.ms":  500,
        "linger.ms":         50,
    })


def _last_day(world: World) -> str:
    """The last simulated day (step() has already advanced next_date past it)."""
    return (world.next_date - timedelta(days=1)).isoformat()


def replay(p: Producer, world: World, days: int) -> int:
    """Produce `days` simulated days; return the number of events sent."""
    total = 0
    for i in range(1, days + 1):
        events = world.step()
        produce_events(p, events)
        total += len(events)
        if i % LOG_EVERY_DAYS == 0 or i == days:
            p.flush()
            logger.info("Replayed %d/%d days | through %s | events=%d",
                        i, days, _last_day(world), total)
    return total


def main() -> None:
    logger.info(
        "World replay starting | bootstrap=%s start=%s days=%d seed=%d tick=%ss",
        BOOTSTRAP_SERVERS, REPLAY_START, REPLAY_DAYS, SEED, TICK_SECONDS or "off",
    )

    p: Producer | None = None
    for attempt in range(1, 11):
        try:
            p = make_producer()
            break
        except Exception as exc:
            logger.warning("Producer connection attempt %d failed: %s — retrying in 5s", attempt, exc)
            time.sleep(5)
    if p is None:
        logger.error("Could not connect to Kafka after 10 attempts — exiting")
        sys.exit(1)

    world = World(REPLAY_START, seed=SEED)
    total = replay(p, world, REPLAY_DAYS)
    logger.info("Replay complete | %d events through %s", total, _last_day(world))

    if TICK_SECONDS <= 0:
        return
    logger.info("Ticking one simulated day every %ss", TICK_SECONDS)
    while True:
        time.sleep(TICK_SECONDS)
        produce_events(p, world.step())
        p.flush()


if __name__ == "__main__":
    main()
