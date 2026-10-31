# ADR-006: One Simulated World for Training and Scoring

**Status:** Accepted, October 2026

## Context

The model was trained and scored on data from two different simulators. Training rows came from a generator that computed features directly from its own day-level simulation and never passed through Kafka, Bronze or Silver. Scoring read Silver features computed from events produced by a separate random-event producer, with unrelated demand, stock and lead-time behavior. Before that, training tiled one day of Bronze events 60 times, which produced the train/serve shift recorded in ADR-005 Decision 7. In both setups the test metrics described a different world from the one being scored.

## Decision 1 — One world model, replayed through the pipeline

`simulation/world.py` models 6 warehouses x 20 SKUs one day at a time: gamma-distributed demand with seasonality and promotional spikes, a reorder-point policy with one open order per pair, and supplier lead times with a Poisson delay for late orders. Its parameters are the former generator's. It emits events, not features.

The Kafka producer replays 365 simulated days (2025-10-01 to 2026-09-30, seed 42) and exits; Bronze stores each event under its simulated date. The training sampler builds rows from Bronze, and Silver computes scoring features from the same Bronze. The generator that bypassed the pipeline, and event tiling, are removed. This replaces the training-data source described in ADR-004 Decision 3.

## Decision 2 — Events carry what the features need

- Orders keep one `order_id` from CREATED to RECEIVED, and shipments one `shipment_id` from SHIPPED to DELIVERED, so open quantities can be netted.
- Each completed order publishes a supplier PERFORMANCE_UPDATE with the actual lead time and an on-time value of 1 or 0, so the mean over a window is the supplier's on-time rate.
- Every pair posts one SALE per day, with zero units when out of stock. A stocked-out pair stays visible to Silver, and the label generator sees the days at zero.
- Demand signals are a Poisson count around expected demand (seasonality and promotions), not around realized demand, so the feature carries signal without revealing the outcome.
- Event IDs are derived from the seed, pair, day and sequence. A repeated replay produces identical IDs, and Silver and the sampler drop duplicates.

## Decision 3 — Silver is a batch job as of a date

`spark/jobs/silver_batch.py` computes one snapshot per pair for the end of an as-of day, from events before the following midnight; the default is the latest day in Bronze. Rerunning a date replaces that date's Silver partitions. The streaming Silver query is deleted: replayed history arrives in bursts, and scoring consumes one snapshot per run. `lineage:silver_ts` remains the wall-clock write time that the scorer's freshness check reads.

## Decision 4 — Identical feature windows, tested in CI

Training features come from the pandas sampler, one row per pair per day; scoring features come from Spark, one snapshot per as-of date. Both now use the same windows, all ending at midnight after the date D:

| Feature | Window |
|---|---|
| `sales_7d`, `stddev_daily_7d` | D-6 to D |
| `sales_30d`, averages, trend | D-29 to D |
| `demand_signal_count_24h` | D |
| supplier lead time and reliability | D-90 to D, per warehouse |
| `active_po_qty` | orders created D-90 to D, not yet received |
| `in_transit_qty` | shipments shipped D-30 to D, not yet delivered |

Aligning them fixed four differences: Silver's 30-day window covered 31 days and its 7-day window 8, both sides counted demand signals over 48 hours, and neither netted out received orders or delivered shipments. `tests/test_spark_parity.py` runs Silver's `compute_features` and the sampler on the same replayed fixture and requires every raw feature to match; CI runs it in its own job with PySpark and Java 17.

## Decision 5 — Bronze settings for a replay

Bronze starts from the earliest offset, and every topic keeps 30 days (demand signals previously kept one). Partitions are event type and date; the hour level is dropped because events are day-level. Rows are repartitioned by partition columns before each write: on the first replay batch, every Kafka-partition task writing into every date directory stalled the driver past Spark's heartbeat timeout.

## Consequences

- Test metrics describe the model on the world it is scored on. On the default replay's test split
  (6,000 rows) the model reaches AUPRC 0.839 against 0.639 for the formula baseline.
- The data is still simulated. Pairs are independent: no transfers between warehouses, no substitution between SKUs, no shared supplier capacity.
- Supplier features stay aggregated per warehouse, as Silver computed them, although suppliers are assigned per product category.
- ADR-005 Decision 7's shift cannot recur between training and scoring by construction; the drift monitor still guards against changes in the world itself.
- The default replay is about 200,000 events. On Docker Desktop the producer finishes in under a
  minute, Bronze ingests the events in about 10 minutes, and Silver takes about 5.
