# ADR-003: Bigtable as Online Feature Store

**Status:** Accepted
**Date:** 2026-06-19
**Phase:** 3 — GCS Silver Layer + Online Feature Store
**Deciders:** Platform Engineering

---

## Context

The Supply Chain Intelligence Platform needs to serve pre-computed inventory features to a real-time shortage prediction model (Phase 4). The model runs on every incoming inventory event and must return a prediction within a strict latency budget.

Two storage tiers are required:

1. **Offline store** — full historical feature snapshots for model training and backfill. Must support batch reads of millions of rows across arbitrary time ranges.
2. **Online store** — latest feature snapshot per `(warehouse_id, sku_id)`, served to the inference path with single-digit millisecond latency under concurrent load.

The Bronze → Silver pipeline already produces the feature computation. What remains is a choice of online serving technology and a dual-write architecture.

---

## Decision 1 — Use Bigtable as the Online Feature Store

**Chosen: Google Cloud Bigtable**

### Options evaluated

| Option | P99 Read Latency | Write Throughput | Ops Overhead | GCP Native |
|---|---|---|---|---|
| **Bigtable** | ~1–5 ms | Very high | Low (managed) | Yes |
| Firestore | ~10–50 ms | Medium | Low (managed) | Yes |
| Redis (Cloud Memorystore) | < 1 ms | High | Medium | Partial |
| AlloyDB / Cloud SQL | ~5–20 ms | Low | High (managed SQL) | Yes |
| Pub/Sub → in-memory cache | < 1 ms | — | High | Yes |

### Why Bigtable

**Latency.** Bigtable SSD instances deliver single-digit millisecond reads at the 99th percentile. The inference path requires reading ~29 features for a `(warehouse_id, sku_id)` pair — a single row read, achievable in < 10 ms including network round-trip.

**Throughput.** Bigtable scales to millions of reads/writes per second by adding nodes. The Silver job writes one row per `(warehouse_id, sku_id)` per micro-batch (~120 rows for 6 warehouses × 20 SKUs in the seed). At production scale (thousands of SKUs), `mutate_rows` batch API handles the write volume without throttling.

**Wide-column model.** The feature vector maps naturally to Bigtable's column family model. Six column families (`inventory`, `velocity`, `demand`, `supplier`, `risk`, `lineage`) match the six feature groups, and GC policies can be set independently per family without schema migrations.

**GCP native.** Bigtable integrates with GCS, Dataflow, Vertex AI Feature Store, and Cloud Monitoring without additional connectors. IAM roles (`bigtable.user`) are fine-grained and already provisioned by the Terraform service account.

**Why not Firestore:** Firestore's document model imposes a 1 MB document size limit and ~10–50 ms read latency — too slow for real-time inference and ill-suited to the wide numeric feature vector. Rejected explicitly before Phase 3 design.

**Why not Redis:** Redis (Cloud Memorystore) is faster but requires manual cache warming after restarts, has no built-in column family GC, and adds operational complexity (cluster sizing, eviction policy). For < 10 ms SLA, Bigtable is sufficient and simpler.

---

## Decision 2 — Row Key Design — `{warehouse_id}#{sku_id}`

**Chosen: `{warehouse_id}#{sku_id}` with `#` separator**

### Rationale

**Access pattern.** The primary read pattern is point lookup by `(warehouse_id, sku_id)`. The primary write pattern is upsert of the full feature vector for one pair per Silver batch. A composite key satisfies both without secondary indexes.

**Key distribution.** Bigtable distributes data by row key range across tablets and splits tablets automatically under load. Warehouse IDs (`W-001` through `W-006`) are short, fixed-width prefixes that keep each warehouse's rows contiguous. With 120 rows the table is a single tablet, so no pre-split points are configured. (An earlier version pre-split at `W-2`, `W-4`, `W-6`; every key begins with `W-0`, so all rows sorted before the first split point and the splits did nothing.)

**Separator choice.** `#` (0x23) sorts before alphanumeric characters in ASCII, ensuring warehouse-level range scans (`ReadRows(prefix="W-001#")`) are tight and do not bleed into adjacent warehouses.

**Avoided hotspot patterns:**
- ❌ Sequential timestamp prefix — all writes land on one tablet
- ❌ SKU-only key — cross-warehouse aggregation requires full table scan
- ❌ UUID prefix — destroys locality for warehouse-level range scans

### Key examples

```
W-001#SKU-001   → inventory features for warehouse W-001, SKU SKU-001
W-001#SKU-002
W-002#SKU-001
W-006#SKU-020
```

---

## Decision 3 — Column Families and GC Policy

**Chosen: One column family per feature group; every feature family keeps exactly the latest cell**

### Column family design

| Family | GC Policy | Rationale |
|---|---|---|
| `inventory` | max 1 version | Current stock snapshot — only the latest value is meaningful |
| `velocity` | max 1 version | Rolling windows are recomputed from Bronze on every Silver run; the previous cell is superseded, not aged |
| `demand` | max 1 version | Demand forecasts are replaced, not appended |
| `supplier` | max 1 version | 90-day rolling stats are recomputed by Silver; the cell is the result, not the history |
| `risk` | max 1 version | Only the latest risk score is read |
| `lineage` | max 1 version | Provenance of the current row; the scorer uses `silver_ts` to refuse stale rows |
| `predictions` | max 3 versions | Current prediction plus two prior rounds for rollback inspection (ADR-005 Decision 2) |

### Why not age-based GC

The first version of this table expired `velocity` after 7 days, `risk` after 48 hours and `lineage` after 7 days. That design assumes a store that is written continuously. This one is written in bursts around demo sessions, so after a quiet week the age-based families were empty while `inventory` and `demand` were intact. The scorer then filled the missing velocity features with defaults (`sales_7d` → 0, so `days_of_supply` → 365) and scored every entity as low risk: a stale store became a confident all-clear. Keeping the latest cell and letting the scorer reject rows whose `lineage:silver_ts` is older than its freshness threshold (`score.py --max-feature-age-hours`, default 48) fails loudly instead.

### Why not a single family

The families still group the 29 features into the six logical domains used throughout Silver, the Bigtable writer and the scorer, and `predictions` needs a different policy (3 versions) from the feature families (1).

### Why not one family per feature

29 families for 29 features would complicate the GC policy management and exceed Bigtable's recommended maximum of ~100 column families per table. Six families match the logical feature groups already established in the data model.

---

## Decision 4 — Value Encoding — Big-Endian Double (IEEE 754)

**Chosen: `struct.pack(">d", float(value))` for numeric features; UTF-8 bytes for string features**

### Rationale

Bigtable stores cell values as raw bytes. Two encoding options were considered:

| Option | Bytes per value | Sortable | Decode complexity |
|---|---|---|---|
| **Big-endian double** | 8 | No (signed) | `struct.unpack(">d", cell.value)` |
| JSON string | 5–20+ | No | `float(json.loads(cell.value))` |
| Little-endian double | 8 | No | Inconsistent across systems |
| Protobuf | 2–10 (varint) | No | Schema dependency |

**Big-endian double** is chosen for:
- **Fixed size** — 8 bytes per cell, predictable storage cost
- **Interoperability** — big-endian (network byte order) is the standard for cross-language numeric exchange; Python `struct`, Java `ByteBuffer.putDouble()`, and Go `binary.BigEndian.Float64()` all use the same wire format
- **Zero schema dependency** — no Protobuf registry, no Avro schema; any consumer that knows the qualifier name can decode the value

String features (`last_event_type`, `last_signal_type`, timestamps, `pipeline_version`) are stored as UTF-8 bytes — the natural encoding for human-readable strings in Bigtable.

---

## Decision 5 — Dual-Write Pattern — Silver First, Bigtable Best-Effort

**Chosen: Write Silver GCS first; Bigtable write in isolated try/except; Silver is source of truth**

### Write sequence per micro-batch

```
foreachBatch
  └── feature_df.cache()
       ├── write_silver(feature_df)          # [1] Primary — durable, immutable
       └── write_to_bigtable(feature_df)     # [2] Secondary — online cache
            └── on failure: log + continue   # Silver unaffected
       └── feature_df.unpersist()
```

### Failure modes

| Failure | Silver | Bigtable | Recovery |
|---|---|---|---|
| Bigtable write fails | Written | Stale/missing | Backfill from Silver Parquet via batch job |
| Silver write fails | Not written | Not attempted | Spark retries epoch from checkpoint |
| Both fail | Not written | Not attempted | Spark retries epoch from checkpoint |
| Spark executor crash | Checkpoint ensures at-least-once | Idempotent upsert (row key = natural dedup key) | Auto-recovery |

### Why Silver first

Silver Parquet on GCS is the durable offline store. It supports model training, backfill, and audit. If Bigtable becomes unavailable, Silver allows a full rebuild of the online store by replaying Parquet files through a batch writer — at the cost of Bigtable being stale until the backfill completes, which is operationally acceptable (degraded mode: inference falls back to last-known features).

Reversing the order (Bigtable first) would mean a Silver write failure causes Bigtable to hold features with no corresponding offline record — a lineage gap that breaks training pipelines.

### `feature_df.cache()` before dual-write

Without caching, the Silver write triggers a full DAG re-execution when Bigtable subsequently consumes the same DataFrame. Caching materialises the feature DataFrame once, then both sinks read from the cached RDD partitions. This halves the compute cost of each micro-batch.

---

## Decision 6 — `foreachPartition` for Bigtable Writes (not `collect` to Driver)

**Chosen: `df.foreachPartition(lambda partition: ...)` with per-partition Bigtable client**

### Rationale

Three approaches were considered for writing a Spark DataFrame to a non-Spark sink:

| Approach | Scalability | Network | Failure isolation |
|---|---|---|---|
| `df.collect()` → driver write | Poor — all data to driver | Driver → Bigtable | Driver OOM on large batches |
| `df.foreachRow()` | Poor — one RPC per row | Executor → Bigtable | Per-row failure |
| **`df.foreachPartition()`** | Good — parallel, bounded | Executor → Bigtable | Per-partition failure |

`foreachPartition` is the correct pattern because:
- Each Spark executor opens its own `bigtable.Client` — no client serialisation across the wire
- Mutations are batched within each partition using `mutate_rows` (up to `BIGTABLE_BATCH_SIZE=500` rows per RPC)
- The Bigtable client is closed in a `finally` block even if the partition fails
- Partition-level parallelism matches Spark's natural execution model

For the SCIP seed data (6 warehouses × 20 SKUs = 120 rows), the difference is negligible. The `foreachPartition` pattern is used because it is correct at production scale.

---

## Decision 7 — SA Key via `gcloud` CLI, Not Terraform

**Chosen: SA key created manually via `gcloud iam service-accounts keys create`; not managed by Terraform**

### Rationale

Terraform stores all resource state in the state file. If an SA key is created via `google_service_account_key`, the private key JSON appears in plaintext in `terraform.tfstate`. This creates a secret management problem:

- State files are often stored in GCS (our `backend.tf` does exactly this)
- Anyone with read access to the state bucket can extract the key
- Rotating the key requires a Terraform apply, which re-creates the key in state

The manual approach (`gcloud iam service-accounts keys create keys/scip-spark-sa.json`) keeps the key out of state. The key file is mounted read-only into the Spark Docker container and is gitignored. Key rotation is a one-command operation that does not touch Terraform state.

---

## Consequences

### Positive

- **< 10 ms inference reads** — Bigtable SSD delivers the latency budget required by Phase 4
- **Independent GC** per feature family — no manual deletion jobs needed
- **Silver as durable source of truth** — Bigtable can always be rebuilt from Silver
- **Dual-write cost** is one extra `foreachPartition` pass over the same cached DataFrame per micro-batch
- **Schema-free** — adding a new feature column requires no DDL; add the qualifier in `feature_config.py` and deploy

### Negative / Trade-offs

- **Bigtable cost** — a node bills about $0.65 per node-hour (~$475/month at us-central1 list price) whether the instance is DEVELOPMENT or PRODUCTION; there is no free tier, and PRODUCTION no longer requires three nodes (Google has deprecated the DEVELOPMENT type). This is the project's dominant cost, so the instance is destroyed between demo sessions and rebuilt from Silver, the durable source of truth.
- **No ad-hoc SQL** — Bigtable does not support SQL. Range scans require careful key design. Complex aggregations must be done in Silver (Spark SQL), not Bigtable.
- **Cold start** — first `terraform apply` + Silver batch before Bigtable has any data. Inference must handle "feature not found" gracefully until the first batch writes.
- **At-least-once writes** — `foreachBatch` at-least-once semantics means a row may be re-written on Spark retry. Bigtable upsert semantics (overwrite on same row key) make re-writes idempotent for the latest-value pattern.

---

## References

- `infrastructure/gcp/main.tf` — Bigtable instance, table, GC policies, IAM
- `spark/utils/bigtable_writer.py` — `write_to_bigtable()`, `foreachPartition` pattern
- `spark/jobs/silver_feature_engineering.py` — dual-write in `process_silver_batch()`
- `spark/config/feature_config.py` — `FEATURE_REGISTRY`, `BIGTABLE_BATCH_SIZE`, GC thresholds
- `spark/jobs/schemas/feature_schema.py` — `SILVER_FEATURE_SCHEMA`, row key convention
