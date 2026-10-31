# ADR-003: Bigtable as the Online Feature Store

**Status:** Accepted, June 2026

## Context

Silver computes 29 features per (warehouse, SKU) pair and writes them as Parquet. The batch scorer needs the latest feature row for every pair at scoring time: 120 rows for 6 warehouses and 20 SKUs.

## Decision 1 — Bigtable for the online store

Bigtable was selected as a learning objective: row-key design, column families, GC policies, and writes from Spark executors. The data volume does not require it. 120 rows fit in Postgres, in Redis, or in the Silver Parquet output read directly by the scorer. The system is sized for the stack, not for the data.

Cost: a single node bills about $0.65 per hour (about $475 per month at us-central1 list price) regardless of row count, and there is no free tier. It is the dominant cost line, so the instance is destroyed between demo sessions and rebuilt from Silver.

The data model does fit Bigtable: one wide row per entity, features grouped into families, and only the latest value read.

## Decision 2 — Row key `{warehouse_id}#{sku_id}`

Reads are point lookups or full scans, and each write replaces one pair's feature vector, so the composite key serves both without secondary indexes. The warehouse prefix keeps each warehouse's rows contiguous, and `#` (0x23) sorts before alphanumerics, so a prefix scan on `W-001#` does not bleed into the adjacent warehouse.

At 120 rows the table is a single tablet. Pre-split points at `W-2`, `W-4` and `W-6` were removed: every key begins with `W-0`, so all rows sorted before the first split and the splits had no effect.

## Decision 3 — One family per feature group, latest cell only

The six families (`inventory`, `velocity`, `demand`, `supplier`, `risk`, `lineage`) match the feature groups used by Silver, the writer and the scorer. Each keeps exactly one version. `predictions` keeps three (ADR-005 Decision 2).

The initial design used age-based GC: `velocity` expired after 7 days, `risk` after 48 hours, `lineage` after 7 days. That assumes continuous writes. The store was written in bursts around demo sessions, so after a quiet week `velocity` was empty while `inventory` and `demand` were intact. The scorer filled the missing velocity features with defaults (`sales_7d` → 0, so `days_of_supply` → 365) and scored every entity as low risk: a stale store produced a confident all-clear.

Every family now keeps its latest cell, and the scorer rejects rows whose `lineage:silver_ts` is older than the freshness limit (`score.py --max-feature-age-hours`, default 48). Staleness fails loudly instead of silently.

## Decision 4 — Big-endian doubles for numbers, UTF-8 for strings

Numeric features are stored as `struct.pack(">d", value)`: a fixed 8 bytes, network byte order readable from any language, and no schema registry. String features (event types, timestamps, pipeline version) are stored as UTF-8.

## Decision 5 — Write Silver first, Bigtable best-effort

Each Silver batch caches the feature DataFrame, writes Silver Parquet, then writes Bigtable inside its own try/except. Silver is the source of truth: a failed Bigtable write leaves Silver intact, and Bigtable is rebuilt from it. This is also what makes destroying the instance between sessions safe. Writing Bigtable first would risk online features with no offline record behind them.

Caching before the two writes prevents Spark from recomputing the plan for the second sink.

## Decision 6 — Write from executors with `foreachPartition`

Each partition opens its own Bigtable client, batches mutations with `mutate_rows` (up to 500 rows per call), and closes the client in a `finally` block. At 120 rows, collecting to the driver would perform the same. `foreachPartition` is used because it is the standard executor-side pattern for writing a DataFrame to an external store, which was part of the learning objective.

## Decision 7 — Service-account keys created outside Terraform

A key created with `google_service_account_key` is stored in plain text in Terraform state, and the state lives in a GCS bucket. Keys are therefore created with `gcloud iam service-accounts keys create`, mounted read-only into the containers, and gitignored. Key rotation never touches state.

## Consequences

- Silver can always rebuild Bigtable, so the most expensive component can be switched off without data loss.
- Adding a feature means adding a qualifier in `spark/config/feature_config.py`; no schema change.
- No SQL. Aggregations happen in Spark before the write.
- Writes are at-least-once; upserts on the same row key make a retried write idempotent.

## References

- `infrastructure/gcp/main.tf` — instance, table, GC policies, IAM
- `spark/utils/bigtable_writer.py` — the writer
- `spark/config/feature_config.py` — families and qualifiers
- `model_serving/pipeline/feature_client.py` — the reader and freshness check
