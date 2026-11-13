"""
Silver job configuration: Bigtable write tuning, the reorder-urgency and
days_of_supply constants, the training set's history requirement, and the
pipeline version.

The Bigtable column layout, row key separator and cell encoding are in
feature_registry.
"""
from __future__ import annotations

# ── Bigtable write tuning ─────────────────────────────────────────────────────

# Maximum rows per mutate_rows RPC call. Bigtable max is 100,000 but
# 500 keeps individual RPC latency <100ms for the feature store write path.
BIGTABLE_BATCH_SIZE: int = 500

# ── Stockout risk thresholds ──────────────────────────────────────────────────
# Used by the risk feature derivations in spark/jobs/silver_features.py.
# Reorder urgency levels written to risk#reorder_urgency:
#   0 = OK        days_of_supply > REORDER_WATCH_DAYS
#   1 = WATCH     days_of_supply in (REORDER_CRITICAL_DAYS, REORDER_WATCH_DAYS]
#   2 = REORDER   days_of_supply in (0, REORDER_CRITICAL_DAYS]
#   3 = CRITICAL  qty == 0 or days_until_stockout <= 1
REORDER_WATCH_DAYS: int = 14
REORDER_CRITICAL_DAYS: int = 7
STOCKOUT_CRITICAL_DAYS: int = 1

# days_of_supply = min(current_qty / max(avg_daily_sales_7d, EPS), CAP), the
# formula ml_training/datasets/feature_engineer.py uses for the model's input.
# tests/test_silver_features.py checks that the two agree.
DAYS_OF_SUPPLY_EPS: float = 1e-6
DAYS_OF_SUPPLY_CAP: float = 365.0

# ── Training set ──────────────────────────────────────────────────────────────
# A sample date needs this many days of Bronze before it, so that the 30-day
# feature windows are full (spark/jobs/training_set.py).
MIN_HISTORY_DAYS: int = 30

# ── Pipeline versioning ───────────────────────────────────────────────────────
# Increment when feature computation logic changes.
# Written to lineage#pipeline_version on every Bigtable row.
PIPELINE_VERSION: str = "1.2.0"
