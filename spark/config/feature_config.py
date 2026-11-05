"""
Silver job configuration: Bigtable write tuning, the row key separator, the
reorder-urgency and days_of_supply constants, and the pipeline version.

The Bigtable column layout and cell encoding are in feature_registry.
"""
from __future__ import annotations

# ── Bigtable write tuning ─────────────────────────────────────────────────────

# Maximum rows per mutate_rows RPC call. Bigtable max is 100,000 but
# 500 keeps individual RPC latency <100ms for the feature store write path.
BIGTABLE_BATCH_SIZE: int = 500

# Bigtable row key separator. Chosen to sort correctly under lexicographic order:
# warehouse IDs (W-001..W-006) sort before the separator naturally.
ROW_KEY_SEPARATOR: str = "#"

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
# tests/test_spark_parity.py checks that the two agree.
DAYS_OF_SUPPLY_EPS: float = 1e-6
DAYS_OF_SUPPLY_CAP: float = 365.0

# ── Pipeline versioning ───────────────────────────────────────────────────────
# Increment when feature computation logic changes.
# Written to lineage#pipeline_version on every Bigtable row.
PIPELINE_VERSION: str = "1.2.0"
