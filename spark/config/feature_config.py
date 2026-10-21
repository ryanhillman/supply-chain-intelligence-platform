"""
Feature engineering configuration for the Silver layer and Bigtable feature store.

Defines:
  - Bigtable column family → qualifier mappings (29 features across 6 families)
  - Write tuning parameters
  - Feature derivation constants
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
# Used by the risk feature derivations in silver_feature_engineering.py.
# Reorder urgency levels written to risk#reorder_urgency:
#   0 = OK        days_of_supply > REORDER_WATCH_DAYS
#   1 = WATCH     days_of_supply in (REORDER_CRITICAL_DAYS, REORDER_WATCH_DAYS]
#   2 = REORDER   days_of_supply in (0, REORDER_CRITICAL_DAYS]
#   3 = CRITICAL  qty == 0 or days_until_stockout <= 1
REORDER_WATCH_DAYS: int = 14
REORDER_CRITICAL_DAYS: int = 7
STOCKOUT_CRITICAL_DAYS: int = 1

# ── Pipeline versioning ───────────────────────────────────────────────────────
# Increment when feature computation logic changes.
# Written to lineage#pipeline_version on every Bigtable row.
PIPELINE_VERSION: str = "1.0.0"

# ── Bigtable column family → qualifier registry ───────────────────────────────
# Single source of truth for all 29 features.
# Structure: {family: {python_attr: bigtable_qualifier}}
#
# python_attr  — name used in PySpark DataFrame / Silver Parquet
# bigtable_qualifier — column qualifier in Bigtable (short, storage-efficient)
#
# Value encoding: all numeric features stored as big-endian double (8 bytes)
# via struct.pack(">d", float(value)). String features stored as UTF-8 bytes.

FEATURE_REGISTRY: dict[str, dict[str, str]] = {
    "inventory": {
        "current_qty":       "cur_qty",
        "reorder_point":     "reorder_pt",
        "safety_stock":      "safety_stk",
        "days_of_supply":    "dos",
        "last_event_type":   "last_evt",    # string
        "last_updated":      "last_upd",    # string (ISO-8601)
    },
    "velocity": {
        "sales_7d":             "sales_7d",
        "sales_30d":            "sales_30d",
        "avg_daily_sales_7d":   "avg_d_7d",
        "avg_daily_sales_30d":  "avg_d_30d",
        "velocity_trend":       "v_trend",   # ratio: avg_d_7d / avg_d_30d
        "stddev_daily_7d":      "stddev_7d",
    },
    "demand": {
        "demand_forecast_7d":       "fcst_7d",
        "demand_forecast_30d":      "fcst_30d",
        "demand_signal_count_24h":  "sig_24h",
        "last_signal_type":         "last_sig",  # string
        "demand_multiplier":        "d_mult",    # fcst_7d / (avg_d_7d * 7)
    },
    "supplier": {
        "lead_time_days":           "lt_days",
        "lead_time_stddev":         "lt_std",
        "supplier_reliability":     "reliability",  # 0.0–1.0
        "active_po_qty":            "po_qty",
        "in_transit_qty":           "transit_qty",
    },
    "risk": {
        "stockout_risk_score":      "risk_score",   # 0.0–1.0, ML output (Phase 4)
        "shortage_prob_7d":         "short_p7d",    # 0.0–1.0
        "reorder_urgency":          "urgency",      # 0|1|2|3
    },
    "lineage": {
        "last_bronze_epoch":    "bronze_epoch",   # int
        "last_silver_write":    "silver_ts",      # string ISO-8601
        "source_event_count":   "evt_count",
        "pipeline_version":     "pipeline_ver",   # string
    },
}

# ── Derived flat lists (convenience for Spark select / Bigtable iteration) ────

# All feature column names in Silver DataFrame (family__qualifier naming to avoid clashes)
SILVER_FEATURE_COLUMNS: list[str] = [
    f"{family}__{attr}"
    for family, qualifiers in FEATURE_REGISTRY.items()
    for attr in qualifiers
]

# String-typed features (stored as UTF-8 in Bigtable, not double)
STRING_FEATURES: frozenset[str] = frozenset({
    "inventory__last_event_type",
    "inventory__last_updated",
    "demand__last_signal_type",
    "lineage__last_silver_write",
    "lineage__pipeline_version",
})

# Features to skip writing to Bigtable when value is None/null
# (avoids writing empty cells that would still consume storage)
NULLABLE_FEATURES: frozenset[str] = frozenset({
    "demand__demand_forecast_7d",
    "demand__demand_forecast_30d",
    "demand__demand_multiplier",
    "risk__stockout_risk_score",
    "risk__shortage_prob_7d",
    "supplier__lead_time_stddev",
})
