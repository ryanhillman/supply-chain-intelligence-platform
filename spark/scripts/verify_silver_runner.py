"""
Silver GCS validation script.
Invoked by verify_silver.ps1 inside the scip-spark Docker container.
Reads Silver Parquet and reports row counts, schema conformance, null rates.
"""
import sys
sys.path.insert(0, "/opt/spark-app")

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

SILVER_PATH = sys.argv[1] if len(sys.argv) > 1 else None
if not SILVER_PATH:
    print("[FAIL] Usage: verify_silver_runner.py <gs://bucket/silver>")
    sys.exit(1)

spark = (
    SparkSession.builder
    .appName("scip-verify-silver")
    .config("spark.hadoop.fs.gs.impl",
            "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
    .config("spark.hadoop.fs.AbstractFileSystem.gs.impl",
            "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

failures = []

def check(condition: bool, pass_msg: str, fail_msg: str) -> None:
    if condition:
        print(f"  [PASS] {pass_msg}")
    else:
        print(f"  [FAIL] {fail_msg}")
        failures.append(fail_msg)

print(f"\n{'='*55}")
print(f"  Silver Validation: {SILVER_PATH}")
print(f"{'='*55}")

# ── Row count ─────────────────────────────────────────────────────────────────
print("\n-- Row count --")
df = spark.read.parquet(SILVER_PATH)
total = df.count()
check(total > 0, f"Total rows: {total}", f"Silver has 0 rows — job may not have run")

# ── Schema check ──────────────────────────────────────────────────────────────
print("\n-- Schema --")
expected_cols = [
    "warehouse_id", "sku_id", "row_key", "window_end",
    # inventory
    "inventory__current_qty", "inventory__reorder_point",
    "inventory__safety_stock", "inventory__days_of_supply",
    "inventory__last_event_type", "inventory__last_updated",
    # velocity
    "velocity__sales_7d", "velocity__sales_30d",
    "velocity__avg_daily_sales_7d", "velocity__avg_daily_sales_30d",
    "velocity__velocity_trend", "velocity__stddev_daily_7d",
    # demand
    "demand__demand_signal_count_24h", "demand__last_signal_type",
    # supplier
    "supplier__lead_time_days", "supplier__supplier_reliability",
    "supplier__active_po_qty", "supplier__in_transit_qty",
    # risk
    "risk__reorder_urgency",
    # lineage
    "lineage__last_bronze_epoch", "lineage__last_silver_write",
    "lineage__source_event_count", "lineage__pipeline_version",
    # partitions
    "year", "month", "day",
]
actual = set(df.columns)
missing = [c for c in expected_cols if c not in actual]
check(len(missing) == 0,
      f"All {len(expected_cols)} expected columns present",
      f"Missing columns: {missing}")
print(f"  [INFO] Total columns in Silver: {len(df.columns)}")

# ── Non-nullable columns ──────────────────────────────────────────────────────
print("\n-- Null rates (non-nullable columns) --")
non_null_cols = [
    "warehouse_id", "sku_id", "row_key",
    "risk__reorder_urgency",
    "lineage__pipeline_version", "lineage__last_bronze_epoch",
    "lineage__source_event_count",
]
for col in non_null_cols:
    if col not in actual:
        continue
    nulls = df.filter(F.col(col).isNull()).count()
    check(nulls == 0,
          f"{col}: 0 nulls",
          f"{col}: {nulls} unexpected null(s)")

# ── Per-warehouse breakdown ───────────────────────────────────────────────────
print("\n-- Row counts by warehouse --")
df.groupBy("warehouse_id").count().orderBy("warehouse_id").show(truncate=False)

# ── Reorder urgency distribution ──────────────────────────────────────────────
print("-- Reorder urgency distribution (0=OK 1=WATCH 2=REORDER 3=CRITICAL) --")
df.groupBy("risk__reorder_urgency").count().orderBy("risk__reorder_urgency").show()

# ── Inventory stats ───────────────────────────────────────────────────────────
print("-- Inventory quantity stats --")
df.select(
    F.min("inventory__current_qty").alias("min_qty"),
    F.avg("inventory__current_qty").alias("avg_qty"),
    F.max("inventory__current_qty").alias("max_qty"),
    F.sum(F.when(F.col("inventory__current_qty") == 0, 1).otherwise(0)).alias("zero_stock_count"),
).show()

# ── Velocity sanity check ─────────────────────────────────────────────────────
print("-- Velocity (top 5 fastest-moving SKUs) --")
if "velocity__sales_7d" in actual:
    df.select("warehouse_id", "sku_id", "velocity__sales_7d", "velocity__velocity_trend") \
      .orderBy(F.col("velocity__sales_7d").desc()) \
      .show(5, truncate=False)

# ── Sample row ────────────────────────────────────────────────────────────────
print("-- Sample feature row --")
df.select(
    "warehouse_id", "sku_id",
    "inventory__current_qty",
    "velocity__sales_7d",
    "risk__reorder_urgency",
    "supplier__lead_time_days",
    "lineage__pipeline_version",
).show(3, truncate=False)

# ── Result ────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
if failures:
    print(f"  FAILED — {len(failures)} check(s):")
    for f in failures:
        print(f"    • {f}")
    spark.stop()
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED")
    print(f"{'='*55}\n")
    spark.stop()
    sys.exit(0)
