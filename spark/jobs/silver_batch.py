"""
Silver Batch — one-shot batch validation for Phase 3.

Reads existing Bronze GCS Parquet in batch mode (spark.read, not readStream),
computes the full 30-feature vector for every active (warehouse_id, sku_id)
pair, writes Silver Parquet to GCS, and dual-writes to Bigtable.

This is NOT the production job. Production uses streaming mode via
silver_feature_engineering.py. This script exists specifically for local
Phase 3 validation under Docker Desktop memory constraints where running
Bronze and Silver streaming jobs simultaneously is not feasible.

Usage (spark-submit inside the scip-spark container):
    spark-submit --master local[2] \\
        --conf spark.driver.memory=384m \\
        jobs/silver_batch.py

Key differences from silver_feature_engineering.py:
    - Uses spark.read (batch) instead of readStream / foreachBatch
    - No checkpoint state — safe to re-run; appends to Silver
    - Processes ALL Bronze data in one pass (not micro-batch)
    - Exits cleanly after one write cycle
    - epoch_id=9000 in lineage to distinguish from streaming epochs
"""
from __future__ import annotations

import logging
import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

sys.path.insert(0, "/opt/spark-app")

from config.bronze_config import (
    BIGTABLE_INSTANCE_ID,
    BIGTABLE_TABLE_ID,
    BRONZE_BASE_PATH,
    GCP_PROJECT_ID,
)
from jobs.schemas.inventory_schema import INVENTORY_EVENT_SCHEMA
from jobs.silver_feature_engineering import (
    SILVER_BASE_PATH,
    _add_lineage,
    _add_partition_and_row_key,
    _compute_demand_features,
    _compute_inventory_snapshot,
    _compute_supplier_features,
    _compute_velocity,
    _derive_risk_features,
    _read_bronze,
    write_silver,
)
from utils.bigtable_writer import write_to_bigtable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("silver_batch")

BATCH_EPOCH_ID: int = 9000


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("scip-silver-batch-validation")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.shuffle.partitions", "6")
        .config("spark.hadoop.fs.gs.impl",
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl",
                "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        .getOrCreate()
    )


def main() -> None:
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info(
        "Silver batch starting | bronze=%s silver=%s",
        BRONZE_BASE_PATH, SILVER_BASE_PATH,
    )

    # ── Step 1: Read ALL Bronze inventory events (batch, not stream) ──────────
    inventory_path = f"{BRONZE_BASE_PATH}/inventory.events"
    inventory_all = (
        spark.read
        .schema(INVENTORY_EVENT_SCHEMA)
        .parquet(inventory_path)
    )

    event_count = inventory_all.count()
    logger.info("Bronze inventory events read | count=%d", event_count)

    if event_count == 0:
        logger.error(
            "No Bronze inventory events at %s — has the Bronze job run?",
            inventory_path,
        )
        spark.stop()
        sys.exit(1)

    # ── Step 2: Derive active (warehouse_id, sku_id) pairs ───────────────────
    hot_pairs = inventory_all.select("warehouse_id", "sku_id").distinct()
    pair_count = hot_pairs.count()
    logger.info("Active (warehouse, sku) pairs | count=%d", pair_count)

    # ── Step 3: Batch-read Bronze sources (same logic as streaming job) ───────
    inventory_30d = (
        _read_bronze(spark, "inventory.events", INVENTORY_EVENT_SCHEMA, days_back=30)
        .join(hot_pairs, ["warehouse_id", "sku_id"], "inner")
    )

    # ── Step 4: Compute feature groups ───────────────────────────────────────
    snapshot_df  = _compute_inventory_snapshot(inventory_30d)
    velocity_df  = _compute_velocity(inventory_30d)
    demand_df    = _compute_demand_features(spark, hot_pairs)
    supplier_df  = _compute_supplier_features(spark, hot_pairs)

    # ── Step 5: Join all feature groups ──────────────────────────────────────
    feature_df = (
        snapshot_df
        .join(velocity_df,  ["warehouse_id", "sku_id"], "left")
        .join(demand_df,    ["warehouse_id", "sku_id"], "left")
        .join(supplier_df,  ["warehouse_id", "sku_id"], "left")
    )

    # ── Step 6: Risk features ─────────────────────────────────────────────────
    feature_df = _derive_risk_features(feature_df)

    # ── Step 7: Reserved demand forecast columns ─────────────────────────────
    # These columns are reserved for future demand forecasting integration.
    # No current pipeline stage populates them.
    feature_df = (
        feature_df
        .withColumn("demand__demand_forecast_7d",  F.lit(None).cast("double"))
        .withColumn("demand__demand_forecast_30d", F.lit(None).cast("double"))
        .withColumn("demand__demand_multiplier",   F.lit(None).cast("double"))
    )

    # ── Step 8: Lineage + partition columns + Bigtable row key ───────────────
    feature_df = _add_lineage(feature_df, BATCH_EPOCH_ID, event_count)
    feature_df = _add_partition_and_row_key(feature_df)

    # Cache before dual-write to avoid recomputing the DAG twice
    feature_df = feature_df.cache()

    try:
        # ── Step 9: Write Silver GCS (primary durable store) ─────────────────
        write_silver(feature_df, BATCH_EPOCH_ID)

        # ── Step 10: Dual-write Bigtable (online feature store) ───────────────
        try:
            write_to_bigtable(
                df=feature_df,
                project_id=GCP_PROJECT_ID,
                instance_id=BIGTABLE_INSTANCE_ID,
                table_id=BIGTABLE_TABLE_ID,
                epoch_id=BATCH_EPOCH_ID,
            )
        except Exception as bt_exc:
            logger.error(
                "Bigtable write FAILED | error=%s | Silver succeeded — "
                "Bigtable can be rebuilt from Silver.",
                bt_exc,
            )
    finally:
        feature_df.unpersist()
        spark.stop()

    logger.info("Silver batch complete.")


if __name__ == "__main__":
    main()
