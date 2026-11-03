"""
Silver through Bigtable: what the scorer reads after two daily Silver runs.

The first day leaves open orders, shipments or demand signals in the store; on
the second day some of them are gone, so Silver holds nulls there. The scorer
must see the second day's values, the same ones the training sampler computes,
not the first day's cells.

Needs PySpark, Java 17 and a Bigtable emulator (BIGTABLE_EMULATOR_HOST). Marked
slow: CI runs it in its own job (.github/workflows/ci.yml).
"""
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytestmark = pytest.mark.slow

_SPARK_DIR = str(Path(__file__).parents[1] / "spark")
sys.path.insert(0, _SPARK_DIR)

from pyspark.sql import SparkSession  # noqa: E402

from config.feature_config import FEATURE_REGISTRY  # noqa: E402
from jobs import silver_batch  # noqa: E402
from jobs.bronze_ingestion import TOPIC_SCHEMA_MAP, prepare_bronze_df  # noqa: E402
from jobs.schemas.feature_schema import conform_to_silver_schema  # noqa: E402
from jobs.silver_features import compute_features  # noqa: E402
from ml_training.datasets.bronze_temporal_sampler import BronzeTemporalSampler  # noqa: E402
from ml_training.datasets.feature_engineer import fill_defaults  # noqa: E402
from model_serving.pipeline.feature_client import FeatureClient  # noqa: E402
from model_serving.pipeline.request_builder import _RAW_FEATURES  # noqa: E402
from utils.bigtable_writer import write_to_bigtable  # noqa: E402
from world_fixture import (  # noqa: E402
    BRONZE_PARTITIONS,
    DAYS,
    START,
    events_by_topic,
    world_events,
)

_PROJECT, _INSTANCE, _TABLE = "test-project", "test-instance", "inventory_features"
# Silver leaves these null when a pair has nothing open or no signal that day.
_ABSENT_MEANS_ZERO = ["active_po_qty", "in_transit_qty", "demand_signal_count_24h"]


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]").appName("scip-bigtable-test")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        # Spark's Python workers import utils.bigtable_writer by name.
        .config("spark.executorEnv.PYTHONPATH", _SPARK_DIR)
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def bronze(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("bronze")
    for topic, payloads in events_by_topic(world_events()).items():
        df = prepare_bronze_df(spark.createDataFrame(payloads, schema=TOPIC_SCHEMA_MAP[topic]), topic)
        df.write.mode("append").partitionBy(*BRONZE_PARTITIONS).parquet(str(root / topic))
    return root


@pytest.fixture()
def table():
    if not os.environ.get("BIGTABLE_EMULATOR_HOST"):
        pytest.skip("BIGTABLE_EMULATOR_HOST is not set")
    from google.cloud import bigtable
    from google.cloud.bigtable import column_family

    t = bigtable.Client(project=_PROJECT, admin=True).instance(_INSTANCE).table(_TABLE)
    if t.exists():
        t.delete()
    t.create(column_families={fam: column_family.MaxVersionsGCRule(1) for fam in FEATURE_REGISTRY})
    yield t
    t.delete()


def _silver_to_bigtable(spark, bronze, as_of: date) -> None:
    features, event_count = compute_features(spark, str(bronze), as_of)
    df = conform_to_silver_schema(silver_batch.add_row_key_and_partitions(
        silver_batch.add_lineage(features, 1, event_count), as_of
    ))
    write_to_bigtable(df, _PROJECT, _INSTANCE, _TABLE, epoch_id=1)


def _days_where_values_disappear(rows: pd.DataFrame) -> tuple[date, date]:
    """Two consecutive sample dates where a value present on the first is gone on the second."""
    wide = rows.set_index(["sample_date", "warehouse_id", "sku_id"]).sort_index()
    dates = sorted(rows.sample_date.unique())
    best, best_columns = None, 0
    for first, second in zip(dates, dates[1:], strict=False):
        a, b = wide.loc[first], wide.loc[second]
        columns = sum(bool(((a[c] > 0) & (b[c] == 0)).any()) for c in _ABSENT_MEANS_ZERO)
        if columns > best_columns:
            best, best_columns = (first, second), columns
    assert best is not None, "the fixture world has no value that disappears"
    return best


def test_scorer_reads_the_second_days_values_after_two_silver_runs(spark, bronze, table):
    rows, _ = BronzeTemporalSampler(bronze_path=bronze).build()
    assert rows.sample_date.max() <= START + timedelta(days=DAYS - 1)
    first, second = _days_where_values_disappear(rows)

    _silver_to_bigtable(spark, bronze, first)
    _silver_to_bigtable(spark, bronze, second)

    read_back = FeatureClient(project_id=_PROJECT, instance_id=_INSTANCE, table_id=_TABLE).scan_all()
    actual = pd.DataFrame([
        {"warehouse_id": r.warehouse_id, "sku_id": r.sku_id, **r.features} for r in read_back
    ]).reindex(columns=["warehouse_id", "sku_id", *_RAW_FEATURES])
    fill_defaults(actual)      # what RequestBuilder does with absent cells
    actual = actual.set_index(["warehouse_id", "sku_id"]).sort_index()

    expected = rows[rows.sample_date == second].set_index(["warehouse_id", "sku_id"]).sort_index()
    assert list(actual.index) == list(expected.index)
    for col in _RAW_FEATURES:
        np.testing.assert_allclose(
            actual[col].astype(float).to_numpy(), expected[col].astype(float).to_numpy(),
            rtol=1e-9, atol=1e-9, err_msg=f"{col} after Silver runs for {first} and {second}",
        )


def test_silver_batch_exits_non_zero_when_the_bigtable_write_fails(monkeypatch):
    for name in ("create_spark_session", "add_lineage", "add_row_key_and_partitions",
                 "conform_to_silver_schema", "write_silver"):
        monkeypatch.setattr(silver_batch, name, mock.MagicMock())
    monkeypatch.setattr(silver_batch, "compute_features", lambda *_: (mock.MagicMock(), 1))
    monkeypatch.setattr(silver_batch, "write_to_bigtable",
                        mock.MagicMock(side_effect=RuntimeError("Bigtable write failed")))
    monkeypatch.setattr(sys, "argv", ["silver_batch.py", "--as-of", "2026-01-01"])

    with pytest.raises(RuntimeError, match="Bigtable write failed"):
        silver_batch.main()
    silver_batch.write_silver.assert_called_once()   # Silver is written first and stays
