"""
The Spark training set (spark/jobs/training_set.py): Silver's stored features
joined with the stockout label.

Checked against the training code's own rows: the features must equal the
sampler's (ml_training/datasets/bronze_temporal_sampler.py) and the label must
equal the label generator's (ml_training/datasets/label_generator.py) on every
sample date of the fixture world. The label window's edges are checked on
single events.

Needs PySpark and Java 17. Marked slow: CI runs it in its own job
(.github/workflows/ci.yml).
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytestmark = pytest.mark.slow

from pyspark.sql import SparkSession  # noqa: E402

from feature_registry import RAW_FEATURE_COLS  # noqa: E402
from jobs import silver_batch  # noqa: E402
from jobs.bronze_ingestion import TOPIC_SCHEMA_MAP, prepare_bronze_df  # noqa: E402
from jobs.schemas.feature_schema import SILVER_PARTITION_COLS, conform_to_silver_schema  # noqa: E402
from jobs.silver_features import compute_features  # noqa: E402
from jobs.training_set import LABEL_COL, compute_labels, training_rows  # noqa: E402
from ml_training.datasets.bronze_temporal_sampler import BronzeTemporalSampler  # noqa: E402
from ml_training.datasets.feature_engineer import fill_defaults  # noqa: E402
from ml_training.datasets.label_generator import generate_labels  # noqa: E402
from world_fixture import (  # noqa: E402
    BRONZE_PARTITIONS,
    DAYS,
    START,
    events_by_topic,
    world_events,
)

FIRST, LAST = START + timedelta(days=30), START + timedelta(days=DAYS - 1 - 7)
ROW_KEY = ["sample_date", "warehouse_id", "sku_id"]


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]").appName("scip-training-set-test")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def _write_bronze(spark, root, topic: str, payloads: list[dict]) -> None:
    df = prepare_bronze_df(spark.createDataFrame(payloads, schema=TOPIC_SCHEMA_MAP[topic]), topic)
    df.write.mode("append").partitionBy(*BRONZE_PARTITIONS).parquet(str(root / topic))


@pytest.fixture(scope="module")
def bronze(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("bronze")
    for topic, payloads in events_by_topic(world_events()).items():
        _write_bronze(spark, root, topic, payloads)
    return root


@pytest.fixture(scope="module")
def expected(bronze):
    """The training code's rows: sampler features with the label generator's label."""
    feature_rows, inv_events = BronzeTemporalSampler(bronze_path=bronze).build()
    return generate_labels(feature_rows, inv_events).set_index(ROW_KEY).sort_index()


def test_labels_match_the_label_generator(spark, bronze, expected):
    labels = pd.DataFrame([r.asDict() for r in compute_labels(spark, str(bronze), FIRST, LAST).collect()])
    actual = labels.set_index(ROW_KEY).sort_index()[LABEL_COL]

    assert list(actual.index) == list(expected.index)   # every pair has events in every window
    assert actual.tolist() == expected["stockout_label_7d"].tolist()


def test_training_rows_are_silver_features_with_the_label(spark, bronze, expected, tmp_path):
    silver = tmp_path / "silver"
    conform_to_silver_schema(silver_batch.add_row_key_and_partitions(
        silver_batch.add_lineage(compute_features(spark, str(bronze), FIRST, LAST))
    )).write.partitionBy(*SILVER_PARTITION_COLS).parquet(str(silver))

    rows = pd.DataFrame([r.asDict() for r in training_rows(spark, str(silver), str(bronze), FIRST, LAST).collect()])
    assert list(rows.columns) == [*["warehouse_id", "sku_id", "sample_date"], *RAW_FEATURE_COLS, LABEL_COL]
    fill_defaults(rows)        # what engineer_features does with Silver's nulls
    actual = rows.set_index(ROW_KEY).sort_index()

    assert list(actual.index) == list(expected.index)
    for col in RAW_FEATURE_COLS:
        np.testing.assert_allclose(
            actual[col].astype(float).to_numpy(), expected[col].astype(float).to_numpy(),
            rtol=1e-9, atol=1e-9, err_msg=col,
        )
    assert actual[LABEL_COL].tolist() == expected["stockout_label_7d"].tolist()


# ── The label window: days D+1..D+7 ───────────────────────────────────────────

SAMPLE_DATE = date(2026, 1, 1)


def _label(spark, tmp_path, timestamp: str, quantity_after: int = 0) -> int:
    """The label on SAMPLE_DATE when Bronze holds one inventory event; 0 when it gets no row."""
    _write_bronze(spark, tmp_path, "inventory.events", [{
        "event_id": "e1", "schema_version": "1.0", "event_type": "SALE",
        "warehouse_id": "W-001", "sku_id": "SKU-001",
        "quantity_before": 5, "quantity_after": quantity_after, "quantity_delta": quantity_after - 5,
        "unit_cost": None, "reorder_point": None, "safety_stock": None, "reference_id": None,
        "timestamp": timestamp, "producer_id": None, "correlation_id": None, "metadata": None,
    }])
    rows = compute_labels(spark, str(tmp_path), SAMPLE_DATE, SAMPLE_DATE).collect()
    assert len(rows) <= 1
    return rows[0][LABEL_COL] if rows else 0


def test_stockout_late_on_day_7_is_labeled(spark, tmp_path):
    assert _label(spark, tmp_path, "2026-01-08T18:00:00+00:00") == 1


def test_stockout_at_midnight_starting_day_1_is_labeled(spark, tmp_path):
    assert _label(spark, tmp_path, "2026-01-02T00:00:00+00:00") == 1


def test_stockout_on_day_8_is_not_labeled(spark, tmp_path):
    assert _label(spark, tmp_path, "2026-01-09T01:00:00+00:00") == 0


def test_stockout_on_the_sample_date_is_not_labeled(spark, tmp_path):
    assert _label(spark, tmp_path, "2026-01-01T23:00:00+00:00") == 0


def test_event_that_leaves_stock_is_not_a_stockout(spark, tmp_path):
    assert _label(spark, tmp_path, "2026-01-05T12:00:00+00:00", quantity_after=3) == 0
