"""
The Spark training set (spark/jobs/training_set.py): Silver's stored features
joined with the stockout label.

The label is checked against a value computed directly from the fixture
world's inventory events, the rows against Silver's own features, and the
label window's edges on single events.

Needs PySpark and Java 17. Marked slow: CI runs it in its own job
(.github/workflows/ci.yml).
"""
from datetime import date, timedelta

import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytestmark = pytest.mark.slow

from pyspark.sql import SparkSession  # noqa: E402

from feature_registry import FEATURE_REGISTRY, HORIZON_DAYS, RAW_FEATURE_COLS  # noqa: E402
from jobs import silver_batch  # noqa: E402
from jobs.bronze_ingestion import TOPIC_SCHEMA_MAP, prepare_bronze_df  # noqa: E402
from jobs.schemas.feature_schema import SILVER_PARTITION_COLS, conform_to_silver_schema  # noqa: E402
from jobs.silver_features import compute_features  # noqa: E402
from jobs.training_set import LABEL_COL, compute_labels, training_rows  # noqa: E402
from world_fixture import (  # noqa: E402
    BRONZE_PARTITIONS,
    DAYS,
    START,
    event_date,
    events_by_topic,
    world_events,
)

FIRST, LAST = START + timedelta(days=30), START + timedelta(days=DAYS - 1 - HORIZON_DAYS)
ROW_KEY = ["sample_date", "warehouse_id", "sku_id"]
_SILVER_TO_RAW = {f"{family}__{attr}": attr for family, attrs in FEATURE_REGISTRY.items() for attr in attrs}


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


def _frame(df) -> pd.DataFrame:
    return pd.DataFrame([r.asDict() for r in df.collect()])


@pytest.fixture(scope="module")
def bronze(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("bronze")
    for topic, payloads in events_by_topic(world_events()).items():
        _write_bronze(spark, root, topic, payloads)
    return root


@pytest.fixture(scope="module")
def expected_labels():
    """{(sample_date, warehouse_id, sku_id): label} from the fixture's inventory events."""
    inventory = events_by_topic(world_events())["inventory.events"]
    labels = {}
    for d in (FIRST + timedelta(days=i) for i in range((LAST - FIRST).days + 1)):
        for p in inventory:
            if d < event_date(p) <= d + timedelta(days=HORIZON_DAYS):
                key = (d, p["warehouse_id"], p["sku_id"])
                labels[key] = max(labels.get(key, 0), int(p["quantity_after"] == 0))
    return labels


def test_label_is_a_stockout_in_the_following_seven_days(spark, bronze, expected_labels):
    labels = _frame(compute_labels(spark, str(bronze), FIRST, LAST)).set_index(ROW_KEY)[LABEL_COL]
    assert labels.to_dict() == expected_labels
    assert set(expected_labels.values()) == {0, 1}, "the fixture must hold both outcomes"


def test_training_rows_are_silver_features_with_the_label(spark, bronze, expected_labels, tmp_path):
    silver = tmp_path / "silver"
    features = compute_features(spark, str(bronze), FIRST, LAST)
    conform_to_silver_schema(silver_batch.add_row_key_and_partitions(silver_batch.add_lineage(features))) \
        .write.partitionBy(*SILVER_PARTITION_COLS).parquet(str(silver))

    rows = _frame(training_rows(spark, str(silver), str(bronze), FIRST, LAST))
    assert list(rows.columns) == ["warehouse_id", "sku_id", "sample_date", *RAW_FEATURE_COLS, LABEL_COL]
    actual = rows.set_index(ROW_KEY).sort_index()

    stored = (_frame(features).rename(columns={**_SILVER_TO_RAW, "as_of": "sample_date"})
              .set_index(ROW_KEY).sort_index())
    cols = list(RAW_FEATURE_COLS)
    # Silver stores every feature as a double; the direct computation keeps Spark's integer sums.
    pd.testing.assert_frame_equal(actual[cols].astype(float), stored[cols].astype(float))   # nulls stay null
    assert actual[LABEL_COL].to_dict() == expected_labels


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
