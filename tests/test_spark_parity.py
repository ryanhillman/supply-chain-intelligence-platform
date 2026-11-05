"""
Silver (spark/jobs/silver_features.py) and the training sampler
(ml_training/datasets/bronze_temporal_sampler.py) compute the same features
from the same Bronze data.

This is the train/serve consistency check: the model is trained on sampler
rows and scored on Silver rows. The fixture world (tests/world_fixture.py) is
written to disk in Bronze's layout by bronze_ingestion.prepare_bronze_df, twice,
the way a repeated replay lands. Both sides must drop the duplicates, and for
several as-of dates every raw feature must match after the scorer's defaults
(feature_engineer.fill_defaults) are applied to Silver's nulls.

Needs PySpark and Java 17. Marked slow: CI runs it in its own job
(.github/workflows/ci.yml); `pytest -m slow tests/test_spark_parity.py` runs it
locally.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytestmark = pytest.mark.slow

from pyspark.sql import SparkSession  # noqa: E402

from feature_registry import FEATURE_REGISTRY  # noqa: E402
from jobs.bronze_ingestion import TOPIC_SCHEMA_MAP, prepare_bronze_df  # noqa: E402
from jobs.silver_features import _risk, compute_features  # noqa: E402
from ml_training.datasets.bronze_temporal_sampler import BronzeTemporalSampler  # noqa: E402
from ml_training.datasets.feature_engineer import engineer_features, fill_defaults  # noqa: E402
from model_serving.pipeline.request_builder import _RAW_FEATURES  # noqa: E402
from world_fixture import (  # noqa: E402
    BRONZE_PARTITIONS,
    DAYS,
    START,
    events_by_topic,
    world_events,
)

# Silver column (family__attribute) -> the raw feature name the sampler and scorer use.
_SILVER_TO_RAW = {f"{family}__{attr}": attr for family, attrs in FEATURE_REGISTRY.items() for attr in attrs}


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]").appName("scip-parity-test")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def bronze(spark, tmp_path_factory):
    root = tmp_path_factory.mktemp("bronze")
    for topic, payloads in events_by_topic(world_events()).items():
        df = prepare_bronze_df(spark.createDataFrame(payloads, schema=TOPIC_SCHEMA_MAP[topic]), topic)
        for _ in range(2):   # a repeated replay
            df.write.mode("append").partitionBy(*BRONZE_PARTITIONS).parquet(str(root / topic))
    return root


def _silver_raw_features(spark, bronze, as_of) -> pd.DataFrame:
    features, _ = compute_features(spark, str(bronze), as_of)
    df = pd.DataFrame([r.asDict() for r in features.collect()]).rename(columns=_SILVER_TO_RAW)
    fill_defaults(df)          # what the scorer does with Silver's nulls
    return df.set_index(["warehouse_id", "sku_id"]).sort_index()


def test_silver_matches_the_sampler_on_every_raw_feature(spark, bronze):
    sampler_rows, _ = BronzeTemporalSampler(bronze_path=bronze).build()
    last_sample_date = START + timedelta(days=DAYS - 1 - 7)

    for as_of in (START + timedelta(days=30), START + timedelta(days=34), last_sample_date):
        expected = (sampler_rows[sampler_rows.sample_date == as_of]
                    .set_index(["warehouse_id", "sku_id"]).sort_index())
        actual = _silver_raw_features(spark, bronze, as_of)

        assert list(actual.index) == list(expected.index), as_of
        for col in _RAW_FEATURES:
            np.testing.assert_allclose(
                actual[col].astype(float).to_numpy(), expected[col].astype(float).to_numpy(),
                rtol=1e-9, atol=1e-9, err_msg=f"{col} on {as_of}",
            )


def test_silver_days_of_supply_is_the_models_formula(spark):
    # current_qty, avg_daily_sales_7d: in range; out of stock with no sales;
    # in stock with no sales; no sales recorded at all; over the cap.
    cases = [(50.0, 10.0), (0.0, 0.0), (50.0, 0.0), (50.0, None), (5000.0, 1.0)]
    silver = _risk(spark.createDataFrame(
        [(i, qty, avg) for i, (qty, avg) in enumerate(cases)],
        "i int, inventory__current_qty double, velocity__avg_daily_sales_7d double",
    )).orderBy("i").collect()

    raw = pd.DataFrame(cases, columns=["current_qty", "avg_daily_sales_7d"])
    model = engineer_features(raw.reindex(columns=list(_RAW_FEATURES)))["days_of_supply"].tolist()

    assert [r["inventory__days_of_supply"] for r in silver] == model == [5.0, 0.0, 365.0, 365.0, 365.0]
    assert [r["risk__reorder_urgency"] for r in silver] == [2, 3, 0, 0, 0]
