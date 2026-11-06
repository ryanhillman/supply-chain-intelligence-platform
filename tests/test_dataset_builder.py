"""
DatasetBuilder (ml_training/datasets/dataset_builder.py) from the Spark
training set's rows: defaults and derived features, the two baseline columns,
the purged time split, and the files it writes.
"""
import json

import numpy as np
import pandas as pd

from feature_registry import HORIZON_DAYS, RAW_FEATURE_COLS
from ml_training.datasets.dataset_builder import DatasetBuilder
from ml_training.datasets.feature_engineer import (
    BASELINE_COL,
    NUMERIC_FEATURE_COLS,
    RISK_SCORE_COL,
    TARGET_COL,
    normal_stockout_prob,
)

N_DAYS = 60


def _training_rows(seed: int = 0) -> pd.DataFrame:
    """The training set's shape: one row per pair and day, nullable where Silver is null."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=N_DAYS, freq="D").date
    rows = pd.DataFrame([
        {"warehouse_id": wh, "sku_id": sku, "sample_date": d}
        for wh in ("W-001", "W-002") for sku in ("SKU-001", "SKU-002") for d in dates
    ])
    for col in RAW_FEATURE_COLS:
        rows[col] = np.abs(rng.normal(50, 30, len(rows)))
    rows.loc[rng.random(len(rows)) < 0.3, "active_po_qty"] = np.nan      # nothing open
    rows.loc[rng.random(len(rows)) < 0.3, "demand_signal_count_24h"] = np.nan
    rows[f"stockout_label_{HORIZON_DAYS}d"] = (rng.random(len(rows)) < 0.3).astype("int64")
    return rows


def test_build_from_rows_writes_the_splits_and_metadata(tmp_path):
    manifest = DatasetBuilder(bucket="unused", output_dir=tmp_path).build_from_rows(_training_rows())
    train, val, test = (pd.read_parquet(tmp_path / f"{s}.parquet") for s in ("train", "val", "test"))
    metadata = json.loads((tmp_path / "feature_metadata.json").read_text())

    assert len(train) + len(val) + len(test) == manifest.n_train + manifest.n_val + manifest.n_test
    assert not train[NUMERIC_FEATURE_COLS].isna().any().any()           # defaults filled
    assert metadata["horizon_days"] == HORIZON_DAYS
    assert metadata["n_numeric_features"] == len(NUMERIC_FEATURE_COLS) == len(metadata["feature_mean"])
    assert set(train[TARGET_COL].unique()) <= {0, 1}

    # The purge gap: HORIZON_DAYS of dates are dropped before each cutoff.
    gap_train_val = (val.sample_date.min() - train.sample_date.max()).days
    gap_val_test  = (test.sample_date.min() - val.sample_date.max()).days
    assert gap_train_val == gap_val_test == HORIZON_DAYS + 1


def test_baseline_columns_are_the_rule_and_the_formula(tmp_path):
    DatasetBuilder(bucket="unused", output_dir=tmp_path).build_from_rows(_training_rows())
    df = pd.read_parquet(tmp_path / "train.parquet")

    assert df[BASELINE_COL].tolist() == (df["days_of_supply"] < HORIZON_DAYS).astype(float).tolist()
    np.testing.assert_allclose(
        df[RISK_SCORE_COL],
        normal_stockout_prob(df["current_qty"], df["avg_daily_sales_7d"], df["stddev_daily_7d"], HORIZON_DAYS),
    )


def test_output_does_not_depend_on_the_rows_order(tmp_path):
    rows = _training_rows()
    DatasetBuilder(bucket="unused", output_dir=tmp_path / "a").build_from_rows(rows)
    DatasetBuilder(bucket="unused", output_dir=tmp_path / "b").build_from_rows(rows.sample(frac=1, random_state=1))
    for split in ("train", "val", "test"):
        assert (tmp_path / "a" / f"{split}.parquet").read_bytes() == (tmp_path / "b" / f"{split}.parquet").read_bytes()
