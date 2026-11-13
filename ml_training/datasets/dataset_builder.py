"""
DatasetBuilder — orchestrates the full dataset construction pipeline.

Pipeline:
  1. Read the Spark training set     → one labeled row per (warehouse, SKU, day)
     (spark/jobs/training_set.py: Silver's stored features + stockout_label_7d)
  2. engineer_features()             → defaults filled, derived features added
  3. Baseline columns                → baseline_pred, stockout_risk_score
  4. Time-based 70 / 15 / 15 split  → train / val / test
  5. Fit per-feature mean/scale on train → stored in feature_metadata.json
  6. Build vocabulary mappings       → warehouse and SKU ids, from the full dataset
  7. Compute baseline metrics        → on the test split
  8. Write train/val/test Parquet    → artifacts/dataset/ (raw feature units)
  9. Write feature_metadata.json     → artifacts/dataset/
 10. Write baseline_metrics.json     → artifacts/dataset/

Standardization statistics are fit only on the train split and stored in
feature_metadata.json as feature_mean / feature_scale. The parquet splits
keep raw units; the model standardizes its own inputs (ShortagePredictor's
Standardize layer), so training, evaluation, the ONNX graph, the drift
monitor and the scorer all work in raw units.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import gcsfs
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_registry import HORIZON_DAYS
from ml_training.datasets.feature_engineer import (
    BASELINE_COL,
    CATEGORICAL_FEATURE_COLS,
    ID_COLS,
    NUMERIC_FEATURE_COLS,
    RISK_SCORE_COL,
    TARGET_COL,
    engineer_features,
    normal_stockout_prob,
)
from ml_training.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

TRAINING_PREFIX: str = "training"   # gs://<bucket>/training/, written by spark/jobs/training_set.py
TRAIN_FRAC: float = 0.70
VAL_FRAC:   float = 0.15   # test = 1 - TRAIN_FRAC - VAL_FRAC = 0.15


@dataclass
class DatasetManifest:
    output_dir:            str
    train_path:            str
    val_path:              str
    test_path:             str
    metadata_path:         str
    baseline_metrics_path: str
    n_train:               int
    n_val:                 int
    n_test:                int
    n_features:            int
    positive_rate_train:   float
    positive_rate_test:    float
    date_range_start:      str
    date_range_end:        str
    n_sample_dates:        int
    baseline_test_metrics: dict


class DatasetBuilder:
    """
    Parameters
    ----------
    bucket:
        GCS bucket name (the training set is under gs://<bucket>/training/).
    credentials_path:
        Service account JSON path, or None for ADC.
    output_dir:
        Local directory for all output files.
    """

    def __init__(
        self,
        bucket:           str,
        credentials_path: str | None = None,
        output_dir:       str | Path  = "ml_training/artifacts/dataset",
    ) -> None:
        self.bucket           = bucket
        self.credentials_path = credentials_path
        self.output_dir       = Path(output_dir)

    def build(self) -> DatasetManifest:
        """Run the full pipeline from the training set on GCS and return a DatasetManifest."""
        # ── Step 1: Read the Spark training set ──────────────────────────────
        return self.build_from_rows(read_training_set(self.bucket, self.credentials_path))

    def build_from_rows(self, rows: pd.DataFrame) -> DatasetManifest:
        """
        Steps 2-10: derived features, baseline columns, split, statistics and
        output files, from the training set's rows.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sorted, so the dataset does not depend on the order Spark wrote the rows.
        rows = rows.sort_values(["warehouse_id", "sku_id", "sample_date"]).reset_index(drop=True)

        # ── Step 2: Defaults and derived features ─────────────────────────────
        df = engineer_features(rows)

        # ── Step 3: Baseline columns ──────────────────────────────────────────
        # The 0/1 rule and the closed-form formula the model is measured against
        # (evaluation/metrics.py), each from the row's own features.
        df[BASELINE_COL]   = (df["days_of_supply"] < HORIZON_DAYS).astype(float)
        df[RISK_SCORE_COL] = normal_stockout_prob(
            df["current_qty"].values, df["avg_daily_sales_7d"].values,
            df["stddev_daily_7d"].values, HORIZON_DAYS,
        )

        # Ensure sample_date is comparable as Timestamp
        df["sample_date"] = pd.to_datetime(df["sample_date"])

        # ── Step 4: Time-based split ───────────────────────────────────────────
        sorted_dates = sorted(df["sample_date"].unique())
        n_dates      = len(sorted_dates)
        if n_dates < 3:
            raise ValueError(
                f"Only {n_dates} unique sample dates — need at least 3 for a split. "
                "Replay more days into Bronze."
            )

        train_cutoff = sorted_dates[int(n_dates * TRAIN_FRAC) - 1]
        val_cutoff   = sorted_dates[int(n_dates * (TRAIN_FRAC + VAL_FRAC)) - 1]

        # Purge gap: a sample's label looks HORIZON_DAYS into the future, so a
        # sample dated right at train_cutoff has its outcome determined by
        # events that chronologically fall inside the val period's date range.
        # Dropping the HORIZON_DAYS immediately before each cutoff from the
        # earlier split removes that boundary contamination — standard
        # purging for time-series train/val/test splits with a lookahead
        # label window (see e.g. López de Prado, "Advances in Financial
        # Machine Learning", ch. 7). These purged dates belong to neither
        # split; they're excluded entirely rather than reassigned.
        purge = pd.Timedelta(days=HORIZON_DAYS)

        train = df[df["sample_date"] <= train_cutoff - purge].copy()
        val   = df[
            (df["sample_date"] > train_cutoff)
            & (df["sample_date"] <= val_cutoff - purge)
        ].copy()
        test  = df[df["sample_date"] > val_cutoff].copy()

        if train.empty or val.empty:
            raise ValueError(
                f"Purge gap of {HORIZON_DAYS}d emptied train (n={len(train)}) "
                f"or val (n={len(val)}) — replay more days into Bronze."
            )

        logger.info(
            "Split: train=%d (dates ≤ %s), val=%d (≤ %s), test=%d  [%dd purge gap before "
            "each cutoff]",
            len(train), train_cutoff.date(), len(val), val_cutoff.date(), len(test),
            HORIZON_DAYS,
        )

        # ── Step 5: Fit standardization statistics on train ─────────────────
        # sklearn's StandardScaler is used only to fit: it gives population
        # std (ddof=0) and the zero-variance guard (scale_ = 1.0 where the
        # std is 0). The splits are NOT transformed — they stay in raw units
        # and the model applies (x - mean) / scale itself.
        scaler        = StandardScaler().fit(train[NUMERIC_FEATURE_COLS].values)
        feature_mean  = scaler.mean_.tolist()
        feature_scale = scaler.scale_.tolist()

        # ── Step 6: Build vocabulary mappings ────────────────────────────────
        # Built from the full dataset so all entities appear in the vocabulary,
        # even if some are absent from the train split.
        wh_vocab  = {w: i for i, w in enumerate(sorted(df["warehouse_id"].unique()))}
        sku_vocab = {s: i for i, s in enumerate(sorted(df["sku_id"].unique()))}

        # ── Step 7: Baseline metrics on test split ────────────────────────────
        # The bar is the closed-form Normal-approx formula (RISK_SCORE_COL);
        # the 0/1 rule (BASELINE_COL) is recorded alongside for context.
        y_true_test  = test[TARGET_COL].values.astype(float)
        y_base_test  = test[RISK_SCORE_COL].values.astype(float)
        y_rule_test  = test[BASELINE_COL].values.astype(float)

        if y_true_test.sum() == 0 or y_true_test.sum() == len(y_true_test):
            logger.warning("Test split has only one class — baseline metrics will be degenerate.")
            baseline_test_metrics: dict = {"auc_roc": None, "auprc": None, "brier": None}
            rule_test_metrics:     dict = dict(baseline_test_metrics)
        else:
            baseline_test_metrics = compute_metrics(y_true_test, y_base_test)
            rule_test_metrics     = compute_metrics(y_true_test, y_rule_test)
            logger.info(
                "Baseline (Normal-approx formula) test metrics: AUC-ROC=%.4f  AUPRC=%.4f  Brier=%.4f",
                baseline_test_metrics["auc_roc"],
                baseline_test_metrics["auprc"],
                baseline_test_metrics["brier"],
            )

        # ── Step 8: Write Parquet splits ──────────────────────────────────────
        _cols_to_save = ID_COLS + NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS + [
            TARGET_COL, RISK_SCORE_COL, BASELINE_COL
        ]
        # ID_COLS and CATEGORICAL_FEATURE_COLS share warehouse_id/sku_id; deduplicate.
        cols_to_save = list(dict.fromkeys(c for c in _cols_to_save if c in df.columns))

        train_path = self.output_dir / "train.parquet"
        val_path   = self.output_dir / "val.parquet"
        test_path  = self.output_dir / "test.parquet"

        train[cols_to_save].to_parquet(train_path, index=False)
        val[cols_to_save].to_parquet(val_path,   index=False)
        test[cols_to_save].to_parquet(test_path,  index=False)
        logger.info("Parquet splits written to %s", self.output_dir)

        # ── Step 9: Save feature metadata ────────────────────────────────────
        pos_rate_train = float(train[TARGET_COL].mean())
        metadata = {
            "numeric_features":   NUMERIC_FEATURE_COLS,
            "categorical_features": CATEGORICAL_FEATURE_COLS,
            "target_col":         TARGET_COL,
            "risk_score_col":     RISK_SCORE_COL,
            "baseline_col":       BASELINE_COL,
            "warehouse_vocab":    wh_vocab,
            "sku_vocab":          sku_vocab,
            "n_warehouses":       len(wh_vocab),
            "n_skus":             len(sku_vocab),
            "n_numeric_features": len(NUMERIC_FEATURE_COLS),
            # Train-split standardization statistics, in numeric_features order.
            # The model registers them as fixed buffers.
            "feature_mean":       feature_mean,
            "feature_scale":      feature_scale,
            "horizon_days":       HORIZON_DAYS,
            "split": {
                "train_end": str(train_cutoff.date()),
                "val_end":   str(val_cutoff.date()),
                "train_frac": TRAIN_FRAC,
                "val_frac":   VAL_FRAC,
            },
            "positive_rate_train": pos_rate_train,
            "created_at":          datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = self.output_dir / "feature_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # ── Step 10: Save baseline metrics ───────────────────────────────────
        baseline_path = self.output_dir / "baseline_metrics.json"
        with open(baseline_path, "w") as f:
            json.dump(
                {
                    "test":             baseline_test_metrics,
                    "description":      "Normal-approx P(stockout <= 7d)",
                    "rule_test":        rule_test_metrics,
                    "rule_description": "days_of_supply < 7",
                },
                f, indent=2,
            )

        manifest = DatasetManifest(
            output_dir            = str(self.output_dir),
            train_path            = str(train_path),
            val_path              = str(val_path),
            test_path             = str(test_path),
            metadata_path         = str(metadata_path),
            baseline_metrics_path = str(baseline_path),
            n_train               = len(train),
            n_val                 = len(val),
            n_test                = len(test),
            n_features            = len(NUMERIC_FEATURE_COLS),
            positive_rate_train   = pos_rate_train,
            positive_rate_test    = float(y_true_test.mean()),
            date_range_start      = str(sorted_dates[0].date()),
            date_range_end        = str(sorted_dates[-1].date()),
            n_sample_dates        = n_dates,
            baseline_test_metrics = baseline_test_metrics,
        )
        logger.info("DatasetBuilder complete: %s", asdict(manifest))
        return manifest


def read_training_set(bucket: str, credentials_path: str | None = None) -> pd.DataFrame:
    """The rows spark/jobs/training_set.py wrote to gs://<bucket>/training/."""
    fs = gcsfs.GCSFileSystem(token=credentials_path if credentials_path else "google_default")
    # Explicit file paths: gcsfs lists a directory as its own first entry, which
    # pyarrow's directory walk then tries to open as a file.
    paths = fs.glob(f"{bucket}/{TRAINING_PREFIX}/*.parquet")
    if not paths:
        raise FileNotFoundError(
            f"No training set under gs://{bucket}/{TRAINING_PREFIX}/ — "
            "run spark/jobs/training_set.py first."
        )
    logger.info("Reading the training set: %d file(s) under gs://%s/%s/", len(paths), bucket, TRAINING_PREFIX)
    return pd.concat([pd.read_parquet(p, filesystem=fs) for p in paths], ignore_index=True)
