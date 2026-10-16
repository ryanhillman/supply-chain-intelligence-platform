"""
DatasetBuilder — orchestrates the full dataset construction pipeline.

Pipeline:
  1. BronzeTemporalSampler.build()  → (feature_rows, inv_events)
  2. generate_labels()               → labeled feature rows
  3. engineer_features()             → derived features added, leakage removed
  4. Time-based 70 / 15 / 15 split  → train / val / test
  5. Fit StandardScaler on train     → saved to preprocessor.pkl
  6. Compute baseline metrics        → saved to baseline_metrics.json
  7. Write train/val/test Parquet    → artifacts/dataset/
  8. Write feature_metadata.json     → artifacts/dataset/

The scaler is fit only on train data and the fitted parameters are stored.
Val and test data are transformed (not re-fit) to prevent data leakage.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_training.datasets.bronze_temporal_sampler import BronzeTemporalSampler
from ml_training.datasets.feature_engineer import (
    BASELINE_COL,
    CATEGORICAL_FEATURE_COLS,
    ID_COLS,
    NUMERIC_FEATURE_COLS,
    RISK_SCORE_COL,
    TARGET_COL,
    engineer_features,
)
from ml_training.datasets.label_generator import generate_labels
from ml_training.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

TRAIN_FRAC: float = 0.70
VAL_FRAC:   float = 0.15   # test = 1 - TRAIN_FRAC - VAL_FRAC = 0.15


@dataclass
class DatasetManifest:
    output_dir:            str
    train_path:            str
    val_path:              str
    test_path:             str
    metadata_path:         str
    preprocessor_path:     str
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
        GCS bucket name.
    credentials_path:
        Service account JSON path, or None for ADC.
    output_dir:
        Local directory for all output files.
    horizon_days:
        Stockout prediction horizon in days (default 7).
    """

    def __init__(
        self,
        bucket:           str,
        credentials_path: str | None = None,
        output_dir:       str | Path  = "ml_training/artifacts/dataset",
        horizon_days:     int         = 7,
        simulate_days:    int         = 0,
    ) -> None:
        self.bucket           = bucket
        self.credentials_path = credentials_path
        self.output_dir       = Path(output_dir)
        self.horizon_days     = horizon_days
        self.simulate_days    = simulate_days

    def build(self) -> DatasetManifest:
        """Run the full pipeline from GCS Bronze and return a DatasetManifest."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Sample Bronze temporal features ───────────────────────────
        sampler = BronzeTemporalSampler(
            bucket=self.bucket,
            credentials_path=self.credentials_path,
            simulate_days=self.simulate_days,
        )
        feature_rows, inv_events = sampler.build()
        return self._build_from_raw(feature_rows, inv_events)

    def build_from_synthetic(self, generator: object) -> DatasetManifest:
        """
        Build the dataset from a SyntheticDataGenerator instead of GCS Bronze.

        The generator must implement .build() returning (feature_rows, inv_events)
        with the same column contract as BronzeTemporalSampler.build().
        All downstream steps (label generation, feature engineering, split,
        StandardScaler, parquet output) are reused unchanged.

        Parameters
        ----------
        generator:
            SyntheticDataGenerator instance (or any object with a compatible
            .build() -> tuple[pd.DataFrame, pd.DataFrame] method).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        feature_rows, inv_events = generator.build()
        return self._build_from_raw(feature_rows, inv_events)

    def _build_from_raw(
        self,
        feature_rows: "pd.DataFrame",
        inv_events:   "pd.DataFrame",
    ) -> DatasetManifest:
        """
        Steps 2–11 of the original build() pipeline, shared by both data paths.

        Accepts (feature_rows, inv_events) in BronzeTemporalSampler format and
        produces the full set of dataset artifacts in self.output_dir.
        """
        # ── Step 2: Generate labels ────────────────────────────────────────────
        labeled = generate_labels(feature_rows, inv_events, self.horizon_days)

        # ── Step 3: Derived features + leakage controls ────────────────────────
        df = engineer_features(labeled)

        # Ensure sample_date is comparable as Timestamp
        df["sample_date"] = pd.to_datetime(df["sample_date"])

        # ── Step 4: Time-based split ───────────────────────────────────────────
        sorted_dates = sorted(df["sample_date"].unique())
        n_dates      = len(sorted_dates)
        if n_dates < 3:
            raise ValueError(
                f"Only {n_dates} unique sample dates — need at least 3 for a split. "
                "Increase Bronze data volume or reduce min_lookback_days."
            )

        train_cutoff = sorted_dates[int(n_dates * TRAIN_FRAC) - 1]
        val_cutoff   = sorted_dates[int(n_dates * (TRAIN_FRAC + VAL_FRAC)) - 1]

        # Purge gap: a sample's label looks horizon_days into the future, so a
        # sample dated right at train_cutoff has its outcome determined by
        # events that chronologically fall inside the val period's date range.
        # Dropping the horizon_days immediately before each cutoff from the
        # earlier split removes that boundary contamination — standard
        # purging for time-series train/val/test splits with a lookahead
        # label window (see e.g. López de Prado, "Advances in Financial
        # Machine Learning", ch. 7). These purged dates belong to neither
        # split; they're excluded entirely rather than reassigned.
        purge = pd.Timedelta(days=self.horizon_days)

        train = df[df["sample_date"] <= train_cutoff - purge].copy()
        val   = df[
            (df["sample_date"] > train_cutoff)
            & (df["sample_date"] <= val_cutoff - purge)
        ].copy()
        test  = df[df["sample_date"] > val_cutoff].copy()

        if train.empty or val.empty:
            raise ValueError(
                f"Purge gap of {self.horizon_days}d emptied train (n={len(train)}) "
                f"or val (n={len(val)}) — increase Bronze data volume or reduce "
                "horizon_days relative to the date range."
            )

        logger.info(
            "Split: train=%d (dates ≤ %s), val=%d (≤ %s), test=%d  [%dd purge gap before "
            "each cutoff]",
            len(train), train_cutoff.date(), len(val), val_cutoff.date(), len(test),
            self.horizon_days,
        )

        # ── Step 5: Fit StandardScaler on train ───────────────────────────────
        scaler = StandardScaler()
        train[NUMERIC_FEATURE_COLS] = scaler.fit_transform(train[NUMERIC_FEATURE_COLS].values)
        val[NUMERIC_FEATURE_COLS]   = scaler.transform(val[NUMERIC_FEATURE_COLS].values)
        test[NUMERIC_FEATURE_COLS]  = scaler.transform(test[NUMERIC_FEATURE_COLS].values)

        # ── Step 6: Build vocabulary mappings ────────────────────────────────
        # Built from the full dataset so all entities appear in the vocabulary,
        # even if some are absent from the train split.
        wh_vocab  = {w: i for i, w in enumerate(sorted(df["warehouse_id"].unique()))}
        sku_vocab = {s: i for i, s in enumerate(sorted(df["sku_id"].unique()))}

        # ── Step 7: Baseline metrics on test split ────────────────────────────
        y_true_test  = test[TARGET_COL].values.astype(float)
        y_base_test  = test[BASELINE_COL].values.astype(float)

        if y_true_test.sum() == 0 or y_true_test.sum() == len(y_true_test):
            logger.warning("Test split has only one class — baseline metrics will be degenerate.")
            baseline_test_metrics: dict = {"auc_roc": None, "auprc": None, "brier": None}
        else:
            baseline_test_metrics = compute_metrics(y_true_test, y_base_test)
            logger.info(
                "Baseline test metrics: AUC-ROC=%.4f  AUPRC=%.4f  Brier=%.4f",
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

        # ── Step 9: Save preprocessor ────────────────────────────────────────
        preprocessor_path = self.output_dir / "preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(scaler, f)

        # ── Step 10: Save feature metadata ───────────────────────────────────
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
            "horizon_days":       self.horizon_days,
            "split": {
                "train_end": str(train_cutoff.date()),
                "val_end":   str(val_cutoff.date()),
                "train_frac": TRAIN_FRAC,
                "val_frac":   VAL_FRAC,
            },
            "positive_rate_train": pos_rate_train,
            "pos_weight":          _compute_pos_weight(train[TARGET_COL].values),
            "created_at":          datetime.now(timezone.utc).isoformat(),
        }
        metadata_path = self.output_dir / "feature_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # ── Step 11: Save baseline metrics ───────────────────────────────────
        baseline_path = self.output_dir / "baseline_metrics.json"
        with open(baseline_path, "w") as f:
            json.dump(
                {"test": baseline_test_metrics, "description": "days_of_supply < 7"},
                f, indent=2,
            )

        manifest = DatasetManifest(
            output_dir            = str(self.output_dir),
            train_path            = str(train_path),
            val_path              = str(val_path),
            test_path             = str(test_path),
            metadata_path         = str(metadata_path),
            preprocessor_path     = str(preprocessor_path),
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


def _compute_pos_weight(labels: np.ndarray) -> float:
    """
    n_negative / n_positive for BCEWithLogitsLoss pos_weight.
    Clamped to [0.1, 10.0] to avoid extreme weighting on tiny datasets.
    """
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0:
        return 1.0
    return float(np.clip(n_neg / n_pos, 0.1, 10.0))
