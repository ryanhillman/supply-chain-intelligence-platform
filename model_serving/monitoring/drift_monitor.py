"""
drift_monitor.py — feature distribution shift and prediction variance monitoring.

Background (ADR-005 Decision 7)
---------------------------------
The Phase 4 model was trained on 60x-tiled Bronze data:
  avg_daily_sales_7d (training) ≈ 2,785  units/day
  avg_daily_sales_7d (real)     ≈   207  units/day
  deviation                     ≈ -9.5 σ

This 13× velocity inflation collapses model predictions to prob≈0, score≈1 for all
120 entities. PSI monitoring detects this shift and blocks prediction consumers
until retraining on real data resolves the distribution mismatch.

Flow
----
1. Load StandardScaler (mean_, scale_) from preprocessor.pkl.
   These encode the training distribution for all 21 NUMERIC_FEATURE_COLS.
2. Load current feature values from Bigtable:
   - FeatureClient.scan_all() returns raw Bigtable features.
   - fill_defaults() + engineer_features() derive all 21 features.
   - Extract the 4 monitored features.
3. Read current predictions from Bigtable predictions column family
   (shortage_prob_7d, stockout_risk_score) — optional; skipped if empty.
4. Compute PSI for each monitored feature using 10 equal-frequency bins
   derived from the training normal distribution.
5. Classify PSI per ADR-005 thresholds:
     PSI < 0.10  → stable
     0.10 ≤ PSI < 0.20 → warning
     PSI ≥ 0.20  → retraining trigger
6. Detect prediction variance collapse: std < 0.01.
7. Compare the model's binary risk call (shortage_prob_7d >= 0.5) against
   the rule-based risk__reorder_urgency heuristic (urgency >= REORDER) for
   every entity with both values present. Agreement > 95% means the model
   adds no signal over the heuristic — retraining trigger (ADR-005
   Decision 7, trigger 3).
8. Print a tabular report. Optionally emit Cloud Monitoring metrics.

PSI formula
-----------
    PSI = Σ_i (A_i - E_i) × ln(A_i / E_i)

    Where:
      N     = 10 equal-probability bins, edges at normal quantiles of (μ, σ)
      E_i   = 0.10 (expected training fraction per bin, equal by construction)
      A_i   = actual fraction of current Bigtable values in bin i
      μ, σ  = scaler.mean_[feature_idx], scaler.scale_[feature_idx]

    Zero-count bins replaced with ε=1e-4 to avoid log(0).

Required packages:
    numpy>=1.26.0
    pandas>=2.1.0
    scipy>=1.11.0          (for normal quantile bins)
    scikit-learn>=1.3.0
    google-cloud-bigtable>=2.22.0

Optional:
    google-cloud-monitoring>=2.18.0   (for Cloud Monitoring metric emission)
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

from model_serving.pipeline.constants import VARIANCE_COLLAPSE_THRESHOLD as _VARIANCE_COLLAPSE_THRESHOLD

_N_BINS                = 10
_PSI_EPSILON           = 1e-4   # replace zero bin counts before log()
_PSI_STABLE_THRESHOLD  = 0.10
_PSI_WARNING_THRESHOLD = 0.20   # >= this → retraining trigger

# Baseline divergence (ADR-005 Decision 7, trigger 3): if the model's binary
# risk call agrees with the rule-based risk__reorder_urgency heuristic more
# than this fraction of the time, the model adds no signal over the heuristic.
_BASELINE_DIVERGENCE_THRESHOLD = 0.95
_BASELINE_RISK_FAMILY          = "risk"
_BASELINE_URGENCY_COL          = b"urgency"
_BASELINE_PROB_THRESHOLD       = 0.5   # shortage_prob_7d >= this → model says "at risk"
_BASELINE_AT_RISK_URGENCY_MIN  = 2     # urgency >= REORDER(2) → heuristic says "at risk"

# Monitored features: (display_name, flat_feature_name)
# The flat name must appear in NUMERIC_FEATURE_COLS from feature_engineer.py.
# days_of_supply is a derived feature computed by engineer_features();
# it is not read directly from Bigtable but re-derived for consistency.
_MONITORED_FEATURES: list[tuple[str, str]] = [
    ("inventory__days_of_supply",       "days_of_supply"),
    ("velocity__sales_7d",              "sales_7d"),
    ("velocity__avg_daily_sales_7d",    "avg_daily_sales_7d"),
    ("supplier__supplier_reliability",  "supplier_reliability"),
]

# Column families to skip in predictions read (only read "predictions" family).
_PREDICTIONS_FAMILY = "predictions"
_PREDICTIONS_PROB_COL   = b"shortage_prob_7d"
_PREDICTIONS_SCORE_COL  = b"stockout_risk_score"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FeatureMetrics:
    """PSI result for one monitored feature."""

    display_name:  str
    feature_name:  str
    training_mean: float
    training_std:  float
    current_mean:  float
    current_std:   float
    current_n:     int
    psi:           float
    status:        str   # "stable" | "warning" | "trigger"


@dataclass
class PredictionMetrics:
    """Variance stats read from the Bigtable predictions column family."""

    prob_mean:                float
    prob_std:                 float
    score_mean:               float
    score_std:                float
    prob_variance_collapsed:  bool
    score_variance_collapsed: bool
    n_entities:               int


@dataclass
class BaselineDivergenceMetrics:
    """
    Agreement between the model's binary risk call and the rule-based
    risk__reorder_urgency heuristic (ADR-005 Decision 7, trigger 3).
    """

    n_compared:     int
    n_agree:        int
    agreement_rate: float
    status:         str   # "stable" | "trigger"


@dataclass
class DriftReport:
    """Full output of a DriftMonitor.run() call."""

    run_timestamp:      str
    feature_metrics:    list[FeatureMetrics]
    prediction_metrics: PredictionMetrics | None
    baseline_metrics:   BaselineDivergenceMetrics | None
    requires_retraining: bool
    trigger_reasons:    list[str]  # human-readable reasons for retraining flag


# ── Main class ────────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Computes feature PSI and prediction variance metrics for the SCIP pipeline.

    Parameters
    ----------
    project_id:
        GCP project ID.
    instance_id:
        Bigtable instance ID (e.g. "scip-feature-store").
    artifacts_dir:
        Directory containing feature_metadata.json.
        Default: "ml_training/artifacts"
    dataset_dir:
        Directory containing preprocessor.pkl.
        Default: "ml_training/artifacts/dataset"
    table_id:
        Bigtable table ID (default "inventory_features").
    credentials_path:
        Path to service account JSON key. None = application default credentials.
    """

    def __init__(
        self,
        project_id: str,
        instance_id: str,
        artifacts_dir: str | Path = "ml_training/artifacts",
        dataset_dir:   str | Path = "ml_training/artifacts/dataset",
        table_id: str = "inventory_features",
        credentials_path: str | None = None,
    ) -> None:
        self.project_id  = project_id
        self.instance_id = instance_id
        self.table_id    = table_id

        self._artifacts_dir = Path(artifacts_dir)
        self._dataset_dir   = Path(dataset_dir)

        self._scaler   = self._load_scaler()
        self._metadata = self._load_metadata()

        # Import from ml_training — sys.path must include the repo root.
        from ml_training.datasets.feature_engineer import (
            NUMERIC_FEATURE_COLS,
            engineer_features,
            fill_defaults,
        )
        self._NUMERIC_FEATURE_COLS = NUMERIC_FEATURE_COLS
        self._engineer_features    = engineer_features
        self._fill_defaults        = fill_defaults

        # Build lookup: flat feature name → scaler index
        self._feature_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(NUMERIC_FEATURE_COLS)
        }

        # Verify all monitored features are in NUMERIC_FEATURE_COLS.
        unknown = [
            fname for _, fname in _MONITORED_FEATURES
            if fname not in self._feature_to_idx
        ]
        if unknown:
            raise ValueError(
                f"Monitored features not found in NUMERIC_FEATURE_COLS: {unknown}"
            )

        # Bigtable client (lazy — set up once, reused for both feature and prediction reads).
        self._creds = None
        if credentials_path:
            from google.oauth2 import service_account
            self._creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        from google.cloud import bigtable
        bigtable_kwargs: dict = {"project": project_id, "admin": False}
        if self._creds:
            bigtable_kwargs["credentials"] = self._creds
        self._bigtable_client = bigtable.Client(**bigtable_kwargs)
        self._table     = self._bigtable_client.instance(instance_id).table(table_id)

        logger.info(
            "DriftMonitor ready: bigtable=%s/%s  artifacts=%s  scaler_features=%d",
            instance_id, table_id, self._dataset_dir,
            len(self._NUMERIC_FEATURE_COLS),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        read_predictions: bool = True,
        report_path: str | Path | None = None,
        emit_cloud_monitoring: bool = False,
    ) -> DriftReport:
        """
        Run the full drift monitoring check.

        Parameters
        ----------
        read_predictions:
            If True, read shortage_prob_7d and stockout_risk_score from the
            Bigtable predictions column family to compute variance stats.
            Set False if Phase 5.4 scoring has not been run yet.
        report_path:
            If provided, save the DriftReport as JSON to this path.
        emit_cloud_monitoring:
            If True, emit PSI and variance metrics to Google Cloud Monitoring.
            Requires google-cloud-monitoring and monitoring.metricWriter IAM.
            The scip-vertex-sa has this role (see infrastructure/gcp/vertex_ai.tf).

        Returns
        -------
        DriftReport
        """
        run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info("=== SCIP Drift Monitor — %s ===", run_ts)

        # Step 1: collect current feature values from Bigtable.
        logger.info("Reading feature distribution from Bigtable %s/%s ...",
                    self.instance_id, self.table_id)
        feature_values = self._collect_feature_values()

        # Step 2: compute PSI for each monitored feature.
        feature_metrics: list[FeatureMetrics] = []
        for display_name, feature_name in _MONITORED_FEATURES:
            values = feature_values.get(feature_name, [])
            fm     = self._compute_feature_metrics(display_name, feature_name, values)
            feature_metrics.append(fm)
            logger.info(
                "  %-40s n=%-4d  PSI=%.4f  %s",
                display_name, fm.current_n, fm.psi, fm.status.upper(),
            )

        # Step 3: collect prediction variance (optional).
        prediction_metrics: PredictionMetrics | None = None
        if read_predictions:
            logger.info("Reading prediction variance from Bigtable predictions family ...")
            prediction_metrics = self._collect_prediction_values()
            if prediction_metrics:
                logger.info(
                    "  shortage_prob_7d    mean=%.4f  std=%.4f  collapsed=%s",
                    prediction_metrics.prob_mean, prediction_metrics.prob_std,
                    prediction_metrics.prob_variance_collapsed,
                )
                logger.info(
                    "  stockout_risk_score mean=%.4f  std=%.4f  collapsed=%s",
                    prediction_metrics.score_mean, prediction_metrics.score_std,
                    prediction_metrics.score_variance_collapsed,
                )
            else:
                logger.warning(
                    "No predictions found in Bigtable predictions family. "
                    "Run model_serving/scripts/score.py first."
                )

        # Step 3b: compare model predictions against the rule-based heuristic.
        baseline_metrics: BaselineDivergenceMetrics | None = None
        if read_predictions:
            logger.info("Comparing predictions against risk__reorder_urgency heuristic ...")
            baseline_metrics = self._collect_baseline_divergence()
            if baseline_metrics:
                logger.info(
                    "  baseline agreement=%.4f  n=%d  %s",
                    baseline_metrics.agreement_rate, baseline_metrics.n_compared,
                    baseline_metrics.status.upper(),
                )
            else:
                logger.warning(
                    "No entities with both a prediction and a risk__reorder_urgency "
                    "value — skipping baseline divergence check."
                )

        # Step 4: build report and assess retraining recommendation.
        trigger_reasons: list[str] = []
        for fm in feature_metrics:
            if fm.status == "trigger":
                trigger_reasons.append(
                    f"{fm.display_name}: PSI={fm.psi:.4f} >= {_PSI_WARNING_THRESHOLD}"
                )
        if prediction_metrics:
            if prediction_metrics.prob_variance_collapsed:
                trigger_reasons.append(
                    f"shortage_prob_7d variance collapsed: "
                    f"std={prediction_metrics.prob_std:.4f} < {_VARIANCE_COLLAPSE_THRESHOLD}"
                )
            if prediction_metrics.score_variance_collapsed:
                trigger_reasons.append(
                    f"stockout_risk_score variance collapsed: "
                    f"std={prediction_metrics.score_std:.4f} < {_VARIANCE_COLLAPSE_THRESHOLD}"
                )
        if baseline_metrics and baseline_metrics.status == "trigger":
            trigger_reasons.append(
                f"baseline divergence: model agrees with risk__reorder_urgency "
                f"heuristic {baseline_metrics.agreement_rate:.1%} of the time "
                f"(n={baseline_metrics.n_compared}) "
                f"> {_BASELINE_DIVERGENCE_THRESHOLD:.0%} — model adds no signal"
            )

        report = DriftReport(
            run_timestamp       = run_ts,
            feature_metrics     = feature_metrics,
            prediction_metrics  = prediction_metrics,
            baseline_metrics    = baseline_metrics,
            requires_retraining = bool(trigger_reasons),
            trigger_reasons     = trigger_reasons,
        )

        self._print_report(report)

        if emit_cloud_monitoring:
            self._emit_cloud_monitoring(report)

        if report_path:
            self._save_report(report, Path(report_path))

        return report

    def close(self) -> None:
        self._bigtable_client.close()

    def __enter__(self) -> "DriftMonitor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Bigtable reads ─────────────────────────────────────────────────────────

    def _collect_feature_values(self) -> dict[str, list[float]]:
        """
        Scan all feature rows from Bigtable, run engineer_features(), and
        return a dict of {flat_feature_name: [float, ...]} for monitored features.

        Scans self._table directly (the shared Bigtable client set up in __init__)
        rather than instantiating a separate FeatureClient, to avoid closing the
        shared client when a context manager exits.
        """
        from google.cloud.bigtable import row_filters as bigtable_filters
        from model_serving.pipeline.feature_client import (
            _BIGTABLE_TO_FEATURE,
            _FEATURE_FAMILIES,
            _decode_double,
            FeatureRow,
        )
        from model_serving.pipeline.request_builder import _RAW_FEATURES

        filter_ = bigtable_filters.RowFilterChain(filters=[
            bigtable_filters.FamilyNameRegexFilter("|".join(sorted(_FEATURE_FAMILIES))),
            bigtable_filters.CellsColumnLimitFilter(1),
        ])

        feature_rows: list[FeatureRow] = []
        n_skipped = 0

        for row_data in self._table.read_rows(filter_=filter_):
            row_key = row_data.row_key.decode("utf-8")
            parts   = row_key.split("#")
            if len(parts) != 2:
                n_skipped += 1
                continue
            warehouse_id, sku_id = parts[0], parts[1]
            raw_features: dict[str, float] = {}

            for cf_name, cf_data in row_data.cells.items():
                if cf_name not in _FEATURE_FAMILIES:
                    continue
                for col_qualifier_bytes, cells in cf_data.items():
                    col_qualifier = col_qualifier_bytes.decode("utf-8")
                    feature_name  = _BIGTABLE_TO_FEATURE.get((cf_name, col_qualifier))
                    if feature_name is None:
                        continue
                    decoded = _decode_double(cells[0].value, cf_name, col_qualifier)
                    if decoded is not None:
                        raw_features[feature_name] = decoded

            feature_rows.append(FeatureRow(
                row_key=row_key, warehouse_id=warehouse_id,
                sku_id=sku_id, features=raw_features,
            ))

        logger.info(
            "Bigtable scan complete: %d rows, %d skipped", len(feature_rows), n_skipped
        )

        if not feature_rows:
            logger.warning("No feature rows returned from Bigtable")
            return {}

        # Build DataFrame and apply feature engineering (same as RequestBuilder.build()).
        records = []
        for fr in feature_rows:
            row = {col: fr.features.get(col, np.nan) for col in _RAW_FEATURES}
            row["warehouse_id"] = fr.warehouse_id
            row["sku_id"]       = fr.sku_id
            records.append(row)

        df = pd.DataFrame(records)
        self._fill_defaults(df)
        df = self._engineer_features(df)

        # Extract only the monitored features.
        result: dict[str, list[float]] = {}
        for _, feature_name in _MONITORED_FEATURES:
            if feature_name in df.columns:
                vals = df[feature_name].dropna().tolist()
                result[feature_name] = [float(v) for v in vals]
            else:
                result[feature_name] = []

        logger.info(
            "Feature values collected: %d entities, %d monitored features",
            len(feature_rows),
            sum(1 for v in result.values() if v),
        )
        return result

    def _collect_prediction_values(self) -> PredictionMetrics | None:
        """
        Read shortage_prob_7d and stockout_risk_score from the Bigtable
        predictions column family. Values are stored as UTF-8 strings by
        prediction_writer.py.

        Returns None if no prediction rows are found.
        """
        from google.cloud.bigtable import row_filters

        filter_ = row_filters.RowFilterChain(filters=[
            row_filters.FamilyNameRegexFilter(_PREDICTIONS_FAMILY),
            row_filters.CellsColumnLimitFilter(1),
        ])

        probs:  list[float] = []
        scores: list[float] = []

        for row_data in self._table.read_rows(filter_=filter_):
            cells = row_data.cells.get(_PREDICTIONS_FAMILY, {})

            prob_cells  = cells.get(_PREDICTIONS_PROB_COL,  [])
            score_cells = cells.get(_PREDICTIONS_SCORE_COL, [])

            if prob_cells:
                try:
                    probs.append(float(prob_cells[0].value.decode("utf-8")))
                except (ValueError, UnicodeDecodeError):
                    pass

            if score_cells:
                try:
                    scores.append(float(score_cells[0].value.decode("utf-8")))
                except (ValueError, UnicodeDecodeError):
                    pass

        if not probs and not scores:
            return None

        n = max(len(probs), len(scores))
        prob_arr  = np.array(probs,  dtype=float)
        score_arr = np.array(scores, dtype=float)

        prob_std  = float(np.std(prob_arr))  if len(prob_arr)  > 1 else 0.0
        score_std = float(np.std(score_arr)) if len(score_arr) > 1 else 0.0

        return PredictionMetrics(
            prob_mean               = float(np.mean(prob_arr))  if len(prob_arr)  > 0 else 0.0,
            prob_std                = prob_std,
            score_mean              = float(np.mean(score_arr)) if len(score_arr) > 0 else 0.0,
            score_std               = score_std,
            prob_variance_collapsed  = prob_std  < _VARIANCE_COLLAPSE_THRESHOLD,
            score_variance_collapsed = score_std < _VARIANCE_COLLAPSE_THRESHOLD,
            n_entities              = n,
        )

    def _collect_baseline_divergence(self) -> BaselineDivergenceMetrics | None:
        """
        Compare the model's binary risk call against the rule-based
        risk__reorder_urgency heuristic for every entity that has both a
        prediction and an urgency value.

        model_at_risk     = shortage_prob_7d >= _BASELINE_PROB_THRESHOLD
        heuristic_at_risk = reorder_urgency  >= _BASELINE_AT_RISK_URGENCY_MIN

        risk__reorder_urgency is written to Bigtable in the "risk" column
        family as a big-endian double (same encoding as every other numeric
        feature — see spark/utils/bigtable_writer.py), so it's decoded with
        feature_client's _decode_double rather than the UTF-8 decode used
        for the predictions family.

        Returns None if no entity has both values present.
        """
        from google.cloud.bigtable import row_filters
        from model_serving.pipeline.feature_client import _decode_double

        filter_ = row_filters.RowFilterChain(filters=[
            row_filters.FamilyNameRegexFilter(
                f"{_PREDICTIONS_FAMILY}|{_BASELINE_RISK_FAMILY}"
            ),
            row_filters.CellsColumnLimitFilter(1),
        ])

        n_compared = 0
        n_agree    = 0

        for row_data in self._table.read_rows(filter_=filter_):
            pred_cells = row_data.cells.get(_PREDICTIONS_FAMILY, {})
            risk_cells = row_data.cells.get(_BASELINE_RISK_FAMILY, {})

            prob_cells    = pred_cells.get(_PREDICTIONS_PROB_COL, [])
            urgency_cells = risk_cells.get(_BASELINE_URGENCY_COL, [])

            if not prob_cells or not urgency_cells:
                continue

            try:
                prob = float(prob_cells[0].value.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue

            urgency = _decode_double(
                urgency_cells[0].value, _BASELINE_RISK_FAMILY, "urgency"
            )
            if urgency is None:
                continue

            model_at_risk     = prob    >= _BASELINE_PROB_THRESHOLD
            heuristic_at_risk = urgency >= _BASELINE_AT_RISK_URGENCY_MIN

            n_compared += 1
            if model_at_risk == heuristic_at_risk:
                n_agree += 1

        if n_compared == 0:
            return None

        agreement_rate = n_agree / n_compared
        status = (
            "trigger" if agreement_rate > _BASELINE_DIVERGENCE_THRESHOLD
            else "stable"
        )

        return BaselineDivergenceMetrics(
            n_compared     = n_compared,
            n_agree        = n_agree,
            agreement_rate = agreement_rate,
            status         = status,
        )

    # ── PSI computation ────────────────────────────────────────────────────────

    def _compute_feature_metrics(
        self,
        display_name: str,
        feature_name: str,
        current_values: list[float],
    ) -> FeatureMetrics:
        """
        Compute PSI and status for one monitored feature.

        The training distribution is approximated as N(μ, σ) where μ and σ
        are taken from StandardScaler.mean_[i] and StandardScaler.scale_[i].
        This is an approximation — actual training data is not needed at
        inference time, but the scaler's fit statistics encode the same
        distributional information.
        """
        idx          = self._feature_to_idx[feature_name]
        train_mean   = float(self._scaler.mean_[idx])
        train_std    = float(self._scaler.scale_[idx])

        if not current_values:
            logger.warning("No current values for %s — PSI set to NaN", display_name)
            return FeatureMetrics(
                display_name  = display_name,
                feature_name  = feature_name,
                training_mean = train_mean,
                training_std  = train_std,
                current_mean  = float("nan"),
                current_std   = float("nan"),
                current_n     = 0,
                psi           = float("nan"),
                status        = "unknown",
            )

        arr = np.array(current_values, dtype=float)
        psi = self._compute_psi(train_mean, train_std, arr)

        return FeatureMetrics(
            display_name  = display_name,
            feature_name  = feature_name,
            training_mean = train_mean,
            training_std  = train_std,
            current_mean  = float(np.mean(arr)),
            current_std   = float(np.std(arr)),
            current_n     = len(arr),
            psi           = psi,
            status        = _classify_psi(psi),
        )

    def _compute_psi(
        self,
        training_mean: float,
        training_std:  float,
        current_values: np.ndarray,
    ) -> float:
        """
        Population Stability Index using 10 equal-frequency bins.

        Bins are defined at the 10th, 20th, ..., 90th percentile of the
        training normal distribution N(training_mean, training_std).

        PSI = Σ_i (A_i - E_i) × ln(A_i / E_i)

        Where:
          E_i = 1/N = 0.10  (equal by construction)
          A_i = actual fraction of current values in bin i

        References
        ----------
        Siddiqi, N. (2006). Credit Risk Scorecards.
        Industry convention: PSI < 0.10 stable, ≥ 0.20 requires retraining.
        """
        if training_std <= 0:
            logger.warning(
                "Training std=%.6f <= 0 for this feature — cannot compute PSI",
                training_std,
            )
            return float("nan")

        # Equal-probability bin edges based on training normal distribution.
        # quantile_points: [0, 0.1, 0.2, ..., 1.0] → 11 values, 10 intervals.
        quantile_points = np.linspace(0.0, 1.0, _N_BINS + 1)
        bin_edges       = scipy_stats.norm.ppf(
            quantile_points, loc=training_mean, scale=training_std
        )
        # Force true infinity at the tails to capture all current values.
        bin_edges[0]  = -np.inf
        bin_edges[-1] =  np.inf

        # Expected fraction: exactly 1/_N_BINS per bin (equal-frequency design).
        expected = np.full(_N_BINS, 1.0 / _N_BINS)

        # Actual fraction of current Bigtable values per bin.
        counts, _  = np.histogram(current_values, bins=bin_edges)
        n          = len(current_values)
        actual     = counts.astype(float) / n

        # Replace zero bins with epsilon before taking the log.
        # A bin with zero actual count means ALL current data is elsewhere —
        # still a valid (and large) PSI signal, capped by epsilon.
        actual   = np.where(actual   < _PSI_EPSILON, _PSI_EPSILON, actual)
        expected = np.where(expected < _PSI_EPSILON, _PSI_EPSILON, expected)

        # Renormalize after epsilon replacement so fractions sum to 1.
        actual   = actual   / actual.sum()
        expected = expected / expected.sum()

        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        return max(psi, 0.0)  # PSI is always non-negative; clamp floating-point noise

    # ── Reporting ──────────────────────────────────────────────────────────────

    def _print_report(self, report: DriftReport) -> None:
        sep  = "=" * 80
        sep2 = "-" * 80

        print(f"\n{sep}")
        print(f"  SCIP Drift Monitor Report  —  {report.run_timestamp}")
        print(sep)

        # Feature PSI table
        print("\nFeature Distribution Shift (PSI)")
        print(sep2)
        print(
            f"  {'Feature':<40} {'Train-μ':>8} {'Train-σ':>8}"
            f" {'Cur-μ':>8} {'Cur-σ':>7} {'N':>5} {'PSI':>7}  Status"
        )
        print(
            f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*5} {'-'*7}  ------"
        )

        for fm in report.feature_metrics:
            status_icon = {
                "stable":  "OK     ",
                "warning": "WARNING",
                "trigger": "TRIGGER",
                "unknown": "UNKNOWN",
            }.get(fm.status, fm.status.upper())

            if fm.current_n == 0:
                print(f"  {fm.display_name:<40}  (no data)")
                continue

            print(
                f"  {fm.display_name:<40}"
                f" {fm.training_mean:>8.2f} {fm.training_std:>8.2f}"
                f" {fm.current_mean:>8.2f} {fm.current_std:>7.2f}"
                f" {fm.current_n:>5}"
                f" {fm.psi:>7.4f}  {status_icon}"
            )

        # PSI legend
        print()
        print(
            f"  PSI legend: "
            f"< {_PSI_STABLE_THRESHOLD}  = stable  |  "
            f"{_PSI_STABLE_THRESHOLD}–{_PSI_WARNING_THRESHOLD} = warning  |  "
            f">= {_PSI_WARNING_THRESHOLD} = TRIGGER"
        )

        # Prediction variance table
        if report.prediction_metrics:
            pm = report.prediction_metrics
            print(f"\nPrediction Variance  (n={pm.n_entities} entities)")
            print(sep2)

            def _var_line(name, mean, std, collapsed):
                flag = f"  [COLLAPSED < {_VARIANCE_COLLAPSE_THRESHOLD}]" if collapsed else ""
                print(f"  {name:<28} mean={mean:>7.4f}  std={std:>7.4f}{flag}")

            _var_line("shortage_prob_7d",    pm.prob_mean,  pm.prob_std,  pm.prob_variance_collapsed)
            _var_line("stockout_risk_score", pm.score_mean, pm.score_std, pm.score_variance_collapsed)

            if pm.prob_variance_collapsed or pm.score_variance_collapsed:
                print(
                    "\n  [NOTE] Variance collapse with prob~0 + score~1 is expected given the\n"
                    "         current model's training/inference distribution mismatch\n"
                    "         (ADR-005 Decision 7). This confirms retraining is required."
                )
        else:
            print("\nPrediction Variance  — no predictions in Bigtable yet")
            print(sep2)
            print("  Run model_serving/scripts/score.py first to populate predictions.")

        # Baseline divergence table
        if report.baseline_metrics:
            bm = report.baseline_metrics
            print(f"\nBaseline Divergence  (vs. risk__reorder_urgency heuristic)")
            print(sep2)
            flag = "  [TRIGGER]" if bm.status == "trigger" else ""
            print(
                f"  agreement={bm.agreement_rate:.1%}  n={bm.n_compared}  "
                f"threshold={_BASELINE_DIVERGENCE_THRESHOLD:.0%}{flag}"
            )
            if bm.status == "trigger":
                print(
                    "\n  [NOTE] The model agrees with the rule-based heuristic on\n"
                    "         almost every entity — it is not adding predictive signal\n"
                    "         over the simple days-of-supply threshold (ADR-005\n"
                    "         Decision 7, trigger 3)."
                )
        else:
            print("\nBaseline Divergence  — no entities with both a prediction and urgency value")
            print(sep2)
            print("  Run model_serving/scripts/score.py first to populate predictions.")

        # Retraining recommendation
        print(f"\nRetraining Recommendation")
        print(sep2)
        if report.requires_retraining:
            print(f"  ** RETRAINING REQUIRED **")
            for reason in report.trigger_reasons:
                print(f"     - {reason}")
            print()
            print(
                "  Retraining gate (ADR-005 Decision 7):\n"
                "    1. Accumulate >= 38 real Bronze days (no --simulate-days flag)\n"
                "    2. Retrain:  python ml_training/scripts/train.py\n"
                "    3. Register: python model_serving/scripts/register.py ...\n"
                "    4. Deploy:   python model_serving/scripts/deploy.py deploy --traffic 0\n"
                "    5. Promote:  python model_serving/scripts/deploy.py promote\n"
                "    6. Re-score: python model_serving/scripts/score.py ...\n"
                "    7. Re-monitor (this script): PSI should drop below 0.10"
            )
        else:
            print("  No retraining required. All PSI values within stable thresholds.")

        print(sep)

    def _save_report(self, report: DriftReport, path: Path) -> None:
        """Serialize DriftReport to JSON for downstream consumption."""
        path.parent.mkdir(parents=True, exist_ok=True)

        def _fm_dict(fm: FeatureMetrics) -> dict:
            return {
                "display_name":  fm.display_name,
                "feature_name":  fm.feature_name,
                "training_mean": fm.training_mean,
                "training_std":  fm.training_std,
                "current_mean":  fm.current_mean if not _is_nan(fm.current_mean) else None,
                "current_std":   fm.current_std  if not _is_nan(fm.current_std)  else None,
                "current_n":     fm.current_n,
                "psi":           fm.psi if not _is_nan(fm.psi) else None,
                "status":        fm.status,
            }

        payload = {
            "run_timestamp":      report.run_timestamp,
            "requires_retraining": report.requires_retraining,
            "trigger_reasons":    report.trigger_reasons,
            "feature_psi":        [_fm_dict(fm) for fm in report.feature_metrics],
            "prediction_variance": {
                "prob_mean":               report.prediction_metrics.prob_mean,
                "prob_std":                report.prediction_metrics.prob_std,
                "score_mean":              report.prediction_metrics.score_mean,
                "score_std":               report.prediction_metrics.score_std,
                "prob_variance_collapsed": report.prediction_metrics.prob_variance_collapsed,
                "score_variance_collapsed": report.prediction_metrics.score_variance_collapsed,
                "n_entities":              report.prediction_metrics.n_entities,
            } if report.prediction_metrics else None,
            "baseline_divergence": {
                "n_compared":     report.baseline_metrics.n_compared,
                "n_agree":        report.baseline_metrics.n_agree,
                "agreement_rate": report.baseline_metrics.agreement_rate,
                "status":         report.baseline_metrics.status,
            } if report.baseline_metrics else None,
            "thresholds": {
                "psi_stable":           _PSI_STABLE_THRESHOLD,
                "psi_trigger":          _PSI_WARNING_THRESHOLD,
                "variance_collapse":    _VARIANCE_COLLAPSE_THRESHOLD,
                "baseline_divergence":  _BASELINE_DIVERGENCE_THRESHOLD,
                "n_psi_bins":           _N_BINS,
            },
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info("Drift report saved: %s", path)

    # ── Cloud Monitoring (optional stubs) ──────────────────────────────────────

    def _emit_cloud_monitoring(self, report: DriftReport) -> None:
        """
        Emit PSI and variance metrics to Google Cloud Monitoring.

        Metrics emitted:
          custom.googleapis.com/scip/drift/feature_psi  [GAUGE, double]
            label: feature_name
          custom.googleapis.com/scip/drift/prob_std     [GAUGE, double]
          custom.googleapis.com/scip/drift/score_std    [GAUGE, double]
          custom.googleapis.com/scip/drift/baseline_agreement_rate  [GAUGE, double]
          custom.googleapis.com/scip/drift/retraining_required  [GAUGE, int64]

        Requires:
          pip install google-cloud-monitoring>=2.18.0
          IAM: monitoring.metricWriter on scip-vertex-sa
               (granted in infrastructure/gcp/vertex_ai.tf)

        This method is a stub — it is safe to call without Cloud Monitoring
        configured. If the package is unavailable or auth fails, a warning is
        logged and the run continues normally.
        """
        try:
            from google.cloud import monitoring_v3
        except ImportError:
            logger.warning(
                "google-cloud-monitoring not installed — skipping Cloud Monitoring emission.\n"
                "Install with: pip install google-cloud-monitoring>=2.18.0"
            )
            return

        try:
            self._do_emit_cloud_monitoring(monitoring_v3, report)
        except Exception as exc:
            logger.warning(
                "Cloud Monitoring emission failed (non-fatal): %s\n"
                "PSI values were logged to the console and optionally to report_path.",
                exc,
            )

    def _do_emit_cloud_monitoring(self, monitoring_v3, report: DriftReport) -> None:
        """Internal: build and write time series to Cloud Monitoring."""
        from google.api import metric_pb2 as ga_metric
        from google.api import monitored_resource_pb2

        client_kwargs: dict = {"project": self.project_id}
        if self._creds:
            client_kwargs["credentials"] = self._creds
        cm_client = monitoring_v3.MetricServiceClient(**client_kwargs)

        project_name = f"projects/{self.project_id}"
        now_seconds  = int(datetime.now(timezone.utc).timestamp())

        def _build_series(metric_type: str, value: float,
                          labels: dict | None = None) -> monitoring_v3.TimeSeries:
            series = monitoring_v3.TimeSeries()
            series.metric.type   = metric_type
            series.resource.type = "global"
            if labels:
                series.metric.labels.update(labels)

            interval = monitoring_v3.TimeInterval(
                end_time={"seconds": now_seconds, "nanos": 0}
            )
            point = monitoring_v3.Point(
                interval=interval,
                value=monitoring_v3.TypedValue(double_value=value),
            )
            series.points.append(point)
            return series

        time_series: list = []

        # PSI per feature
        for fm in report.feature_metrics:
            if not _is_nan(fm.psi):
                time_series.append(_build_series(
                    "custom.googleapis.com/scip/drift/feature_psi",
                    fm.psi,
                    labels={"feature_name": fm.feature_name},
                ))

        # Prediction variance
        if report.prediction_metrics:
            pm = report.prediction_metrics
            time_series.append(_build_series(
                "custom.googleapis.com/scip/drift/prob_std",  pm.prob_std
            ))
            time_series.append(_build_series(
                "custom.googleapis.com/scip/drift/score_std", pm.score_std
            ))

        # Baseline divergence
        if report.baseline_metrics:
            time_series.append(_build_series(
                "custom.googleapis.com/scip/drift/baseline_agreement_rate",
                report.baseline_metrics.agreement_rate,
            ))

        # Retraining flag (1 = required, 0 = not required)
        retraining_series = monitoring_v3.TimeSeries()
        retraining_series.metric.type   = "custom.googleapis.com/scip/drift/retraining_required"
        retraining_series.resource.type = "global"
        interval = monitoring_v3.TimeInterval(
            end_time={"seconds": now_seconds, "nanos": 0}
        )
        point = monitoring_v3.Point(
            interval=interval,
            value=monitoring_v3.TypedValue(
                int64_value=1 if report.requires_retraining else 0
            ),
        )
        retraining_series.points.append(point)
        time_series.append(retraining_series)

        if time_series:
            cm_client.create_time_series(
                request={
                    "name":        project_name,
                    "time_series": time_series,
                }
            )
            logger.info(
                "Emitted %d Cloud Monitoring time series to project %s",
                len(time_series), self.project_id,
            )

    # ── Artifact loading ───────────────────────────────────────────────────────

    def _load_scaler(self):
        path = self._dataset_dir / "preprocessor.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"preprocessor.pkl not found at {path}. "
                "Run ml_training/scripts/build_dataset.py or download from GCS."
            )
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        logger.info(
            "Loaded StandardScaler: %d features, mean range [%.2f, %.2f]",
            len(scaler.mean_), float(scaler.mean_.min()), float(scaler.mean_.max()),
        )
        return scaler

    def _load_metadata(self) -> dict:
        path = self._artifacts_dir / "feature_metadata.json"
        if not path.exists():
            raise FileNotFoundError(
                f"feature_metadata.json not found at {path}. "
                "Run ml_training/scripts/train.py or download from GCS."
            )
        with open(path) as f:
            return json.load(f)


# ── Module helpers ─────────────────────────────────────────────────────────────

def _classify_psi(psi: float) -> str:
    """Classify a PSI value as stable / warning / trigger."""
    if _is_nan(psi):
        return "unknown"
    if psi < _PSI_STABLE_THRESHOLD:
        return "stable"
    if psi < _PSI_WARNING_THRESHOLD:
        return "warning"
    return "trigger"


def _is_nan(value: float) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True
