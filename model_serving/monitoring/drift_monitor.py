"""
drift_monitor.py — feature distribution shift and prediction variance monitoring.

Background
----------
A model scored on features unlike its training data keeps returning
predictions; nothing fails. This monitor detects that kind of shift: feature
PSI against the training split and the spread of the model's predictions,
flagged in the printed report / drift_report.json.
It is advisory only: nothing in score.py, batch_scorer.py or promote_model.py
reads requires_retraining back, so it does not block or reject predictions.
A human has to act on the report.

Flow
----
1. Load feature_metadata.json (feature_mean / feature_scale fit on the train
   split, reported next to the live mean/std) and drift_reference.json (PSI
   bins computed from the raw training split when the model was trained;
   required, no approximate fallback).
2. Load current feature values from Bigtable:
   - FeatureClient.scan_all() returns raw Bigtable features.
   - fill_defaults() + engineer_features() derive all 21 features.
   - Extract the PSI features and the warehouse-grain features.
3. Read shortage_prob_7d from the Bigtable predictions column family —
   optional; skipped if empty.
4. Compute PSI for each monitored feature using bins at the deciles of the
   raw training rows (drift_reference.json). supplier_reliability is a
   per-warehouse 90-day rate, so a snapshot holds 6 values, too few for a
   statistical trigger; each is checked against its warehouse's training
   p1-p99 and reported as an informational warning, because out-of-range
   values usually reflect seasonality outside the training window, which the
   test metrics show the model handles.
5. Classify PSI:
     PSI < 0.10  → stable
     0.10 ≤ PSI < 0.20 → warning
     PSI ≥ 0.20  → retraining trigger
6. Detect prediction variance collapse: std of shortage_prob_7d < 0.01.
   stockout_risk_score is not checked: it is a formula of stock and sales,
   so its spread reflects stock levels, not the model.
7. Print a tabular report. Optionally emit Cloud Monitoring metrics.

PSI formula
-----------
    PSI = Σ_i (A_i - E_i) × ln(A_i / E_i)

    Where:
      N     = up to 10 bins with edges at the (deduplicated) deciles of the
              raw training rows, stored in drift_reference.json — correct for
              any feature shape, and in the same units as the live values.
      E_i   = the actual training fraction in bin i (not assumed uniform:
              clustered features such as days_of_supply, capped at 365,
              collapse several decile edges onto one value)
      A_i   = actual fraction of current Bigtable values in bin i

    Zero-count bins replaced with ε=1e-4 to avoid log(0). drift_reference.json
    is required; there is no approximate fallback.

Required packages:
    numpy>=1.26.0
    pandas>=2.1.0
    google-cloud-bigtable>=2.22.0

Optional:
    google-cloud-monitoring>=2.18.0   (for Cloud Monitoring metric emission)
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from feature_registry import RAW_FEATURE_COLS
from ml_training.evaluation.drift_reference import (
    MONITORED_FEATURES as _MONITORED_FEATURES,
    WAREHOUSE_RANGE_FEATURES as _WAREHOUSE_RANGE_FEATURES,
    load_drift_reference,
    load_warehouse_ranges,
)
from model_serving import cli

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

from model_serving.pipeline.constants import (
    VARIANCE_COLLAPSE_THRESHOLD as _VARIANCE_COLLAPSE_THRESHOLD,
    variance_collapsed,
)

_N_BINS                = 10
_PSI_EPSILON           = 1e-4   # replace zero bin counts before log()
_PSI_STABLE_THRESHOLD  = 0.10
_PSI_WARNING_THRESHOLD = 0.20   # >= this → retraining trigger

# Monitored features come from ml_training.evaluation.drift_reference, shared
# with the trainer, which writes their PSI bins next to the model.


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
    """Spread of shortage_prob_7d read from the Bigtable predictions column family."""

    prob_mean:               float
    prob_std:                float
    prob_variance_collapsed: bool
    n_entities:              int


@dataclass
class WarehouseRangeMetrics:
    """Per-warehouse range check for one warehouse-grain feature."""

    display_name: str
    feature_name: str
    values:       dict[str, float]               # live value per warehouse
    ranges:       dict[str, tuple[float, float]]  # training p1-p99 per warehouse
    out_of_range: list[str]                       # outside its range, or no training range
    status:       str                             # "stable" | "warning" | "unknown"


@dataclass
class DriftReport:
    """Full output of a DriftMonitor.run() call."""

    run_timestamp:      str
    feature_metrics:    list[FeatureMetrics]
    range_metrics:      list[WarehouseRangeMetrics]
    prediction_metrics: PredictionMetrics | None
    requires_retraining: bool
    trigger_reasons:    list[str]  # human-readable reasons for retraining flag
    warnings:           list[str]  # informational only; never set requires_retraining


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
        Directory containing feature_metadata.json and drift_reference.json,
        both written next to model.pt by the trainer.
        Default: "ml_training/artifacts"
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
        table_id: str = "inventory_features",
        credentials_path: str | None = None,
    ) -> None:
        self.project_id  = project_id
        self.instance_id = instance_id
        self.table_id    = table_id

        self._artifacts_dir = Path(artifacts_dir)

        self._metadata      = self._load_metadata()
        self._feature_mean  = np.asarray(self._metadata["feature_mean"],  dtype=float)
        self._feature_scale = np.asarray(self._metadata["feature_scale"], dtype=float)
        self._training_quantiles, self._reference_info = self._load_drift_reference()
        self._warehouse_ranges = load_warehouse_ranges(self._artifacts_dir / "drift_reference.json")

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

        # For the Cloud Monitoring client; FeatureClient loads its own from the same key.
        self._creds = cli.load_credentials(credentials_path)

        from model_serving.pipeline.feature_client import FeatureClient
        self._feature_client = FeatureClient(
            project_id=project_id, instance_id=instance_id,
            table_id=table_id, credentials_path=credentials_path,
        )

        logger.info(
            "DriftMonitor ready: bigtable=%s/%s  artifacts=%s  n_features=%d",
            instance_id, table_id, self._artifacts_dir,
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
            If True, read shortage_prob_7d from the Bigtable predictions
            column family to compute variance stats.
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
        live = self._live_features()
        feature_values = self._collect_feature_values(live)

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

        # Step 2b: warehouse-grain features against each warehouse's training range.
        range_metrics = [
            self._check_warehouse_range(display_name, name, live)
            for display_name, name in _WAREHOUSE_RANGE_FEATURES
        ]

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
            else:
                logger.warning(
                    "No predictions found in Bigtable predictions family. "
                    "Run model_serving/scripts/score.py first."
                )

        # Step 4: build report and assess retraining recommendation.
        trigger_reasons: list[str] = []
        for fm in feature_metrics:
            if fm.status == "trigger":
                trigger_reasons.append(
                    f"{fm.display_name}: PSI={fm.psi:.4f} >= {_PSI_WARNING_THRESHOLD}"
                )
        warnings = [
            f"{rm.display_name}: {', '.join(rm.out_of_range)} outside the warehouse's "
            "training p1-p99 (informational, not a retraining trigger)"
            for rm in range_metrics if rm.status == "warning"
        ]
        if prediction_metrics and prediction_metrics.prob_variance_collapsed:
            trigger_reasons.append(
                f"shortage_prob_7d variance collapsed: "
                f"std={prediction_metrics.prob_std:.4f} < {_VARIANCE_COLLAPSE_THRESHOLD}"
            )

        report = DriftReport(
            run_timestamp       = run_ts,
            feature_metrics     = feature_metrics,
            range_metrics       = range_metrics,
            prediction_metrics  = prediction_metrics,
            requires_retraining = bool(trigger_reasons),
            trigger_reasons     = trigger_reasons,
            warnings            = warnings,
        )

        self._print_report(report)

        if emit_cloud_monitoring:
            self._emit_cloud_monitoring(report)

        if report_path:
            self._save_report(report, Path(report_path))

        return report

    def close(self) -> None:
        self._feature_client.close()

    def __enter__(self) -> "DriftMonitor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Bigtable reads ─────────────────────────────────────────────────────────

    def _live_features(self) -> pd.DataFrame:
        """
        Scan all feature rows via FeatureClient and run fill_defaults() and
        engineer_features(): the same read path and derivation as
        RequestBuilder.build(), so the monitor sees exactly what the scorer
        scores. One row per (warehouse_id, sku_id).
        """
        feature_rows = self._feature_client.scan_all()
        if not feature_rows:
            logger.warning("No feature rows returned from Bigtable")
            return pd.DataFrame()

        records = []
        for fr in feature_rows:
            row = {col: fr.features.get(col, np.nan) for col in RAW_FEATURE_COLS}
            row["warehouse_id"] = fr.warehouse_id
            row["sku_id"]       = fr.sku_id
            records.append(row)

        df = pd.DataFrame(records)
        self._fill_defaults(df)
        return self._engineer_features(df)

    @staticmethod
    def _collect_feature_values(live: pd.DataFrame) -> dict[str, list[float]]:
        """{flat_feature_name: [float, ...]} for the PSI features."""
        result: dict[str, list[float]] = {}
        for _, feature_name in _MONITORED_FEATURES:
            vals = live[feature_name].dropna().tolist() if feature_name in live.columns else []
            result[feature_name] = [float(v) for v in vals]
        logger.info("Feature values collected: %d entities", len(live))
        return result

    def _check_warehouse_range(
        self, display_name: str, feature_name: str, live: pd.DataFrame
    ) -> WarehouseRangeMetrics:
        """
        One live value per warehouse (the median over its SKUs, which all carry
        the same value) against that warehouse's training p1-p99. A warehouse
        with no training range counts as out of range.
        """
        ranges = self._warehouse_ranges.get(feature_name, {})
        if live.empty or feature_name not in live.columns:
            return WarehouseRangeMetrics(display_name, feature_name, {}, ranges, [], "unknown")
        values = {str(wh): float(v) for wh, v in
                  live.groupby("warehouse_id")[feature_name].median().items()}
        out = _out_of_range(values, ranges)
        return WarehouseRangeMetrics(
            display_name, feature_name, values, ranges, out, "warning" if out else "stable"
        )

    def _collect_prediction_values(self) -> PredictionMetrics | None:
        """Mean and spread of shortage_prob_7d. Returns None if no predictions are found."""
        probs = np.array(
            [r.shortage_prob_7d for r in self._feature_client.read_predictions()
             if r.shortage_prob_7d is not None],
            dtype=float,
        )
        if len(probs) == 0:
            return None

        prob_std = float(np.std(probs)) if len(probs) > 1 else 0.0
        return PredictionMetrics(
            prob_mean               = float(np.mean(probs)),
            prob_std                = prob_std,
            prob_variance_collapsed = variance_collapsed(prob_std),
            n_entities              = len(probs),
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

        μ and σ (training_mean / training_std in the report) come from
        feature_metadata.json; the PSI bins themselves come from
        drift_reference.json (_load_drift_reference).
        """
        idx          = self._feature_to_idx[feature_name]
        train_mean   = float(self._feature_mean[idx])
        train_std    = float(self._feature_scale[idx])
        bin_edges, expected = self._training_quantiles[feature_name]

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
        psi = self._compute_psi(arr, bin_edges, expected)

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
        current_values: np.ndarray,
        bin_edges: np.ndarray,
        expected: np.ndarray,
    ) -> float:
        """
        Population Stability Index using the training data's own bins.

        bin_edges/expected come from _load_drift_reference: bin_edges
        are the training distribution's own (deduplicated) decile edges,
        and expected is the ACTUAL fraction of training rows that fall in
        each of those bins — not assumed to be a uniform 1/10, since
        a monitored feature can be heavily clustered (days_of_supply is
        capped at 365, so that one value can hold a large share of rows) and
        naive equal-frequency binning degenerates in that case.

        Both sides are in raw feature units: the reference bins come from the
        raw training split (DatasetBuilder no longer standardizes the splits)
        and the live values from engineer_features() on raw Bigtable cells. An
        earlier version binned raw live values against standardized
        training rows and reported PSI 2–8 (TRIGGER) for the training data
        compared with itself; before that, a normal-quantile approximation
        of the training distribution did the same for skewed features.
        Neither path exists any more — see tests/test_drift_monitor_psi.py.

        PSI = Σ_i (A_i - E_i) × ln(A_i / E_i)

        Where:
          E_i = expected training fraction in bin i
          A_i = actual fraction of current values in bin i

        References
        ----------
        Siddiqi, N. (2006). Credit Risk Scorecards.
        Industry convention: PSI < 0.10 stable, ≥ 0.20 requires retraining.
        """
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
        info = self._reference_info
        print(
            f"  Reference: drift_reference.json — {info.get('n_rows')} training rows, "
            f"created {info.get('created_at')}"
        )

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

        # Per-warehouse range checks
        for rm in report.range_metrics:
            print(f"\nPer-Warehouse Range  {rm.display_name}  (training p1-p99)")
            print(sep2)
            if rm.status == "unknown":
                print("  (no data)")
                continue
            print(f"  {'Warehouse':<10} {'Live':>8} {'p1':>8} {'p99':>8}  Status")
            for wh in sorted(rm.values):
                lo, hi = rm.ranges.get(wh, (float("nan"), float("nan")))
                flag = "OUT OF RANGE (info)" if wh in rm.out_of_range else "OK"
                print(f"  {wh:<10} {rm.values[wh]:>8.3f} {lo:>8.3f} {hi:>8.3f}  {flag}")

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
            flag = f"  [COLLAPSED < {_VARIANCE_COLLAPSE_THRESHOLD}]" if pm.prob_variance_collapsed else ""
            print(f"  {'shortage_prob_7d':<28} mean={pm.prob_mean:>7.4f}  std={pm.prob_std:>7.4f}{flag}")

            if pm.prob_variance_collapsed:
                print(
                    "\n  [NOTE] Near-constant predictions mean the model no longer separates\n"
                    "         entities. Check the feature PSI above and retrain."
                )
        else:
            print("\nPrediction Variance  — no predictions in Bigtable yet")
            print(sep2)
            print("  Run model_serving/scripts/score.py first to populate predictions.")

        # Informational warnings (do not affect the recommendation)
        if report.warnings:
            print("\nWarnings (informational)")
            print(sep2)
            for w in report.warnings:
                print(f"  - {w}")

        # Retraining recommendation
        print("\nRetraining Recommendation")
        print(sep2)
        if report.requires_retraining:
            print("  ** RETRAINING REQUIRED **")
            for reason in report.trigger_reasons:
                print(f"     - {reason}")
            print()
            print(
                "  Retraining:\n"
                "    1. Rebuild:  python ml_training/scripts/build_dataset.py --bucket ...\n"
                "    2. Retrain:  python ml_training/scripts/train.py\n"
                "    3. Gate:     register.py, deploy.py deploy --traffic 0, then deploy.py promote\n"
                "    4. Re-score: python model_serving/scripts/score.py ...\n"
                "    5. Re-monitor (this script): PSI should drop below 0.10"
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
            "warnings":           report.warnings,
            "feature_psi":        [_fm_dict(fm) for fm in report.feature_metrics],
            "warehouse_ranges":   [
                {
                    "display_name": rm.display_name,
                    "feature_name": rm.feature_name,
                    "values":       rm.values,
                    "ranges":       {wh: list(r) for wh, r in rm.ranges.items()},
                    "out_of_range": rm.out_of_range,
                    "status":       rm.status,
                }
                for rm in report.range_metrics
            ],
            "prediction_variance": {
                "prob_mean":               report.prediction_metrics.prob_mean,
                "prob_std":                report.prediction_metrics.prob_std,
                "prob_variance_collapsed": report.prediction_metrics.prob_variance_collapsed,
                "n_entities":              report.prediction_metrics.n_entities,
            } if report.prediction_metrics else None,
            "thresholds": {
                "psi_stable":           _PSI_STABLE_THRESHOLD,
                "psi_trigger":          _PSI_WARNING_THRESHOLD,
                "variance_collapse":    _VARIANCE_COLLAPSE_THRESHOLD,
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

        cm_client = monitoring_v3.MetricServiceClient(credentials=self._creds)

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
            time_series.append(_build_series(
                "custom.googleapis.com/scip/drift/prob_std",
                report.prediction_metrics.prob_std,
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

    def _load_metadata(self) -> dict:
        return cli.read_json(
            self._artifacts_dir / "feature_metadata.json",
            "Run ml_training/scripts/train.py or download from GCS.",
        )

    def _load_drift_reference(self) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
        """
        PSI bins for the monitored features from drift_reference.json, which
        the trainer writes next to model.pt from the raw training split (see
        ml_training/evaluation/drift_reference.py). The bins are the training
        rows' deduplicated deciles with each bin's actual training fraction as
        `expected`, so they are correct for skewed and clustered features and
        give PSI ~0 for training-vs-itself.

        Reading the reference from the model's artifacts directory, rather than
        from a dataset directory, means the baseline is the one the deployed
        model was trained on. Raises FileNotFoundError if absent: there is no
        approximate fallback.
        """
        path = self._artifacts_dir / "drift_reference.json"
        if not path.exists():
            raise FileNotFoundError(
                f"drift_reference.json not found at {path}. It is written by "
                "ml_training/scripts/train.py next to model.pt; retrain or copy it "
                "into the artifacts directory."
            )

        bins, info = load_drift_reference(path)
        missing = [name for _, name in _MONITORED_FEATURES if name not in bins]
        if missing:
            raise ValueError(f"drift_reference.json lacks monitored features {missing}")

        logger.info(
            "Loaded drift reference (%d features, %s training rows, created %s) from %s",
            len(bins), info.get("n_rows"), info.get("created_at"), path,
        )
        return bins, info

# ── Module helpers ─────────────────────────────────────────────────────────────

def _out_of_range(values: dict[str, float], ranges: dict[str, tuple[float, float]]) -> list[str]:
    """Warehouses whose live value is outside their training range, or that have none."""
    out = []
    for wh, v in sorted(values.items()):
        lo, hi = ranges.get(wh, (None, None))
        if lo is None or not (lo <= v <= hi):
            out.append(wh)
    return out


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
