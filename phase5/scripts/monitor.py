"""
monitor.py — CLI entry point for Phase 5.5 drift monitoring.

Reads the current feature distribution from Bigtable, computes PSI for the
four key monitoring features, checks prediction variance collapse, and prints
a structured report with a retraining recommendation.

Usage
-----
    # Full monitoring run (feature PSI + prediction variance):
    python phase5/scripts/monitor.py \\
        --project supply-chain-platform-499920 \\
        --bigtable-instance scip-feature-store \\
        --sa-key infrastructure/gcp/keys/scip-vertex-sa.json

    # Feature PSI only (no Bigtable predictions read):
    python phase5/scripts/monitor.py \\
        --project supply-chain-platform-499920 \\
        --bigtable-instance scip-feature-store \\
        --no-predictions \\
        --sa-key infrastructure/gcp/keys/scip-vertex-sa.json

    # Save JSON report for downstream tooling:
    python phase5/scripts/monitor.py \\
        --project supply-chain-platform-499920 \\
        --bigtable-instance scip-feature-store \\
        --report-path phase5/artifacts/drift_report.json

    # With Cloud Monitoring emission (requires google-cloud-monitoring):
    python phase5/scripts/monitor.py \\
        --project supply-chain-platform-499920 \\
        --bigtable-instance scip-feature-store \\
        --emit-cloud-monitoring

Exit codes
----------
  0 — no retraining required (all PSI values stable, no variance collapse)
  1 — retraining recommended (one or more PSI values trigger threshold, or
      prediction variance collapsed)
  2 — runtime error (Bigtable unavailable, artifact missing, etc.)

Environment variables (alternatives to CLI flags):
    GCP_PROJECT_ID               -- --project
    BIGTABLE_INSTANCE_ID         -- --bigtable-instance
    GOOGLE_APPLICATION_CREDENTIALS -- --sa-key
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
_REPO_ROOT = Path(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from phase5.monitoring.drift_monitor import DriftMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("monitor")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SCIP drift monitoring: feature PSI + prediction variance check"
    )

    # ── GCP / Bigtable ────────────────────────────────────────────────────────
    p.add_argument(
        "--project",
        default=os.environ.get("GCP_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT", "")),
        help="GCP project ID (GCP_PROJECT_ID env var)",
    )
    p.add_argument(
        "--bigtable-instance",
        default=os.environ.get("BIGTABLE_INSTANCE_ID", "scip-feature-store"),
        help="Bigtable instance ID (default: scip-feature-store)",
    )
    p.add_argument(
        "--bigtable-table",
        default="inventory_features",
        help="Bigtable table ID (default: inventory_features)",
    )
    p.add_argument(
        "--sa-key",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Service account key path (scip-vertex-sa has monitoring.metricWriter role)",
    )

    # ── Local artifacts ───────────────────────────────────────────────────────
    p.add_argument(
        "--artifacts",
        default="phase4/artifacts",
        help="Directory containing feature_metadata.json (default: phase4/artifacts)",
    )
    p.add_argument(
        "--dataset",
        default="phase4/artifacts/dataset",
        help="Directory containing preprocessor.pkl (default: phase4/artifacts/dataset)",
    )

    # ── Monitoring options ────────────────────────────────────────────────────
    p.add_argument(
        "--no-predictions",
        action="store_true",
        help=(
            "Skip reading the Bigtable predictions family. "
            "Use this before running score.py for the first time."
        ),
    )
    p.add_argument(
        "--report-path",
        default=None,
        help=(
            "Path to write a JSON drift report "
            "(e.g. phase5/artifacts/drift_report.json). "
            "If omitted, report is printed to console only."
        ),
    )
    p.add_argument(
        "--emit-cloud-monitoring",
        action="store_true",
        help=(
            "Emit PSI and variance metrics to Google Cloud Monitoring. "
            "Requires google-cloud-monitoring>=2.18.0. "
            "The scip-vertex-sa has the required monitoring.metricWriter role."
        ),
    )

    # ── Control ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Set log level to DEBUG",
    )
    p.add_argument(
        "--exit-zero",
        action="store_true",
        help=(
            "Always exit 0 even when retraining is recommended. "
            "Useful in CI pipelines that should not fail on drift alerts."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.project:
        logger.error("Missing required --project flag or GCP_PROJECT_ID env var")
        sys.exit(2)

    # Verify local artifacts exist before touching Bigtable.
    artifacts_dir = Path(args.artifacts)
    dataset_dir   = Path(args.dataset)
    _preflight_check(artifacts_dir, dataset_dir)

    logger.info(
        "Starting drift monitor:\n"
        "  project          : %s\n"
        "  bigtable         : %s / %s\n"
        "  artifacts        : %s\n"
        "  read_predictions : %s\n"
        "  report_path      : %s\n"
        "  cloud_monitoring : %s",
        args.project,
        args.bigtable_instance, args.bigtable_table,
        dataset_dir,
        not args.no_predictions,
        args.report_path or "(not saved)",
        args.emit_cloud_monitoring,
    )

    try:
        with DriftMonitor(
            project_id       = args.project,
            instance_id      = args.bigtable_instance,
            artifacts_dir    = artifacts_dir,
            dataset_dir      = dataset_dir,
            table_id         = args.bigtable_table,
            credentials_path = args.sa_key,
        ) as monitor:
            report = monitor.run(
                read_predictions      = not args.no_predictions,
                report_path           = args.report_path,
                emit_cloud_monitoring = args.emit_cloud_monitoring,
            )
    except FileNotFoundError as exc:
        logger.error("Artifact missing: %s", exc)
        sys.exit(2)
    except Exception as exc:
        logger.exception("Drift monitor failed: %s", exc)
        sys.exit(2)

    if args.exit_zero:
        sys.exit(0)

    sys.exit(1 if report.requires_retraining else 0)


def _preflight_check(artifacts_dir: Path, dataset_dir: Path) -> None:
    required = [
        (dataset_dir   / "preprocessor.pkl",      "preprocessor.pkl"),
        (artifacts_dir / "feature_metadata.json", "feature_metadata.json"),
    ]
    missing = [(path, name) for path, name in required if not path.exists()]
    if missing:
        for path, name in missing:
            logger.error("Required artifact not found: %s  (expected at %s)", name, path)
        logger.error("Run phase4/scripts/train.py to generate Phase 4 artifacts.")
        sys.exit(2)

    for path, name in required:
        logger.info("Artifact verified: %s  (%d bytes)", path, path.stat().st_size)


if __name__ == "__main__":
    main()
