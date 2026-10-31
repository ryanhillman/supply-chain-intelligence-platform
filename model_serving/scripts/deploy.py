"""
deploy.py — CLI for EndpointDeployer and ModelPromoter.

Subcommands:

    deploy     — deploy the latest registered model to the Vertex AI endpoint.
                 Reads model_resource_name from model_serving/artifacts/registration_info.json.
                 Defaults to 0% traffic (challenger pattern).

    smoke-test — score a few Bigtable entities through the endpoint.

    promote    — compare champion vs challenger and promote the challenger if it
                 beats the champion on AUPRC, AUC-ROC, and Brier score.

    undeploy   — undeploy every model from the endpoint (the endpoint resource
                 stays). Deployed replicas bill by the hour with or without
                 traffic, so run this between demo sessions.

Usage
-----
    # First deployment (no champion yet):
    python model_serving/scripts/deploy.py deploy \\
        --project your-gcp-project-id \\
        --region  us-central1 \\
        --endpoint $env:VERTEX_AI_ENDPOINT_RESOURCE_NAME \\
        --sa-key  infrastructure/gcp/keys/scip-vertex-sa.json \\
        --traffic 100

    # Challenger deployment (run smoke test first):
    python model_serving/scripts/deploy.py deploy \\
        --project your-gcp-project-id \\
        --region  us-central1 \\
        --endpoint $env:VERTEX_AI_ENDPOINT_RESOURCE_NAME \\
        --sa-key  infrastructure/gcp/keys/scip-vertex-sa.json \\
        --traffic 0

    # Smoke test after challenger deployment:
    python model_serving/scripts/deploy.py smoke-test \\
        --project your-gcp-project-id \\
        --region  us-central1 \\
        --endpoint $env:VERTEX_AI_ENDPOINT_RESOURCE_NAME \\
        --bigtable-instance scip-feature-store \\
        --sa-key  infrastructure/gcp/keys/scip-vertex-sa.json

    # Promote challenger to champion:
    python model_serving/scripts/deploy.py promote \\
        --project your-gcp-project-id \\
        --region  us-central1 \\
        --endpoint $env:VERTEX_AI_ENDPOINT_RESOURCE_NAME \\
        --sa-key  infrastructure/gcp/keys/scip-vertex-sa.json \\
        --dry-run   # preview the comparison before committing

    # Between demo sessions — stop paying for idle replicas:
    python model_serving/scripts/deploy.py undeploy \\
        --project your-gcp-project-id \\
        --region  us-central1 \\
        --endpoint $env:VERTEX_AI_ENDPOINT_RESOURCE_NAME \\
        --sa-key  infrastructure/gcp/keys/scip-vertex-sa.json

Environment variables:
    GCP_PROJECT_ID                   -- --project
    GCP_REGION                       -- --region
    VERTEX_AI_ENDPOINT_RESOURCE_NAME -- --endpoint
    GOOGLE_APPLICATION_CREDENTIALS   -- --sa-key
    BIGTABLE_INSTANCE_ID             -- --bigtable-instance
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from model_serving.serving.deploy_endpoint import EndpointDeployer
from model_serving.registry.promote_model  import ModelPromoter
from model_serving.serving.smoke_test      import SmokeTest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("deploy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deploy, smoke-test, promote, or undeploy the SCIP shortage predictor"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── Common flags ───────────────────────────────────────────────────────────
    def _add_common(sp) -> None:
        sp.add_argument(
            "--project",
            default=os.environ.get("GCP_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT", "")),
        )
        sp.add_argument(
            "--region",
            default=os.environ.get("GCP_REGION", "us-central1"),
        )
        sp.add_argument(
            "--endpoint",
            default=os.environ.get("VERTEX_AI_ENDPOINT_RESOURCE_NAME", ""),
            help=(
                "Full Vertex AI endpoint resource name. "
                "Use the 'vertex_ai_endpoint_resource_name' Terraform output."
            ),
        )
        sp.add_argument(
            "--sa-key",
            default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        )

    # ── deploy subcommand ─────────────────────────────────────────────────────
    deploy_p = sub.add_parser("deploy", help="Deploy model to the Vertex AI endpoint")
    _add_common(deploy_p)
    deploy_p.add_argument(
        "--model-resource-name",
        default=None,
        help=(
            "Vertex AI model resource name "
            "(projects/.../locations/.../models/...). "
            "Defaults to the value in model_serving/artifacts/registration_info.json."
        ),
    )
    deploy_p.add_argument(
        "--traffic",
        type=int,
        default=0,
        choices=range(0, 101),
        metavar="0-100",
        help="Traffic percentage for this deployment (default: 0 = challenger)",
    )
    deploy_p.add_argument(
        "--machine-type",
        default="n1-standard-2",
        help="Vertex AI machine type (default: n1-standard-2)",
    )
    deploy_p.add_argument(
        "--min-replicas",
        type=int,
        default=1,
    )
    deploy_p.add_argument(
        "--max-replicas",
        type=int,
        default=1,
    )
    deploy_p.add_argument(
        "--service-account",
        default=os.environ.get("VERTEX_SA_EMAIL"),
        help=(
            "Service account email for the serving container "
            "(e.g. scip-vertex-sa@{project}.iam.gserviceaccount.com). "
            "Required: the default Vertex AI compute SA lacks GCS bucket read access. "
            "Use the 'vertex_sa_email' Terraform output or set VERTEX_SA_EMAIL."
        ),
    )

    # ── smoke-test subcommand ─────────────────────────────────────────────────
    smoke_p = sub.add_parser("smoke-test", help="Smoke-test the deployed endpoint")
    _add_common(smoke_p)
    smoke_p.add_argument(
        "--bigtable-instance",
        default=os.environ.get("BIGTABLE_INSTANCE_ID", ""),
        help="Bigtable instance ID (e.g. scip-feature-store). Optional; uses synthetic data if omitted.",
    )
    smoke_p.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of entities to score (default: 3)",
    )
    smoke_p.add_argument(
        "--artifacts",
        default="ml_training/artifacts",
    )

    # ── promote subcommand ────────────────────────────────────────────────────
    promote_p = sub.add_parser(
        "promote",
        help="Compare champion vs challenger and promote if challenger is better",
    )
    _add_common(promote_p)
    promote_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print comparison without updating endpoint traffic split",
    )
    promote_p.add_argument(
        "--dataset",
        default="ml_training/artifacts/dataset",
        help=(
            "Directory holding the challenger's test.parquet. Used to rescore both "
            "models on one holdout when their evaluation splits differ "
            "(default: ml_training/artifacts/dataset)"
        ),
    )

    # ── undeploy subcommand ───────────────────────────────────────────────────
    undeploy_p = sub.add_parser(
        "undeploy",
        help="Undeploy every model from the endpoint (keeps the endpoint resource)",
    )
    _add_common(undeploy_p)

    return p.parse_args()


def _require(args: argparse.Namespace, flag: str, env: str) -> None:
    val = getattr(args, flag.replace("-", "_"), None)
    if not val:
        logger.error("Missing required flag --%s or env var %s", flag, env)
        sys.exit(1)


def main() -> None:
    args = parse_args()

    # Validate required args shared by all subcommands
    _require(args, "project",  "GCP_PROJECT_ID")
    _require(args, "endpoint", "VERTEX_AI_ENDPOINT_RESOURCE_NAME")

    if args.command == "deploy":
        if not getattr(args, "service_account", None):
            logger.warning(
                "No --service-account specified. The default Vertex AI compute SA "
                "lacks GCS bucket read access and the container will fail to start. "
                "Set VERTEX_SA_EMAIL or pass --service-account "
                "scip-vertex-sa@{project}.iam.gserviceaccount.com"
            )
        deployer = EndpointDeployer(
            project_id             = args.project,
            region                 = args.region,
            endpoint_resource_name = args.endpoint,
            credentials_path       = args.sa_key,
        )
        info = deployer.deploy(
            model_resource_name = args.model_resource_name,
            traffic_pct         = args.traffic,
            machine_type        = args.machine_type,
            min_replicas        = args.min_replicas,
            max_replicas        = args.max_replicas,
            service_account     = args.service_account,
        )
        print("\n" + "=" * 68)
        print("  Deployment Info")
        print("=" * 68)
        print(json.dumps(info, indent=2))
        print("=" * 68)
        if info["traffic_pct"] == 0:
            print(
                f"\nChallenger is live at 0% traffic.\n"
                f"1. Run smoke test:\n"
                f"   python model_serving/scripts/deploy.py smoke-test --project {args.project} "
                f"--endpoint {args.endpoint} --bigtable-instance scip-feature-store\n"
                f"2. Promote if smoke test passes:\n"
                f"   python model_serving/scripts/deploy.py promote --project {args.project} "
                f"--endpoint {args.endpoint} --dry-run"
            )

    elif args.command == "smoke-test":
        smoke = SmokeTest(
            project_id             = args.project,
            region                 = args.region,
            endpoint_resource_name = args.endpoint,
            artifacts_dir          = args.artifacts,
            credentials_path       = args.sa_key,
        )
        passed = smoke.run(
            n_samples   = args.n_samples,
            instance_id = args.bigtable_instance or None,
        )
        sys.exit(0 if passed else 1)

    elif args.command == "promote":
        promoter = ModelPromoter(
            project_id             = args.project,
            region                 = args.region,
            endpoint_resource_name = args.endpoint,
            credentials_path       = args.sa_key,
            dataset_dir            = args.dataset,
        )
        promoted = promoter.promote(dry_run=args.dry_run)
        sys.exit(0 if promoted else 1)

    elif args.command == "undeploy":
        deployer = EndpointDeployer(
            project_id             = args.project,
            region                 = args.region,
            endpoint_resource_name = args.endpoint,
            credentials_path       = args.sa_key,
        )
        removed = deployer.undeploy_all()
        if removed:
            print(f"\nUndeployed {len(removed)} model(s) from the endpoint: {removed}")
        else:
            print("\nNothing was deployed on the endpoint.")
        print("The endpoint resource is kept (Terraform-managed); redeploy with `deploy.py deploy`.")


if __name__ == "__main__":
    main()
