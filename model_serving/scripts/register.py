"""
register.py — CLI for ModelRegistrar.

Uploads Phase 4 artifacts to a versioned GCS path and registers the model
in Vertex AI Model Registry. Does NOT deploy to the endpoint.

Usage
-----
    python model_serving/scripts/register.py \\
        --bucket  your-gcp-project-id-scip-lake \\
        --project your-gcp-project-id \\
        --region  us-central1 \\
        --image-uri us-central1-docker.pkg.dev/your-gcp-project-id/scip-serving/shortage-predictor-serving:latest \\
        --sa-key  infrastructure/gcp/keys/scip-vertex-sa.json

    # Dry-run (no uploads, no Vertex AI calls):
    python model_serving/scripts/register.py ... --dry-run

Environment variables (alternatives to CLI flags):
    GCS_BUCKET_NAME              -- --bucket
    GCP_PROJECT_ID               -- --project
    GCP_REGION                   -- --region
    ARTIFACT_REGISTRY_URL        -- base URL; image URI = {URL}/shortage-predictor-serving:{image-tag}
    GOOGLE_APPLICATION_CREDENTIALS -- --sa-key

After successful registration, model_serving/artifacts/registration_info.json is written.
Pass that path to model_serving/scripts/deploy.py (or use the defaults).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from model_serving.registry.register_model import ModelRegistrar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("register")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Upload Phase 4 artifacts to GCS and register in Vertex AI Model Registry"
    )
    p.add_argument(
        "--bucket",
        default=os.environ.get("GCS_BUCKET_NAME", ""),
        help="GCS data lake bucket name (GCS_BUCKET_NAME env var)",
    )
    p.add_argument(
        "--project",
        default=os.environ.get("GCP_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT", "")),
        help="GCP project ID",
    )
    p.add_argument(
        "--region",
        default=os.environ.get("GCP_REGION", "us-central1"),
        help="GCP region (default: us-central1)",
    )
    p.add_argument(
        "--image-uri",
        default=os.environ.get("SERVING_IMAGE_URI", ""),
        help=(
            "Full Artifact Registry image URI, e.g. "
            "us-central1-docker.pkg.dev/{project}/scip-serving/shortage-predictor-serving:latest. "
            "Alternatively, set ARTIFACT_REGISTRY_URL and --image-tag."
        ),
    )
    p.add_argument(
        "--image-tag",
        default="latest",
        help=(
            "Image tag used when ARTIFACT_REGISTRY_URL is set and --image-uri is not. "
            "Image URI = {ARTIFACT_REGISTRY_URL}/shortage-predictor-serving:{tag}"
        ),
    )
    p.add_argument(
        "--artifacts",
        default="ml_training/artifacts",
        help="Local directory containing model.onnx, feature_metadata.json, evaluation_test.json",
    )
    p.add_argument(
        "--version",
        default=None,
        help="Version tag (e.g. v20260620_181523). Defaults to v{UTC timestamp}.",
    )
    p.add_argument(
        "--sa-key",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Path to service account JSON key",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be registered without uploading or calling Vertex AI",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate required arguments
    missing: list[str] = []
    if not args.bucket:
        missing.append("--bucket / GCS_BUCKET_NAME")
    if not args.project:
        missing.append("--project / GCP_PROJECT_ID")

    # Resolve image URI
    image_uri = args.image_uri
    if not image_uri:
        registry_url = os.environ.get("ARTIFACT_REGISTRY_URL", "")
        if registry_url:
            image_uri = f"{registry_url}/shortage-predictor-serving:{args.image_tag}"
        else:
            missing.append(
                "--image-uri or (ARTIFACT_REGISTRY_URL + --image-tag). "
                "ARTIFACT_REGISTRY_URL is the 'artifact_registry_url' Terraform output."
            )

    if missing:
        for m in missing:
            logger.error("Missing required argument: %s", m)
        sys.exit(1)

    # Pre-flight: verify local artifacts exist
    artifacts_dir = Path(args.artifacts)
    _check_local_artifacts(artifacts_dir)

    logger.info("Registering model version: %s", args.version or "(auto)")
    logger.info("GCS bucket: %s", args.bucket)
    logger.info("Project: %s / Region: %s", args.project, args.region)
    logger.info("Image URI: %s", image_uri)

    registrar = ModelRegistrar(
        project_id        = args.project,
        region            = args.region,
        bucket_name       = args.bucket,
        credentials_path  = args.sa_key,
    )

    info = registrar.register(
        artifacts_dir = artifacts_dir,
        image_uri     = image_uri,
        version       = args.version,
        dry_run       = args.dry_run,
    )

    print("\n" + "=" * 68)
    print("  Registration Complete")
    print("=" * 68)
    print(json.dumps(info, indent=2))
    print("=" * 68)
    print(
        f"\nNext step: deploy the model to the endpoint.\n"
        f"  python model_serving/scripts/deploy.py \\\n"
        f"      --project {args.project} \\\n"
        f"      --region  {args.region} \\\n"
        f"      --sa-key  {args.sa_key or 'infrastructure/gcp/keys/scip-vertex-sa.json'} \\\n"
        f"      --traffic 0    # challenger; use --traffic 100 for first deployment"
    )


def _check_local_artifacts(artifacts_dir: Path) -> None:
    required: list[Path] = [
        artifacts_dir / "model.onnx",
        artifacts_dir / "feature_metadata.json",
        artifacts_dir / "evaluation_test.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.error(
            "Required Phase 4 artifacts not found:\n%s",
            "\n".join(f"  {p}" for p in missing),
        )
        logger.error(
            "Run ml_training/scripts/train.py to generate them, then retry."
        )
        sys.exit(1)

    logger.info("Local artifacts verified:")
    for p in [artifacts_dir / "model.onnx", artifacts_dir / "feature_metadata.json",
              artifacts_dir / "evaluation_test.json"]:
        if p.exists():
            logger.info("  %s  (%d bytes)", p, p.stat().st_size)


if __name__ == "__main__":
    main()
