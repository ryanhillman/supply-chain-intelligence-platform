"""
ModelRegistrar — uploads Phase 4 artifacts to a versioned GCS path and
registers the model in Vertex AI Model Registry.

GCS layout after registration:
  gs://{bucket}/models/shortage-predictor/{version}/
    model.onnx            — ONNX model graph (required by serving container)
    model.onnx.data       — ONNX external weights sidecar (present when >~20 MB)
    feature_metadata.json — vocab mappings + feature list (used by batch_scorer)
    preprocessor.pkl      — StandardScaler (used by batch_scorer, NOT the container)
    evaluation_test.json  — held-out test metrics for promote_model.py comparison
    registration_info.json — Vertex AI resource name + registration timestamp

The serving container's MODEL_GCS_PATH environment variable is set to the version
directory during Vertex AI Model.upload(), so the container knows where to
download model.onnx at startup.

Required packages (pip install):
    google-cloud-aiplatform>=1.38.0
    google-cloud-storage>=2.16.0
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage as gcs_storage
from google.cloud import aiplatform

logger = logging.getLogger(__name__)

# Files uploaded from the artifacts directory (model.pt is intentionally omitted
# — the serving container uses model.onnx, and model.pt is 50× larger).
_ARTIFACTS_TO_UPLOAD: list[str] = [
    "model.onnx",
    "model.onnx.data",       # present only for larger models; skipped if absent
    "feature_metadata.json",
    "evaluation_test.json",
    "training_metrics.json",
]

# Uploaded from the dataset directory (separate from artifacts_dir).
_DATASET_ARTIFACTS_TO_UPLOAD: list[str] = [
    "preprocessor.pkl",
]

_GCS_MODEL_PREFIX = "models/shortage-predictor"


class ModelRegistrar:
    """
    Uploads Phase 4 artifacts to GCS and registers the model with Vertex AI.

    Parameters
    ----------
    project_id:
        GCP project ID.
    region:
        GCP region where the Vertex AI endpoint was created (e.g. "us-central1").
    bucket_name:
        GCS data lake bucket name (GCS_BUCKET_NAME env var).
    credentials_path:
        Path to service account JSON key. None = application default credentials.
    """

    def __init__(
        self,
        project_id: str,
        region: str,
        bucket_name: str,
        credentials_path: str | None = None,
    ) -> None:
        self.project_id   = project_id
        self.region       = region
        self.bucket_name  = bucket_name

        if credentials_path:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self._storage_client = gcs_storage.Client(project=project_id, credentials=creds)
            aiplatform.init(project=project_id, location=region, credentials=creds)
        else:
            self._storage_client = gcs_storage.Client(project=project_id)
            aiplatform.init(project=project_id, location=region)

    def register(
        self,
        artifacts_dir: str | Path,
        dataset_dir: str | Path,
        image_uri: str,
        version: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Upload artifacts to GCS and register with Vertex AI Model Registry.

        Parameters
        ----------
        artifacts_dir:
            Local directory containing model.onnx, feature_metadata.json,
            evaluation_test.json (output of phase4/scripts/train.py).
        dataset_dir:
            Local directory containing preprocessor.pkl
            (output of phase4/scripts/build_dataset.py).
        image_uri:
            Full Artifact Registry URI for the serving container image, e.g.
            us-central1-docker.pkg.dev/{project}/scip-serving/shortage-predictor-serving:latest
        version:
            Version tag, e.g. "v20260620_181523". Defaults to v{UTC timestamp}.
        dry_run:
            If True, log what would be uploaded/registered but do nothing.

        Returns
        -------
        dict with keys: version, gcs_artifact_uri, model_resource_name,
                        image_uri, registered_at
        """
        artifacts_dir = Path(artifacts_dir)
        dataset_dir   = Path(dataset_dir)
        version       = version or _make_version()

        gcs_prefix       = f"{_GCS_MODEL_PREFIX}/{version}"
        gcs_artifact_uri = f"gs://{self.bucket_name}/{gcs_prefix}"

        logger.info("Registering model version %s", version)
        logger.info("GCS artifact URI: %s", gcs_artifact_uri)
        logger.info("Serving container: %s", image_uri)

        self._check_version_not_exists(gcs_prefix)

        if dry_run:
            logger.info("[dry-run] Would upload artifacts and register with Vertex AI.")
            return {"version": version, "gcs_artifact_uri": gcs_artifact_uri, "dry_run": True}

        # Upload artifacts
        uploaded = self._upload_artifacts(
            artifacts_dir, dataset_dir, gcs_prefix
        )
        logger.info("Uploaded %d files to %s", len(uploaded), gcs_artifact_uri)

        # Register with Vertex AI Model Registry
        model = self._register_with_vertex(
            gcs_artifact_uri=gcs_artifact_uri,
            image_uri=image_uri,
            version=version,
        )
        logger.info("Model registered: %s", model.resource_name)

        # Persist registration info locally and to GCS
        registered_at = datetime.now(timezone.utc).isoformat()
        info = {
            "version":             version,
            "gcs_artifact_uri":    gcs_artifact_uri,
            "model_resource_name": model.resource_name,
            "image_uri":           image_uri,
            "registered_at":       registered_at,
        }

        # Embed test metrics for promote_model.py (avoids a separate GCS download)
        eval_path = artifacts_dir / "evaluation_test.json"
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
            info["test_metrics"] = eval_data.get("model", {})

        self._save_registration_info(info, gcs_prefix)
        return info

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_version_not_exists(self, gcs_prefix: str) -> None:
        bucket = self._storage_client.bucket(self.bucket_name)
        existing = list(bucket.list_blobs(prefix=f"{gcs_prefix}/", max_results=1))
        if existing:
            raise FileExistsError(
                f"Model version already exists at gs://{self.bucket_name}/{gcs_prefix}/. "
                "Use a new version tag or delete the existing one first."
            )

    def _upload_artifacts(
        self,
        artifacts_dir: Path,
        dataset_dir: Path,
        gcs_prefix: str,
    ) -> list[str]:
        bucket = self._storage_client.bucket(self.bucket_name)
        uploaded: list[str] = []

        for fname in _ARTIFACTS_TO_UPLOAD:
            src = artifacts_dir / fname
            if not src.exists():
                if fname == "model.onnx.data":
                    logger.debug("model.onnx.data not present — single-file ONNX export")
                    continue
                logger.warning("Expected artifact not found, skipping: %s", src)
                continue
            blob = bucket.blob(f"{gcs_prefix}/{fname}")
            blob.upload_from_filename(str(src))
            logger.info("  Uploaded %-30s  (%s bytes)", fname, src.stat().st_size)
            uploaded.append(fname)

        for fname in _DATASET_ARTIFACTS_TO_UPLOAD:
            src = dataset_dir / fname
            if not src.exists():
                logger.warning("Dataset artifact not found, skipping: %s", src)
                continue
            blob = bucket.blob(f"{gcs_prefix}/{fname}")
            blob.upload_from_filename(str(src))
            logger.info("  Uploaded %-30s  (%s bytes)", fname, src.stat().st_size)
            uploaded.append(fname)

        return uploaded

    def _register_with_vertex(
        self,
        gcs_artifact_uri: str,
        image_uri: str,
        version: str,
    ) -> aiplatform.Model:
        return aiplatform.Model.upload(
            display_name=f"shortage-predictor-{version}",
            description=(
                f"ShortagePredictor ONNX model, version {version}. "
                "Trained on Phase 4 dataset (simulated Bronze data). "
                "See ADR-005 for distribution-shift caveats."
            ),
            artifact_uri=gcs_artifact_uri,
            serving_container_image_uri=image_uri,
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_ports=[8080],
            # MODEL_GCS_PATH tells the container where to download model.onnx at startup.
            serving_container_environment_variables={
                "MODEL_GCS_PATH": gcs_artifact_uri,
                "PYTHONIOENCODING": "utf-8",
            },
            labels={
                "phase":   "5",
                "project": "scip",
            },
            sync=True,
        )

    def _save_registration_info(self, info: dict, gcs_prefix: str) -> None:
        payload = json.dumps(info, indent=2)

        # Save locally (picked up by deploy_endpoint.py as the default model source)
        local_path = Path("phase5/artifacts/registration_info.json")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(payload)
        logger.info("Registration info saved locally: %s", local_path)

        # Save to GCS alongside model files
        bucket = self._storage_client.bucket(self.bucket_name)
        blob = bucket.blob(f"{gcs_prefix}/registration_info.json")
        blob.upload_from_string(payload, content_type="application/json")
        logger.info(
            "Registration info saved to gs://%s/%s/registration_info.json",
            self.bucket_name, gcs_prefix,
        )


def _make_version() -> str:
    return datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
