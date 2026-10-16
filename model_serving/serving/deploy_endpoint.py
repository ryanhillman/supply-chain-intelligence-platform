"""
EndpointDeployer — deploys a registered Vertex AI model to the SCIP endpoint.

Design
------
- Reads the model to deploy from local model_serving/artifacts/registration_info.json
  (written by register_model.py) or from --model-resource-name CLI arg.
- Deploys at 0% traffic by default (challenger pattern per ADR-005 Decision 6).
- Records the deployed model ID in model_serving/artifacts/deployment_info.json and
  in GCS for traceability.

Usage
-----
    # First deployment (no champion yet): set traffic to 100
    python model_serving/scripts/deploy.py --traffic 100

    # Challenger deployment (champion already live): keep at 0, then promote
    python model_serving/scripts/deploy.py --traffic 0
    python model_serving/scripts/deploy.py promote   # after smoke test passes

Required packages:
    google-cloud-aiplatform>=1.38.0
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import aiplatform
from google.cloud import storage as gcs_storage

logger = logging.getLogger(__name__)

_DEFAULT_MACHINE_TYPE = "n1-standard-2"
_DEFAULT_MIN_REPLICAS = 1
_DEFAULT_MAX_REPLICAS = 1
_LOCAL_DEPLOYMENT_INFO = Path("model_serving/artifacts/deployment_info.json")
_LOCAL_REGISTRATION_INFO = Path("model_serving/artifacts/registration_info.json")


class EndpointDeployer:
    """
    Parameters
    ----------
    project_id:
        GCP project ID.
    region:
        GCP region (e.g. "us-central1").
    endpoint_resource_name:
        Full Vertex AI endpoint resource name from the Terraform output
        `vertex_ai_endpoint_resource_name`.
    credentials_path:
        Path to service account JSON key. None = application default credentials.
    """

    def __init__(
        self,
        project_id: str,
        region: str,
        endpoint_resource_name: str,
        credentials_path: str | None = None,
    ) -> None:
        self.project_id             = project_id
        self.region                 = region
        self.endpoint_resource_name = endpoint_resource_name

        if credentials_path:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            aiplatform.init(project=project_id, location=region, credentials=creds)
            self._storage_client = gcs_storage.Client(project=project_id, credentials=creds)
        else:
            aiplatform.init(project=project_id, location=region)
            self._storage_client = gcs_storage.Client(project=project_id)

    def deploy(
        self,
        model_resource_name: str | None = None,
        traffic_pct: int = 0,
        machine_type: str = _DEFAULT_MACHINE_TYPE,
        min_replicas: int = _DEFAULT_MIN_REPLICAS,
        max_replicas: int = _DEFAULT_MAX_REPLICAS,
        service_account: str | None = None,
    ) -> dict:
        """
        Deploy a model version to the Vertex AI endpoint.

        Parameters
        ----------
        model_resource_name:
            Vertex AI model resource name from registration_info.json.
            If None, reads from model_serving/artifacts/registration_info.json.
        traffic_pct:
            Traffic percentage for this deployment. Default 0 (challenger).
            Use 100 for the first deployment when no champion exists.
        machine_type:
            Vertex AI machine type. n1-standard-2 is the minimum for CPU serving.
        min_replicas / max_replicas:
            Replica count. Keep at 1/1 for batch-only scoring; scale up for
            real-time latency requirements.
        service_account:
            SA email for the serving container (e.g. scip-vertex-sa@...).
            If None, Vertex AI uses the project default compute SA.

        Returns
        -------
        dict with keys: deployed_model_id, model_resource_name,
                        endpoint_resource_name, traffic_pct, deployed_at
        """
        model_resource_name = model_resource_name or self._load_model_resource_name()
        reg_info = self._load_registration_info()

        logger.info("Deploying model: %s", model_resource_name)
        logger.info("Endpoint: %s", self.endpoint_resource_name)
        logger.info("Traffic: %d%%  Machine: %s  Replicas: %d-%d",
                    traffic_pct, machine_type, min_replicas, max_replicas)

        endpoint = aiplatform.Endpoint(endpoint_name=self.endpoint_resource_name)

        # Snapshot traffic split before deploy to identify new deployed model ID.
        split_before = set(endpoint.traffic_split.keys()) if endpoint.traffic_split else set()

        model = aiplatform.Model(model_name=model_resource_name)

        deploy_kwargs: dict = {
            "endpoint":                       endpoint,
            "deployed_model_display_name":    f"shortage-predictor-{reg_info.get('version', 'unknown')}",
            "machine_type":                   machine_type,
            "min_replica_count":              min_replicas,
            "max_replica_count":              max_replicas,
            "traffic_percentage":             traffic_pct,
            "sync":                           True,
        }
        if service_account:
            deploy_kwargs["service_account"] = service_account

        endpoint_after = model.deploy(**deploy_kwargs)

        # Identify the newly created deployed model ID by diff.
        split_after = set(endpoint_after.traffic_split.keys())
        new_ids     = split_after - split_before
        if not new_ids:
            # Fallback: the ID with value == traffic_pct that wasn't there before.
            # This case can occur when traffic_pct matches an existing entry's value.
            logger.warning(
                "Could not identify new deployed model ID by diff. "
                "Check endpoint.traffic_split manually."
            )
            deployed_model_id = "unknown"
        else:
            deployed_model_id = next(iter(new_ids))

        deployed_at = datetime.now(timezone.utc).isoformat()

        info = {
            "deployed_model_id":      deployed_model_id,
            "model_resource_name":    model_resource_name,
            "endpoint_resource_name": self.endpoint_resource_name,
            "traffic_pct":            traffic_pct,
            "machine_type":           machine_type,
            "min_replicas":           min_replicas,
            "max_replicas":           max_replicas,
            "deployed_at":            deployed_at,
            "version":                reg_info.get("version", "unknown"),
        }

        self._save_deployment_info(info, reg_info)

        logger.info(
            "Deployment complete | deployed_model_id=%s | traffic=%d%%",
            deployed_model_id, traffic_pct,
        )
        if traffic_pct == 0:
            logger.info(
                "Challenger deployed at 0%% traffic. "
                "Run smoke_test.py, then promote_model.py to go live."
            )

        return info

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_registration_info(self) -> dict:
        if not _LOCAL_REGISTRATION_INFO.exists():
            raise FileNotFoundError(
                f"{_LOCAL_REGISTRATION_INFO} not found. "
                "Run model_serving/scripts/register.py first."
            )
        with open(_LOCAL_REGISTRATION_INFO) as f:
            return json.load(f)

    def _load_model_resource_name(self) -> str:
        info = self._load_registration_info()
        name = info.get("model_resource_name")
        if not name:
            raise ValueError(
                "model_resource_name not found in registration_info.json. "
                "Re-run register.py or pass --model-resource-name explicitly."
            )
        return name

    def _save_deployment_info(self, info: dict, reg_info: dict) -> None:
        payload = json.dumps(info, indent=2)

        _LOCAL_DEPLOYMENT_INFO.parent.mkdir(parents=True, exist_ok=True)
        _LOCAL_DEPLOYMENT_INFO.write_text(payload)
        logger.info("Deployment info saved locally: %s", _LOCAL_DEPLOYMENT_INFO)

        # Also persist to GCS alongside the model artifacts for auditability.
        gcs_uri = reg_info.get("gcs_artifact_uri", "")
        if gcs_uri:
            path_without_scheme = gcs_uri.removeprefix("gs://")
            bucket_name, _, prefix = path_without_scheme.partition("/")
            bucket = self._storage_client.bucket(bucket_name)
            blob   = bucket.blob(f"{prefix.rstrip('/')}/deployment_info.json")
            blob.upload_from_string(payload, content_type="application/json")
            logger.info("Deployment info saved to %s/deployment_info.json", gcs_uri)
