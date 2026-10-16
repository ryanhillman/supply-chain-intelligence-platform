"""
ModelPromoter — champion/challenger comparison and traffic-split promotion.

Flow
----
1. List deployed models on the Vertex AI endpoint.
2. Identify the champion (traffic > 0%) and the most recently deployed
   challenger (traffic == 0%).
3. Download evaluation_test.json from each model's GCS artifact URI.
4. Compare metrics: challenger must beat champion on ALL THREE —
     AUPRC    (higher is better)
     AUC-ROC  (higher is better)
     Brier    (lower is better)
5. If dry_run: print comparison and exit without mutating state.
6. If all three criteria pass: update endpoint traffic split to
   {challenger: 100%, champion: 0%}.
7. If any criterion fails: exit non-zero. The champion remains live.

The traffic-split update uses the Vertex AI v1 gapic client (not the high-level
SDK) because aiplatform.Endpoint does not expose a direct update_traffic_split()
method in all SDK versions. The gapic PATCH call is the canonical approach.

Distribution-shift note (ADR-005 Decision 7)
--------------------------------------------
A model trained on simulated data and then evaluated on real Bronze data WILL
fail the PSI check in drift_monitor.py before reaching this promotion gate.
The metrics comparison here is a secondary gate for ML quality, not a substitute
for the distribution-shift check.

Required packages:
    google-cloud-aiplatform>=1.38.0
    google-cloud-storage>=2.16.0
    protobuf>=4.21.0
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google.cloud import storage as gcs_storage
from google.protobuf import field_mask_pb2

logger = logging.getLogger(__name__)

# Promotion requires improvement on ALL three metrics.
# Brier is "lower is better"; AUPRC and AUC-ROC are "higher is better".
_HIGHER_IS_BETTER = ("auprc", "auc_roc")
_LOWER_IS_BETTER  = ("brier",)


@dataclass
class DeployedModelInfo:
    deployed_model_id: str
    model_resource_name: str
    traffic_pct: int
    create_time: datetime
    artifact_uri: str
    test_metrics: dict  # {"auprc": float, "auc_roc": float, "brier": float}


class ModelPromoter:
    """
    Parameters
    ----------
    project_id:
        GCP project ID.
    region:
        GCP region (e.g. "us-central1").
    endpoint_resource_name:
        Full Vertex AI endpoint resource name:
        "projects/{project}/locations/{region}/endpoints/{endpoint_id}"
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
        self.project_id              = project_id
        self.region                  = region
        self.endpoint_resource_name  = endpoint_resource_name

        if credentials_path:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            aiplatform.init(project=project_id, location=region, credentials=creds)
            self._storage_client = gcs_storage.Client(project=project_id, credentials=creds)
            self._gapic_client   = aiplatform_v1.EndpointServiceClient(
                credentials=creds,
                client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"},
            )
        else:
            aiplatform.init(project=project_id, location=region)
            self._storage_client = gcs_storage.Client(project=project_id)
            self._gapic_client   = aiplatform_v1.EndpointServiceClient(
                client_options={"api_endpoint": f"{region}-aiplatform.googleapis.com"},
            )

    def promote(self, dry_run: bool = False) -> bool:
        """
        Compare champion vs challenger and promote if challenger is better.

        Parameters
        ----------
        dry_run:
            If True, prints the comparison and returns the promotion decision
            without modifying the endpoint.

        Returns
        -------
        True if the challenger was promoted (or would be promoted in dry_run).
        False if the challenger did not beat the champion.

        Raises
        ------
        ValueError
            If there is no deployed champion, or no deployed challenger.
        """
        champion, challenger = self._find_champion_and_challenger()
        self._print_deployed_summary(champion, challenger)

        passes, summary = self._compare_metrics(champion, challenger)
        self._print_metric_comparison(champion, challenger, summary)

        if not passes:
            logger.warning(
                "PROMOTION BLOCKED — challenger did not beat champion on all metrics. "
                "Champion remains at 100%% traffic."
            )
            return False

        if dry_run:
            logger.info(
                "[dry-run] Challenger WOULD be promoted to 100%% traffic. "
                "Re-run without --dry-run to apply."
            )
            return True

        self._apply_traffic_split(
            champion_id=champion.deployed_model_id,
            challenger_id=challenger.deployed_model_id,
        )

        logger.info(
            "PROMOTED — challenger %s is now champion at 100%% traffic. "
            "Former champion %s is at 0%% (retained for rollback).",
            challenger.deployed_model_id,
            champion.deployed_model_id,
        )
        return True

    # ── Internals ─────────────────────────────────────────────────────────────

    def _find_champion_and_challenger(
        self,
    ) -> tuple[DeployedModelInfo, DeployedModelInfo]:
        endpoint = aiplatform.Endpoint(endpoint_name=self.endpoint_resource_name)
        traffic_split = dict(endpoint.traffic_split or {})

        if not traffic_split:
            raise ValueError(
                "No models are deployed on this endpoint. "
                "Run model_serving/scripts/deploy.py first."
            )

        champions   = [dm_id for dm_id, pct in traffic_split.items() if pct > 0]
        challengers = [dm_id for dm_id, pct in traffic_split.items() if pct == 0]

        if not champions:
            raise ValueError(
                "No champion found (no model at > 0%% traffic). "
                "For the first deployment, use deploy.py --traffic 100 directly."
            )
        if not challengers:
            raise ValueError(
                "No challenger found (no model at 0%% traffic). "
                "Deploy a challenger first with deploy.py --traffic 0."
            )
        if len(champions) > 1:
            raise ValueError(
                f"Multiple champions found: {champions}. "
                "Traffic split must have exactly one model at > 0%%."
            )

        # Build DeployedModelInfo for each
        proto_models = {
            dm.id: dm
            for dm in endpoint.gca_resource.deployed_models
        }

        champion_id   = champions[0]
        # If multiple challengers: pick the most recently deployed (latest create_time)
        challenger_id = max(
            challengers,
            key=lambda dm_id: proto_models[dm_id].create_time.timestamp()
            if dm_id in proto_models else 0,
        )

        if len(challengers) > 1:
            logger.warning(
                "%d challengers found. Selecting most recently deployed: %s. "
                "Consider undeploying older challengers.",
                len(challengers), challenger_id,
            )

        champion   = self._build_info(champion_id,   proto_models, traffic_split)
        challenger = self._build_info(challenger_id, proto_models, traffic_split)

        return champion, challenger

    def _build_info(
        self,
        deployed_model_id: str,
        proto_models: dict,
        traffic_split: dict[str, int],
    ) -> DeployedModelInfo:
        proto = proto_models.get(deployed_model_id)
        if proto is None:
            raise ValueError(
                f"Deployed model {deployed_model_id} not found in endpoint proto. "
                "The endpoint state may be stale — refresh and retry."
            )

        model_obj   = aiplatform.Model(model_name=proto.model)
        artifact_uri = model_obj.artifact_uri

        test_metrics = self._load_test_metrics(artifact_uri)

        return DeployedModelInfo(
            deployed_model_id=deployed_model_id,
            model_resource_name=proto.model,
            traffic_pct=traffic_split.get(deployed_model_id, 0),
            create_time=datetime.fromtimestamp(
                proto.create_time.timestamp(), tz=timezone.utc
            ),
            artifact_uri=artifact_uri,
            test_metrics=test_metrics,
        )

    def _load_test_metrics(self, artifact_uri: str) -> dict:
        """
        Download evaluation_test.json from the model's GCS artifact URI.
        Returns {"auprc": float, "auc_roc": float, "brier": float}.
        """
        path_without_scheme = artifact_uri.removeprefix("gs://")
        bucket_name, _, prefix = path_without_scheme.partition("/")
        blob_name = prefix.rstrip("/") + "/evaluation_test.json"

        bucket = self._storage_client.bucket(bucket_name)
        blob   = bucket.blob(blob_name)

        if not blob.exists():
            raise FileNotFoundError(
                f"evaluation_test.json not found at {artifact_uri}. "
                "Re-register the model with register_model.py to include it."
            )

        data = json.loads(blob.download_as_text())
        metrics = data.get("model", {})

        required = {"auprc", "auc_roc", "brier"}
        missing  = required - set(metrics)
        if missing:
            raise ValueError(
                f"evaluation_test.json is missing keys: {missing}. "
                f"Found: {list(metrics.keys())}"
            )

        return {k: float(metrics[k]) for k in required}

    def _compare_metrics(
        self,
        champion: DeployedModelInfo,
        challenger: DeployedModelInfo,
    ) -> tuple[bool, dict[str, dict]]:
        """
        Returns (all_pass, per_metric_summary).
        all_pass is True only if challenger beats champion on ALL metrics.
        """
        summary: dict[str, dict] = {}
        all_pass = True

        for metric in _HIGHER_IS_BETTER:
            champ_val = champion.test_metrics[metric]
            chal_val  = challenger.test_metrics[metric]
            delta     = chal_val - champ_val
            passes    = delta > 0
            if not passes:
                all_pass = False
            summary[metric] = {
                "champion":   champ_val,
                "challenger": chal_val,
                "delta":      delta,
                "passes":     passes,
                "criterion":  "higher is better",
            }

        for metric in _LOWER_IS_BETTER:
            champ_val = champion.test_metrics[metric]
            chal_val  = challenger.test_metrics[metric]
            delta     = chal_val - champ_val
            passes    = delta < 0
            if not passes:
                all_pass = False
            summary[metric] = {
                "champion":   champ_val,
                "challenger": chal_val,
                "delta":      delta,
                "passes":     passes,
                "criterion":  "lower is better",
            }

        return all_pass, summary

    def _apply_traffic_split(
        self,
        champion_id: str,
        challenger_id: str,
    ) -> None:
        """
        Flip traffic to 100% challenger / 0% all others via the gapic client.

        Uses a targeted field mask update (PATCH traffic_split only) to avoid
        inadvertently touching other endpoint settings.
        """
        # Get current traffic split to preserve all deployed model IDs
        endpoint     = aiplatform.Endpoint(endpoint_name=self.endpoint_resource_name)
        current_ids  = set(endpoint.traffic_split.keys())
        new_split    = {dm_id: 0 for dm_id in current_ids}
        new_split[challenger_id] = 100

        endpoint_proto = aiplatform_v1.types.Endpoint(
            name=self.endpoint_resource_name,
            traffic_split=new_split,
        )
        update_mask = field_mask_pb2.FieldMask(paths=["traffic_split"])

        self._gapic_client.update_endpoint(
            request=aiplatform_v1.UpdateEndpointRequest(
                endpoint=endpoint_proto,
                update_mask=update_mask,
            )
        )

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _print_deployed_summary(
        self,
        champion: DeployedModelInfo,
        challenger: DeployedModelInfo,
    ) -> None:
        sep = "=" * 68
        print(f"\n{sep}")
        print("  Champion/Challenger Promotion Check")
        print(sep)
        print(f"  Champion   : {champion.deployed_model_id}  ({champion.traffic_pct}% traffic)")
        print(f"               {champion.model_resource_name}")
        print(f"  Challenger : {challenger.deployed_model_id}  ({challenger.traffic_pct}% traffic)")
        print(f"               {challenger.model_resource_name}")
        print(f"               deployed {challenger.create_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    def _print_metric_comparison(
        self,
        champion: DeployedModelInfo,
        challenger: DeployedModelInfo,
        summary: dict[str, dict],
    ) -> None:
        print()
        print(f"  {'Metric':<10} {'Champion':>10} {'Challenger':>12} {'Delta':>10}  Pass")
        print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10}  ----")
        for metric, data in summary.items():
            sign   = "+" if data["delta"] > 0 else ""
            status = "YES" if data["passes"] else "NO "
            print(
                f"  {metric:<10} {data['champion']:>10.4f} {data['challenger']:>12.4f} "
                f"{sign}{data['delta']:>9.4f}  {status}"
            )
        all_pass = all(d["passes"] for d in summary.values())
        verdict  = "PROMOTE" if all_pass else "BLOCKED"
        print(f"\n  Verdict: {verdict}")
        print("=" * 68)
