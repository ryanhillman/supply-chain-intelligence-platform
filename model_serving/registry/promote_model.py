"""
ModelPromoter — champion/challenger comparison and traffic-split promotion.

Flow
----
1. List deployed models on the Vertex AI endpoint.
2. Identify the champion (traffic > 0%) and the most recently deployed
   challenger (traffic == 0%).
3. Download evaluation_test.json from each model's GCS artifact URI. Each one
   carries the metrics and a sha256 of the test split they were computed on.
4. Put both models on one holdout. If the hashes match, the stored metrics are
   comparable and are used as-is. Otherwise — the normal case after a retrain
   on new data — download both models' model.onnx + feature_metadata.json and
   score both on the challenger's test split (dataset_dir/test.parquet, whose
   hash must equal the challenger's recorded hash) locally with onnxruntime.
5. Compare metrics: challenger must beat champion on ALL THREE —
     AUPRC    (higher is better)
     AUC-ROC  (higher is better)
     Brier    (lower is better)
6. If dry_run: print comparison and exit without mutating state.
7. If all three criteria pass: update endpoint traffic split to
   {challenger: 100%, champion: 0%}.
8. If any criterion fails: exit non-zero. The champion remains live.

Offline (`deploy.py promote --local`)
-------------------------------------
promote_local() runs steps 3-5 between two training runs' artifact
directories (train.py output: model.onnx, feature_metadata.json,
evaluation_test.json) and reports the verdict. Nothing is deployed; the
local run (local/README.md) serves whichever directory wins by pointing
MODEL_LOCAL_PATH at it.

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
    onnxruntime>=1.17.0      (local rescoring)
    pandas, numpy, scikit-learn (via ml_training.evaluation.metrics)
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
from google.cloud import storage as gcs_storage
from google.protobuf import field_mask_pb2

from ml_training.datasets.feature_engineer import TARGET_COL
from ml_training.evaluation.metrics import compute_metrics

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
    split_sha256: str | None  # sha256 of the test split test_metrics came from

    @property
    def label(self) -> str:
        return self.deployed_model_id


@dataclass
class LocalModelInfo:
    """A training run's artifacts directory as a promotion candidate."""
    artifacts_dir: Path
    test_metrics: dict
    split_sha256: str | None

    @property
    def label(self) -> str:
        return str(self.artifacts_dir)


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
    dataset_dir:
        Directory holding the challenger's test.parquet (raw feature units).
        Used only when the two models were evaluated on different splits.
    """

    def __init__(
        self,
        project_id: str,
        region: str,
        endpoint_resource_name: str,
        credentials_path: str | None = None,
        dataset_dir: str | Path = "ml_training/artifacts/dataset",
    ) -> None:
        self.project_id              = project_id
        self.region                  = region
        self.endpoint_resource_name  = endpoint_resource_name
        self._dataset_dir            = Path(dataset_dir)

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

        basis = align_metrics(champion, challenger, self._dataset_dir, self._fetch_model)
        passes, summary = compare_metrics(champion.test_metrics, challenger.test_metrics)
        print_metric_comparison(summary, basis)

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

        evaluation = self._load_evaluation(artifact_uri)

        return DeployedModelInfo(
            deployed_model_id=deployed_model_id,
            model_resource_name=proto.model,
            traffic_pct=traffic_split.get(deployed_model_id, 0),
            create_time=datetime.fromtimestamp(
                proto.create_time.timestamp(), tz=timezone.utc
            ),
            artifact_uri=artifact_uri,
            test_metrics=evaluation["metrics"],
            split_sha256=evaluation["split_sha256"],
        )

    def _load_evaluation(self, artifact_uri: str) -> dict:
        """
        Download evaluation_test.json from the model's GCS artifact URI.
        Returns {"metrics": {"auprc", "auc_roc", "brier"}, "split_sha256": str | None}.
        """
        bucket_name, prefix = _split_gcs_uri(artifact_uri)
        blob = self._storage_client.bucket(bucket_name).blob(f"{prefix}/evaluation_test.json")

        if not blob.exists():
            raise FileNotFoundError(
                f"evaluation_test.json not found at {artifact_uri}. "
                "Re-register the model with register_model.py to include it."
            )

        return parse_evaluation(json.loads(blob.download_as_text()), artifact_uri)

    def _fetch_model(self, info: DeployedModelInfo, workdir: Path) -> tuple[Path, dict]:
        """align_metrics() callback: a deployed model's files come from its GCS artifact URI."""
        return self._download_model_files(info.artifact_uri, workdir / info.deployed_model_id)

    def _download_model_files(self, artifact_uri: str, dest: Path) -> tuple[Path, dict]:
        """Fetch model.onnx (+ sidecar if any) and feature_metadata.json into dest."""
        bucket_name, prefix = _split_gcs_uri(artifact_uri)
        bucket = self._storage_client.bucket(bucket_name)
        dest.mkdir(parents=True, exist_ok=True)

        for fname, required in (("model.onnx", True), ("model.onnx.data", False),
                                ("feature_metadata.json", True)):
            blob = bucket.blob(f"{prefix}/{fname}")
            if not blob.exists():
                if required:
                    raise FileNotFoundError(f"{fname} not found at {artifact_uri}")
                continue
            blob.download_to_filename(str(dest / fname))

        with open(dest / "feature_metadata.json") as f:
            metadata = json.load(f)
        return dest / "model.onnx", metadata

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


# ── Offline gate: two artifact directories ───────────────────────────────────

def local_model_info(artifacts_dir: str | Path) -> LocalModelInfo:
    """One training run's artifacts directory (train.py output) as a candidate."""
    artifacts_dir = Path(artifacts_dir)
    for fname in ("model.onnx", "feature_metadata.json", "evaluation_test.json"):
        if not (artifacts_dir / fname).is_file():
            raise FileNotFoundError(
                f"{fname} not found in {artifacts_dir}: pass a train.py artifacts directory"
            )
    evaluation = parse_evaluation(_read_json(artifacts_dir / "evaluation_test.json"), str(artifacts_dir))
    return LocalModelInfo(
        artifacts_dir=artifacts_dir,
        test_metrics=evaluation["metrics"],
        split_sha256=evaluation["split_sha256"],
    )


def _local_model_files(info: LocalModelInfo, workdir: Path) -> tuple[Path, dict]:
    """align_metrics() callback: a local candidate's files are already on disk."""
    return info.artifacts_dir / "model.onnx", _read_json(info.artifacts_dir / "feature_metadata.json")


def promote_local(
    champion_dir: str | Path,
    challenger_dir: str | Path,
    dataset_dir: str | Path = "ml_training/artifacts/dataset",
) -> bool:
    """
    The promotion gate between two training runs, with nothing deployed.

    Same decision as ModelPromoter.promote(): the stored evaluation_test.json
    metrics when both runs were evaluated on the same test split, otherwise
    both models rescored on the challenger's split (dataset_dir/test.parquet)
    with onnxruntime. Returns True when the challenger wins on all three metrics.
    """
    champion   = local_model_info(champion_dir)
    challenger = local_model_info(challenger_dir)

    sep = "=" * 68
    print(f"\n{sep}")
    print("  Champion/Challenger Promotion Check (local artifacts)")
    print(sep)
    print(f"  Champion   : {champion.label}")
    print(f"  Challenger : {challenger.label}")

    basis = align_metrics(champion, challenger, Path(dataset_dir), _local_model_files)
    passes, summary = compare_metrics(champion.test_metrics, challenger.test_metrics)
    print_metric_comparison(summary, basis)

    if not passes:
        logger.warning(
            "PROMOTION BLOCKED — challenger did not beat champion on all metrics. "
            "Keep serving the champion."
        )
        return False
    logger.info(
        "PROMOTE — challenger beats champion on AUPRC, AUC-ROC and Brier. "
        "Serve %s (MODEL_LOCAL_PATH for the local container).",
        challenger.label,
    )
    return True


# ── Module helpers (pure; unit-tested without GCP) ────────────────────────────

def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_evaluation(data: dict, where: str) -> dict:
    """
    The promotion-relevant part of an evaluation_test.json document:
    {"metrics": {"auprc", "auc_roc", "brier"}, "split_sha256": str | None}.
    """
    metrics  = data.get("model", {})
    required = {"auprc", "auc_roc", "brier"}
    missing  = required - set(metrics)
    if missing:
        raise ValueError(
            f"evaluation_test.json at {where} is missing keys: {missing}. "
            f"Found: {list(metrics.keys())}"
        )
    return {
        "metrics":      {k: float(metrics[k]) for k in required},
        "split_sha256": data.get("split_sha256"),
    }


def load_challenger_split(dataset_dir: Path, challenger_hash: str | None) -> pd.DataFrame:
    """dataset_dir/test.parquet, verified by sha256 to be the challenger's own split."""
    test_path = dataset_dir / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(
            f"{test_path} not found. Rescoring needs the challenger's test split; "
            "pass --dataset pointing at the dataset directory it was trained from."
        )
    if challenger_hash is None:
        raise ValueError(
            "The challenger's evaluation_test.json has no split_sha256, so the "
            "local test.parquet cannot be verified as its split. Re-run evaluate.py "
            "for the challenger."
        )
    local_hash = file_sha256(test_path)
    if local_hash != challenger_hash:
        raise ValueError(
            f"{test_path} (sha256 {local_hash[:12]}…) is not the split the challenger "
            f"was evaluated on ({challenger_hash[:12]}…). Point --dataset at the "
            "challenger's dataset directory; refusing to compare on a different holdout."
        )
    return pd.read_parquet(test_path)


def align_metrics(champion, challenger, dataset_dir: Path, fetch_model) -> str:
    """
    Make champion.test_metrics and challenger.test_metrics comparable (in place).

    Same split hash → the stored metrics already are. Otherwise rescore both
    models on the challenger's test split. `fetch_model(info, workdir)` returns
    (model.onnx path, feature_metadata dict) for a candidate: downloaded from
    GCS for deployed models, read from disk for local ones. Returns a one-line
    description of the basis for the report.
    """
    if not needs_rescore(champion.split_sha256, challenger.split_sha256):
        return (
            "stored evaluation_test.json metrics — both models were evaluated "
            f"on the same test split (sha256 {challenger.split_sha256[:12]}…)"
        )

    logger.info(
        "Test splits differ (champion %s, challenger %s) — rescoring both models "
        "on the challenger's split with onnxruntime",
        (champion.split_sha256 or "unknown")[:12], (challenger.split_sha256 or "unknown")[:12],
    )
    df     = load_challenger_split(dataset_dir, challenger.split_sha256)
    y_true = df[TARGET_COL].to_numpy(dtype=float)

    with tempfile.TemporaryDirectory() as tmp:
        for info in (champion, challenger):
            onnx_path, metadata = fetch_model(info, Path(tmp))
            info.test_metrics = compute_metrics(y_true, score_onnx_on_split(onnx_path, metadata, df))
            logger.info(
                "Rescored %s: AUPRC=%.4f AUC=%.4f Brier=%.4f",
                info.label, info.test_metrics["auprc"],
                info.test_metrics["auc_roc"], info.test_metrics["brier"],
            )
    return (
        f"both models rescored locally on the challenger's test split "
        f"(n={len(df)}, sha256 {challenger.split_sha256[:12]}…)"
    )


def compare_metrics(champion: dict, challenger: dict) -> tuple[bool, dict[str, dict]]:
    """
    Returns (all_pass, per_metric_summary).
    all_pass is True only if challenger beats champion on ALL metrics.
    """
    summary: dict[str, dict] = {}
    all_pass = True

    for metric in _HIGHER_IS_BETTER:
        delta  = challenger[metric] - champion[metric]
        passes = delta > 0
        all_pass &= passes
        summary[metric] = {
            "champion":   champion[metric],
            "challenger": challenger[metric],
            "delta":      delta,
            "passes":     passes,
            "criterion":  "higher is better",
        }

    for metric in _LOWER_IS_BETTER:
        delta  = challenger[metric] - champion[metric]
        passes = delta < 0
        all_pass &= passes
        summary[metric] = {
            "champion":   champion[metric],
            "challenger": challenger[metric],
            "delta":      delta,
            "passes":     passes,
            "criterion":  "lower is better",
        }

    return all_pass, summary


def print_metric_comparison(summary: dict[str, dict], basis: str) -> None:
    print()
    print(f"  Basis: {basis}")
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


def needs_rescore(champion_hash: str | None, challenger_hash: str | None) -> bool:
    """Stored metrics are comparable only when both hashes exist and match."""
    return champion_hash is None or challenger_hash is None or champion_hash != challenger_hash


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes; same function evaluate.py uses for split_sha256."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_onnx_on_split(onnx_path: Path, metadata: dict, df: pd.DataFrame) -> np.ndarray:
    """
    Run one exported model over a raw-unit split and return shortage_prob_7d.

    The graph standardizes internally (ADR-005 Decision 3), so the parquet's
    raw columns go straight in, in the model's own numeric_features order and
    with its own vocab. Any entity the model has never seen is an error, not
    a silent default: the comparison must cover the same rows for both models.
    """
    cols    = metadata["numeric_features"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Split is missing model features {missing}")

    wh  = df["warehouse_id"].map(metadata["warehouse_vocab"])
    sku = df["sku_id"].map(metadata["sku_vocab"])
    if wh.isna().any() or sku.isna().any():
        unknown = sorted(set(df.loc[wh.isna(), "warehouse_id"]) | set(df.loc[sku.isna(), "sku_id"]))
        raise ValueError(f"Split contains entities outside this model's vocab: {unknown}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (logit,) = session.run(
        ["logit_prob"],
        {
            "numeric_features": df[cols].to_numpy(dtype=np.float32),
            "warehouse_ids":    wh.to_numpy(dtype=np.int64),
            "sku_ids":          sku.to_numpy(dtype=np.int64),
        },
    )
    return 1.0 / (1.0 + np.exp(-logit[:, 0].astype(np.float64)))


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    """'gs://bucket/some/prefix/' -> ('bucket', 'some/prefix')."""
    bucket_name, _, prefix = uri.removeprefix("gs://").partition("/")
    return bucket_name, prefix.rstrip("/")
