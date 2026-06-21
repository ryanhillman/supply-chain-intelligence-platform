"""
ONNXPredictor — loads model.onnx from GCS at container startup and runs
inference via onnxruntime.

Inputs received from batch_scorer.py (all scaling and vocab mapping is
done client-side; see ADR-005 Decision 3):
  numeric_features : float32 (batch, 21)  — StandardScaler-transformed
  warehouse_ids    : int64   (batch,)     — vocab int, NOT warehouse string ID
  sku_ids          : int64   (batch,)     — vocab int, NOT SKU string ID

Outputs returned to caller:
  shortage_prob_7d    : float32 (batch,)  — sigmoid(logit_prob), P(stockout<=7d)
  stockout_risk_score : float32 (batch,)  — direct risk_score ONNX output in [0,1]

Environment variables:
  MODEL_GCS_PATH            — GCS URI to the versioned model directory,
                              e.g. gs://bucket/models/shortage-predictor/v20260620_180918
  GOOGLE_APPLICATION_CREDENTIALS — path to SA key JSON (not needed on Vertex AI;
                              the container SA is injected automatically)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from google.cloud import storage

logger = logging.getLogger(__name__)

# Must match len(phase4.datasets.feature_engineer.NUMERIC_FEATURE_COLS).
# If the model architecture changes, rebuild the container image.
_N_NUMERIC_FEATURES: int = 21

# Model files are downloaded here at startup. /tmp is world-writable so
# this works under both root and the non-root appuser.
_MODEL_DIR: Path = Path("/tmp/model")


class ONNXPredictor:
    """
    Singleton loaded once during FastAPI lifespan startup.

    Parameters
    ----------
    gcs_path:
        GCS URI to the model directory, e.g.
        gs://bucket/models/shortage-predictor/v20260620_180918
        Defaults to the MODEL_GCS_PATH environment variable.
    """

    def __init__(self, gcs_path: str | None = None) -> None:
        self._gcs_path = gcs_path or os.environ["MODEL_GCS_PATH"]
        self.model_version: str = self._parse_version(self._gcs_path)

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self._download_model_files()
        self._session = self._create_session()

        logger.info(
            "ONNXPredictor ready | version=%s | inputs=%s",
            self.model_version,
            [inp.name for inp in self._session.get_inputs()],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        numeric_features: np.ndarray,
        warehouse_ids: np.ndarray,
        sku_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run ONNX inference on a pre-scaled, pre-vocabbed batch.

        Parameters
        ----------
        numeric_features : float32, shape (batch, 21)
            Already StandardScaler-transformed. Do NOT pass raw Silver values.
        warehouse_ids : int64, shape (batch,)
            Vocab indices from feature_metadata.json["warehouse_vocab"].
        sku_ids : int64, shape (batch,)
            Vocab indices from feature_metadata.json["sku_vocab"].

        Returns
        -------
        shortage_prob_7d : float32, shape (batch,), in [0, 1]
        stockout_risk_score : float32, shape (batch,), in [0, 1]
        """
        outputs = self._session.run(
            output_names=["logit_prob", "risk_score"],
            input_feed={
                "numeric_features": numeric_features.astype(np.float32),
                "warehouse_ids":    warehouse_ids.astype(np.int64),
                "sku_ids":          sku_ids.astype(np.int64),
            },
        )
        logit_prob, risk_score = outputs

        # logit_prob shape: (batch, 1) — apply sigmoid to get probability
        shortage_prob = _sigmoid(logit_prob[:, 0])

        # risk_score shape: (batch, 1) — already sigmoid from model's score_head
        stockout_risk = risk_score[:, 0]

        return shortage_prob, stockout_risk

    @property
    def n_numeric_features(self) -> int:
        return _N_NUMERIC_FEATURES

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_version(gcs_path: str) -> str:
        """Extract version tag from the last non-empty path segment."""
        return gcs_path.rstrip("/").rsplit("/", 1)[-1]

    def _download_model_files(self) -> None:
        """
        Download model.onnx and model.onnx.data (if present) from GCS.

        model.onnx.data is the external weights sidecar created when PyTorch
        exports large models. onnxruntime requires it to be co-located with
        model.onnx at load time.
        """
        path_without_scheme = self._gcs_path.removeprefix("gs://")
        bucket_name, _, prefix = path_without_scheme.partition("/")
        prefix = prefix.rstrip("/") + "/"

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            raise FileNotFoundError(
                f"No files found at {self._gcs_path}. "
                "Run phase5/scripts/register.py first to upload model artifacts."
            )

        downloaded: list[str] = []
        for blob in blobs:
            filename = blob.name.rsplit("/", 1)[-1]
            if filename in ("model.onnx", "model.onnx.data"):
                dest = _MODEL_DIR / filename
                blob.download_to_filename(str(dest))
                downloaded.append(filename)
                logger.info(
                    "Downloaded gs://%s/%s -> %s (%s bytes)",
                    bucket_name, blob.name, dest, blob.size,
                )

        if "model.onnx" not in downloaded:
            raise FileNotFoundError(
                f"model.onnx not found at {self._gcs_path}. "
                "Verify the MODEL_GCS_PATH environment variable."
            )

    def _create_session(self) -> ort.InferenceSession:
        model_path = _MODEL_DIR / "model.onnx"
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # suppress onnxruntime INFO spam
        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

        # Validate expected I/O at startup, not at first predict call.
        input_names = {inp.name for inp in session.get_inputs()}
        expected_inputs = {"numeric_features", "warehouse_ids", "sku_ids"}
        if input_names != expected_inputs:
            raise ValueError(
                f"ONNX input names mismatch. "
                f"Expected {expected_inputs}, got {input_names}."
            )

        output_names = {out.name for out in session.get_outputs()}
        expected_outputs = {"logit_prob", "risk_score"}
        if output_names != expected_outputs:
            raise ValueError(
                f"ONNX output names mismatch. "
                f"Expected {expected_outputs}, got {output_names}."
            )

        return session


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for float32."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    ).astype(np.float32)
