"""
LocalEndpoint — the serving container reached over plain HTTP.

Stand-in for aiplatform.Endpoint when the container runs locally
(`docker run -p 8080:8080` with MODEL_LOCAL_PATH; see local/README.md)
instead of behind a Vertex AI endpoint. It exposes the one call the batch
scorer and the smoke test make, with the same request and response shape,
so neither caller branches on where the model is served:

    endpoint.predict(instances=[{...}, ...]).predictions
    -> [{"shortage_prob_7d": float}, ...]

The container's /predict already speaks the Vertex AI prediction contract
(POST {"instances": [...]} -> {"predictions": [...]}), so this is standard
library urllib and nothing else.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class LocalPrediction:
    """The attribute the callers read from aiplatform's Prediction object."""

    predictions: list[dict]


class LocalEndpoint:
    """
    Parameters
    ----------
    base_url:
        Where the serving container listens, e.g. "http://localhost:8080".
    timeout:
        Seconds to wait for one /predict call (default 60).
    """

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def health(self) -> dict:
        """GET /health: {"status", "model_version", "n_numeric_features"}."""
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def predict(self, instances: list[dict]) -> LocalPrediction:
        """POST /predict with the Vertex AI request shape; raises on a non-2xx reply."""
        body = json.dumps({"instances": instances}).encode("utf-8")
        req  = urllib.request.Request(
            f"{self.base_url}/predict",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{self.base_url}/predict returned HTTP {exc.code}: {detail}"
            ) from exc
        return LocalPrediction(predictions=payload["predictions"])
