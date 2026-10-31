"""
FastAPI serving container for the SCIP ShortagePredictor.

Vertex AI contract:
  GET  /health  -> 200 {"status": "ok", "model_version": str}
                   503 if the model has not loaded yet
  POST /predict -> 200 {"predictions": [{"shortage_prob_7d": float,
                                          "stockout_risk_score": float}, ...]}
                   422 if request schema is invalid
                   500 on inference error

Vertex AI deploy flags (set in model_serving/scripts/deploy.py):
  --health-route   /health
  --predict-route  /predict
  --port           8080

The container does NOT apply vocab mapping — that is the caller's
responsibility (see request_builder.py). Numeric features arrive in raw
units; the ONNX graph standardizes them itself (ADR-005 Decision 3).
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from predictor import ONNXPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("scip.serving")

# Module-level singleton; populated during lifespan startup.
_predictor: ONNXPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Download model from GCS and create onnxruntime session on startup."""
    global _predictor
    logger.info("Container startup — loading ONNX model from GCS")
    _predictor = ONNXPredictor()
    logger.info("Container ready")
    yield
    # Nothing to clean up; onnxruntime session is GC'd normally.


app = FastAPI(
    title="SCIP Shortage Predictor",
    description="7-day stockout risk scoring for warehouse-SKU pairs",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────

class Instance(BaseModel):
    """One warehouse-SKU pair to score.

    Fields
    ------
    numeric_features:
        Raw feature values in the same order as NUMERIC_FEATURE_COLS. Do not
        pre-scale: the ONNX graph standardizes with the training statistics.
        The count must match the loaded model's input width (reported by
        /health as n_numeric_features); a mismatch is rejected with 422.
    warehouse_id:
        Integer vocab index from feature_metadata.json["warehouse_vocab"].
        NOT the string ID "W-001".
    sku_id:
        Integer vocab index from feature_metadata.json["sku_vocab"].
        NOT the string ID "SKU-001".
    """
    numeric_features: list[float]
    warehouse_id: int
    sku_id: int

    @field_validator("warehouse_id", "sku_id")
    @classmethod
    def check_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Vocab IDs must be non-negative integers.")
        return v


class PredictRequest(BaseModel):
    instances: list[Instance]

    @field_validator("instances")
    @classmethod
    def check_non_empty(cls, v: list[Instance]) -> list[Instance]:
        if len(v) == 0:
            raise ValueError("instances must contain at least one element.")
        return v


class Prediction(BaseModel):
    shortage_prob_7d: float
    stockout_risk_score: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, Any]:
    """Vertex AI health probe. Returns 503 until the model is loaded."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return {
        "status": "ok",
        "model_version": _predictor.model_version,
        "n_numeric_features": _predictor.n_numeric_features,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict[str, Any]:
    """
    Score a batch of warehouse-SKU instances.

    Accepts the standard Vertex AI prediction request format:
      {"instances": [{...}, {...}]}

    Returns:
      {"predictions": [{"shortage_prob_7d": float, "stockout_risk_score": float}]}
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")

    expected_width = _predictor.n_numeric_features
    bad_width = [
        (i, len(inst.numeric_features))
        for i, inst in enumerate(request.instances)
        if len(inst.numeric_features) != expected_width
    ]
    if bad_width:
        raise HTTPException(
            status_code=422,
            detail=(
                f"numeric_features must have exactly {expected_width} elements "
                f"(the loaded model's input width). Offending instances "
                f"(index, length): {bad_width}"
            ),
        )

    out_of_range = [
        (i, inst.warehouse_id, inst.sku_id)
        for i, inst in enumerate(request.instances)
        if inst.warehouse_id >= _predictor.n_warehouses
        or inst.sku_id >= _predictor.n_skus
    ]
    if out_of_range:
        raise HTTPException(
            status_code=422,
            detail=(
                f"warehouse_id must be < {_predictor.n_warehouses} and "
                f"sku_id must be < {_predictor.n_skus}. Out-of-range instances "
                f"(index, warehouse_id, sku_id): {out_of_range}"
            ),
        )

    batch_size = len(request.instances)
    t0 = time.perf_counter()

    numeric_features = np.array(
        [inst.numeric_features for inst in request.instances],
        dtype=np.float32,
    )
    warehouse_ids = np.array(
        [inst.warehouse_id for inst in request.instances],
        dtype=np.int64,
    )
    sku_ids = np.array(
        [inst.sku_id for inst in request.instances],
        dtype=np.int64,
    )

    try:
        probs, scores = _predictor.predict(numeric_features, warehouse_ids, sku_ids)
    except Exception as exc:
        logger.exception("Inference failed for batch_size=%d", batch_size)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Predicted batch_size=%d | prob_mean=%.3f | score_mean=%.3f | latency=%.1fms",
        batch_size,
        float(probs.mean()),
        float(scores.mean()),
        elapsed_ms,
    )

    predictions = [
        Prediction(
            shortage_prob_7d=float(p),
            stockout_risk_score=float(s),
        )
        for p, s in zip(probs.tolist(), scores.tolist())
    ]
    return {"predictions": predictions}
