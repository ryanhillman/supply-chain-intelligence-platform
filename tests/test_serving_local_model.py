"""
model_serving/serving/container/predictor.py in MODEL_LOCAL_PATH mode: the
container serves a model from a mounted directory with no GCS download (the
zero-cost local run). The model is a tiny ShortagePredictor exported the way
ml_training/inference/onnx_export.py exports the real one.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ml_training.datasets.feature_engineer import NUMERIC_FEATURE_COLS
from ml_training.training.config import TrainingConfig
from ml_training.training.model import ShortagePredictor

_CONTAINER_DIR = Path(__file__).parents[1] / "model_serving" / "serving" / "container"
N = len(NUMERIC_FEATURE_COLS)


def _predictor_module():
    if str(_CONTAINER_DIR) not in sys.path:
        sys.path.insert(0, str(_CONTAINER_DIR))
    import predictor  # noqa: PLC0415 -- path-dependent import by design
    return predictor


def _write_artifacts(dest: Path) -> dict:
    metadata = {
        "numeric_features":   NUMERIC_FEATURE_COLS,
        "n_numeric_features": N,
        "n_warehouses":       2,
        "n_skus":             3,
        "warehouse_vocab":    {"W-001": 0, "W-002": 1},
        "sku_vocab":          {"SKU-001": 0, "SKU-002": 1, "SKU-003": 2},
        "feature_mean":       [100.0] * N,
        "feature_scale":      [50.0] * N,
    }
    torch.manual_seed(0)
    config = TrainingConfig(hidden_dims=(8,), dropout_rates=(0.0,))
    model  = ShortagePredictor.from_metadata(metadata, config).eval()
    torch.onnx.export(
        model,
        (torch.randn(2, N) * 50 + 100, torch.tensor([0, 1]), torch.tensor([0, 1])),
        str(dest / "model.onnx"),
        dynamo=False, opset_version=17,
        input_names=["numeric_features", "warehouse_ids", "sku_ids"],
        output_names=["logit_prob"],
        dynamic_axes={"numeric_features": {0: "batch"}, "warehouse_ids": {0: "batch"},
                      "sku_ids": {0: "batch"}, "logit_prob": {0: "batch"}},
    )
    (dest / "feature_metadata.json").write_text(json.dumps(metadata))
    return metadata


def test_local_path_serves_the_mounted_model(tmp_path, monkeypatch):
    _write_artifacts(tmp_path)
    monkeypatch.delenv("MODEL_GCS_PATH", raising=False)
    monkeypatch.setenv("MODEL_LOCAL_PATH", str(tmp_path))
    monkeypatch.setenv("MODEL_VERSION", "local-test")

    p = _predictor_module().ONNXPredictor()

    assert p.model_version == "local-test"
    assert p.n_numeric_features == N
    assert (p.n_warehouses, p.n_skus) == (2, 3)
    probs = p.predict(
        np.full((4, N), 100.0, dtype=np.float32),
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 2, 0]),
    )
    assert probs.shape == (4,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_version_defaults_to_the_directory_name(tmp_path, monkeypatch):
    model_dir = tmp_path / "v20261031_test"
    model_dir.mkdir()
    _write_artifacts(model_dir)
    monkeypatch.delenv("MODEL_GCS_PATH", raising=False)
    monkeypatch.delenv("MODEL_VERSION", raising=False)

    p = _predictor_module().ONNXPredictor(local_path=str(model_dir))

    assert p.model_version == "v20261031_test"


def test_missing_files_and_ambiguous_source_fail_clearly(tmp_path, monkeypatch):
    predictor = _predictor_module()
    monkeypatch.delenv("MODEL_GCS_PATH", raising=False)
    monkeypatch.delenv("MODEL_LOCAL_PATH", raising=False)

    with pytest.raises(ValueError, match="exactly one"):
        predictor.ONNXPredictor()
    with pytest.raises(ValueError, match="exactly one"):
        predictor.ONNXPredictor(gcs_path="gs://bucket/models/v1", local_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="model.onnx"):
        predictor.ONNXPredictor(local_path=str(tmp_path))
