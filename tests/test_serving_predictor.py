"""
model_serving/serving/container/predictor.py: the container downloads a model
from GCS at startup and serves it. GCS is a fake that copies from a directory;
the model is a tiny ShortagePredictor exported the way
ml_training/inference/onnx_export.py exports the real one.
"""
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from ml_training.datasets.feature_engineer import NUMERIC_FEATURE_COLS
from ml_training.training.config import TrainingConfig
from ml_training.training.model import ShortagePredictor

_CONTAINER_DIR = Path(__file__).parents[1] / "model_serving" / "serving" / "container"
_GCS_PATH = "gs://bucket/models/shortage-predictor/v20261031_test"
N = len(NUMERIC_FEATURE_COLS)


def _predictor_module():
    if str(_CONTAINER_DIR) not in sys.path:
        sys.path.insert(0, str(_CONTAINER_DIR))
    import predictor  # noqa: PLC0415 -- path-dependent import by design
    return predictor


def _write_artifacts(dest: Path) -> None:
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


@pytest.fixture()
def predictor(tmp_path, monkeypatch):
    """The predictor module with GCS replaced by the directory tmp_path/gcs."""
    module = _predictor_module()
    gcs = tmp_path / "gcs"
    gcs.mkdir()

    def blob(path: Path):
        return types.SimpleNamespace(
            name=f"models/shortage-predictor/v20261031_test/{path.name}",
            size=path.stat().st_size,
            download_to_filename=lambda dest, _p=path: shutil.copy(_p, dest),
        )

    bucket = types.SimpleNamespace(list_blobs=lambda prefix: [blob(p) for p in sorted(gcs.iterdir())])
    monkeypatch.setattr(module.storage, "Client", lambda: types.SimpleNamespace(bucket=lambda _: bucket))
    monkeypatch.setattr(module, "_DOWNLOAD_DIR", tmp_path / "download")
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    monkeypatch.delenv("MODEL_GCS_PATH", raising=False)
    module.gcs_dir = gcs
    return module


def test_serves_the_model_downloaded_from_gcs(predictor, monkeypatch):
    _write_artifacts(predictor.gcs_dir)
    monkeypatch.setenv("MODEL_GCS_PATH", _GCS_PATH)
    monkeypatch.setenv("MODEL_VERSION", "test-version")

    p = predictor.ONNXPredictor()

    assert p.model_version == "test-version"
    assert p.n_numeric_features == N
    assert (p.n_warehouses, p.n_skus) == (2, 3)
    probs = p.predict(
        np.full((4, N), 100.0, dtype=np.float32),
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 2, 0]),
    )
    assert probs.shape == (4,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_version_defaults_to_the_last_path_segment(predictor):
    _write_artifacts(predictor.gcs_dir)
    assert predictor.ONNXPredictor(gcs_path=_GCS_PATH).model_version == "v20261031_test"


def test_missing_path_and_missing_files_fail_clearly(predictor):
    with pytest.raises(ValueError, match="MODEL_GCS_PATH"):
        predictor.ONNXPredictor()
    with pytest.raises(FileNotFoundError, match="No files found"):
        predictor.ONNXPredictor(gcs_path=_GCS_PATH)
