"""
The artifacts the trainer writes and the export reads back: model.pt is
trained on a small random dataset, loaded with weights_only=True, then
exported to ONNX.
"""
import json

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
import torch

from ml_training.datasets.feature_engineer import NUMERIC_FEATURE_COLS, TARGET_COL
from ml_training.inference.onnx_export import export_to_onnx
from ml_training.training.config import TrainingConfig
from ml_training.training.trainer import Trainer

N = len(NUMERIC_FEATURE_COLS)
_CONFIG = TrainingConfig(max_epochs=2, batch_size=32)


def _write_dataset(root) -> None:
    rng = np.random.default_rng(0)
    for split, n_rows in (("train", 256), ("val", 64)):
        df = pd.DataFrame(rng.normal(100, 50, (n_rows, N)), columns=NUMERIC_FEATURE_COLS)
        df["warehouse_id"] = rng.choice(["W-001", "W-002"], n_rows)
        df["sku_id"]       = rng.choice(["SKU-001", "SKU-002"], n_rows)
        df[TARGET_COL]     = (rng.random(n_rows) < 0.3).astype(int)
        df.to_parquet(root / f"{split}.parquet")
    (root / "feature_metadata.json").write_text(json.dumps({
        "numeric_features":   NUMERIC_FEATURE_COLS,
        "n_numeric_features": N,
        "n_warehouses":       2,
        "n_skus":             2,
        "warehouse_vocab":    {"W-001": 0, "W-002": 1},
        "sku_vocab":          {"SKU-001": 0, "SKU-002": 1},
        "feature_mean":       [100.0] * N,
        "feature_scale":      [50.0] * N,
    }))


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """A directory holding the dataset and the trainer's output for it."""
    root = tmp_path_factory.mktemp("artifacts")
    _write_dataset(root)
    Trainer(_CONFIG, dataset_dir=root, output_dir=root).train()
    return root


def test_export_writes_a_graph_that_outputs_shortage_prob_7d(artifacts):
    onnx_path = export_to_onnx(artifacts)   # also checks the graph against PyTorch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert [out.name for out in session.get_outputs()] == ["shortage_prob_7d"]


def test_checkpoint_loads_with_weights_only(artifacts):
    checkpoint = torch.load(artifacts / "model.pt", map_location="cpu", weights_only=True)
    assert TrainingConfig(**checkpoint["config"]) == _CONFIG
