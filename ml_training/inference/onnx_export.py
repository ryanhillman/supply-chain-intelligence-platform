"""
ONNX export — prepares the trained ShortagePredictor for Vertex AI serving.

Exports to ONNX opset 17 with dynamic batch axis. The model's Standardize
buffers (train-split mean/scale) and the final sigmoid are exported inside the
graph, so the ONNX model takes raw feature units and outputs shortage_prob_7d.
Validates the exported graph with onnxruntime before reporting success.

model_serving/scripts/register.py uploads model.onnx to GCS and registers it
with Vertex AI Model Registry.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ml_training.datasets.feature_engineer import NUMERIC_FEATURE_COLS
from ml_training.training.config import TrainingConfig
from ml_training.training.model import ShortagePredictor

logger = logging.getLogger(__name__)

N_NUMERIC = len(NUMERIC_FEATURE_COLS)


class _WithSigmoid(torch.nn.Module):
    """The model followed by the sigmoid, so the graph outputs a probability of shape (batch,)."""

    def __init__(self, model: ShortagePredictor) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x_numeric: torch.Tensor,
        wh_ids:    torch.Tensor,
        sku_ids:   torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.model(x_numeric, wh_ids, sku_ids)).squeeze(1)


def export_graph(model: ShortagePredictor, example_inputs: tuple, output_path: str | Path) -> None:
    """
    Write `model` to output_path as an ONNX graph that takes numeric_features,
    warehouse_ids and sku_ids and returns shortage_prob_7d.

    example_inputs is one (x_numeric, wh_ids, sku_ids) batch for tracing.
    """
    # dynamo=False pins the TorchScript exporter. torch >= 2.9 defaults to
    # the dynamo exporter, which ignores dynamic_axes (it wants dynamic_shapes),
    # upgrades opset 17 to 18 on its own, and crashed here on a Windows console
    # while printing its progress line. The TorchScript path honors the
    # arguments below and emits the opset-17 graph the container documents.
    torch.onnx.export(
        _WithSigmoid(model),
        example_inputs,
        str(output_path),
        dynamo           = False,
        opset_version    = 17,
        input_names      = ["numeric_features", "warehouse_ids", "sku_ids"],
        output_names     = ["shortage_prob_7d"],
        dynamic_axes     = {
            "numeric_features": {0: "batch"},
            "warehouse_ids":    {0: "batch"},
            "sku_ids":          {0: "batch"},
            "shortage_prob_7d": {0: "batch"},
        },
        do_constant_folding = True,
    )


def export_to_onnx(
    artifacts_dir: str | Path,
    output_path:   str | Path | None = None,
) -> Path:
    """
    Load model.pt from artifacts_dir, export to ONNX, validate with onnxruntime.

    Parameters
    ----------
    artifacts_dir:
        Directory containing model.pt and feature_metadata.json.
    output_path:
        Destination .onnx file. Defaults to artifacts_dir/model.onnx.

    Returns
    -------
    Path to the written .onnx file.
    """
    artifacts_dir = Path(artifacts_dir)
    output_path   = Path(output_path) if output_path else artifacts_dir / "model.onnx"

    metadata = _load_json(artifacts_dir / "feature_metadata.json")
    ckpt   = torch.load(artifacts_dir / "model.pt", map_location="cpu", weights_only=True)
    config = TrainingConfig(**ckpt["config"])

    model = ShortagePredictor.from_metadata(metadata, config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Dummy input for tracing (batch_size=2 to exercise batch dimension).
    # Rows are deliberately distinct (not both zero) so validation below can
    # assert the two rows produce different outputs — all-identical rows
    # would pass shape checks even if a transpose/broadcast bug collapsed
    # the batch dimension.
    batch_size = 2
    torch.manual_seed(0)
    # Raw-unit dummy rows (mean ± noise·scale); the graph standardizes internally.
    mean  = torch.tensor(metadata["feature_mean"],  dtype=torch.float32)
    scale = torch.tensor(metadata["feature_scale"], dtype=torch.float32)
    dummy_numeric = mean + torch.randn(batch_size, metadata["n_numeric_features"]) * scale
    dummy_wh      = torch.tensor(
        [0, 1 % max(metadata["n_warehouses"], 1)], dtype=torch.long
    )
    dummy_sku     = torch.tensor(
        [0, 1 % max(metadata["n_skus"], 1)], dtype=torch.long
    )

    export_graph(model, (dummy_numeric, dummy_wh, dummy_sku), output_path)

    logger.info("ONNX model written to %s", output_path)

    # ── Validate with onnxruntime ─────────────────────────────────────────────
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    inputs = {
        "numeric_features": dummy_numeric.numpy(),
        "warehouse_ids":    dummy_wh.numpy(),
        "sku_ids":          dummy_sku.numpy(),
    }
    (prob_out,) = sess.run(None, inputs)

    if prob_out.shape != (batch_size,):
        raise ValueError(f"Unexpected output shape: {prob_out.shape}")

    # The two rows are distinct, so the outputs should differ across the
    # batch — if they didn't, that would indicate a broadcast/transpose bug
    # collapsing the batch dimension that identical dummy rows would hide.
    if prob_out[0] == prob_out[1]:
        raise ValueError(
            "ONNX outputs are identical across distinct batch rows — "
            "possible batch-dimension handling bug in the exported graph."
        )

    # Cross-check: the graph's probability should match PyTorch predict()
    with torch.no_grad():
        pt_prob = model.predict(dummy_numeric, dummy_wh, dummy_sku)

    np.testing.assert_allclose(
        prob_out,
        pt_prob.squeeze(1).numpy(),
        atol=1e-5,
        err_msg="ONNX probability output diverges from PyTorch",
    )
    logger.info("ONNX validation passed — outputs match PyTorch within 1e-5")

    return output_path


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
