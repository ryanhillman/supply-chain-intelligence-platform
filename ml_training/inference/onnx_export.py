"""
ONNX export — prepares the trained ShortagePredictor for Vertex AI serving.

Exports to ONNX opset 17 with dynamic batch axis. The model's Standardize
buffers (train-split mean/scale) are exported inside the graph, so the ONNX
model takes raw feature units (ADR-005 Decision 3).
Validates the exported graph with onnxruntime before reporting success.

Phase 5 will consume model.onnx from GCS to register with Vertex AI
Model Registry and deploy to a Vertex AI Endpoint.
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
    ckpt   = torch.load(artifacts_dir / "model.pt", map_location="cpu", weights_only=False)
    config = ckpt.get("config", TrainingConfig())

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

    # dynamo=False pins the TorchScript exporter. torch >= 2.9 defaults to
    # the dynamo exporter, which ignores dynamic_axes (it wants dynamic_shapes),
    # upgrades opset 17 to 18 on its own, and crashed here on a Windows console
    # while printing its progress line. The TorchScript path honors the
    # arguments below and emits the opset-17 graph the container documents.
    torch.onnx.export(
        model,
        (dummy_numeric, dummy_wh, dummy_sku),
        str(output_path),
        dynamo           = False,
        opset_version    = 17,
        input_names      = ["numeric_features", "warehouse_ids", "sku_ids"],
        output_names     = ["logit_prob", "risk_score"],
        dynamic_axes     = {
            "numeric_features": {0: "batch"},
            "warehouse_ids":    {0: "batch"},
            "sku_ids":          {0: "batch"},
            "logit_prob":       {0: "batch"},
            "risk_score":       {0: "batch"},
        },
        do_constant_folding = True,
    )

    # Merge any external-data sidecar into the ONNX file so the serving
    # container receives a single self-contained artifact.
    _consolidate_external_data(output_path)
    logger.info("ONNX model written to %s", output_path)

    # ── Validate with onnxruntime ─────────────────────────────────────────────
    onnx_model = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    inputs = {
        "numeric_features": dummy_numeric.numpy(),
        "warehouse_ids":    dummy_wh.numpy(),
        "sku_ids":          dummy_sku.numpy(),
    }
    outputs = sess.run(None, inputs)
    logit_out, score_out = outputs

    if logit_out.shape != (batch_size, 1):
        raise ValueError(f"Unexpected logit shape: {logit_out.shape}")
    if score_out.shape != (batch_size, 1):
        raise ValueError(f"Unexpected score shape: {score_out.shape}")

    # With the two rows now distinct, outputs should differ across the
    # batch — if they didn't, that would indicate a broadcast/transpose bug
    # collapsing the batch dimension that identical dummy rows would hide.
    if np.allclose(logit_out[0], logit_out[1]) and np.allclose(score_out[0], score_out[1]):
        raise ValueError(
            "ONNX outputs are identical across distinct batch rows — "
            "possible batch-dimension handling bug in the exported graph."
        )

    # Cross-check: onnxruntime logit → sigmoid should match PyTorch predict()
    with torch.no_grad():
        pt_prob, pt_score = model.predict(dummy_numeric, dummy_wh, dummy_sku)

    np.testing.assert_allclose(
        1.0 / (1.0 + np.exp(-logit_out)),   # sigmoid of ONNX logit output
        pt_prob.numpy(),
        atol=1e-5,
        err_msg="ONNX probability output diverges from PyTorch",
    )
    logger.info("ONNX validation passed — outputs match PyTorch within 1e-5")

    return output_path


def _consolidate_external_data(onnx_path: Path) -> None:
    """Merge any .onnx.data sidecar into the ONNX file as inline tensors."""
    from onnx.external_data_helper import load_external_data_for_model

    sidecar = onnx_path.parent / (onnx_path.name + ".data")
    if not sidecar.exists():
        return

    proto = onnx.load(str(onnx_path), load_external_data=False)
    load_external_data_for_model(proto, str(onnx_path.parent))
    onnx.save(proto, str(onnx_path), save_as_external_data=False)
    sidecar.unlink()
    logger.info("Merged external data sidecar into %s", onnx_path)


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
