"""
Tests for the pure parts of model_serving/registry/promote_model.py that make
champion/challenger comparison happen on one holdout, and for promote_local(),
the same gate run between two artifact directories with nothing deployed.

ModelPromoter itself needs Vertex AI and GCS clients, so these tests cover
the module-level helpers it composes: the split-hash guard, the file hash,
scoring an exported ONNX model over a raw-unit split, and the metric gate.
"""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from ml_training.datasets.feature_engineer import NUMERIC_FEATURE_COLS, TARGET_COL
from ml_training.training.config import TrainingConfig
from ml_training.training.model import ShortagePredictor
from model_serving.registry import promote_model as pm

N = len(NUMERIC_FEATURE_COLS)


def _metadata() -> dict:
    return {
        "numeric_features":   NUMERIC_FEATURE_COLS,
        "n_numeric_features": N,
        "n_warehouses":       2,
        "n_skus":             2,
        "warehouse_vocab":    {"W-001": 0, "W-002": 1},
        "sku_vocab":          {"SKU-001": 0, "SKU-002": 1},
        "feature_mean":       [100.0] * N,
        "feature_scale":      [50.0] * N,
    }


def _export(tmp_path, metadata: dict, seed: int):
    torch.manual_seed(seed)
    config = TrainingConfig(hidden_dims=(8,), dropout_rates=(0.0,))
    model  = ShortagePredictor.from_metadata(metadata, config).eval()
    path   = tmp_path / f"model_{seed}.onnx"
    torch.onnx.export(
        model,
        (torch.randn(2, N) * 50 + 100, torch.tensor([0, 1]), torch.tensor([0, 1])),
        str(path),
        dynamo=False, opset_version=17,
        input_names=["numeric_features", "warehouse_ids", "sku_ids"],
        output_names=["logit_prob"],
        dynamic_axes={"numeric_features": {0: "batch"}, "warehouse_ids": {0: "batch"},
                      "sku_ids": {0: "batch"}, "logit_prob": {0: "batch"}},
    )
    return model, path


def _split(n_rows: int = 64, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df  = pd.DataFrame(rng.normal(100, 50, (n_rows, N)), columns=NUMERIC_FEATURE_COLS)
    df["warehouse_id"] = np.where(rng.random(n_rows) < 0.5, "W-001", "W-002")
    df["sku_id"]       = np.where(rng.random(n_rows) < 0.5, "SKU-001", "SKU-002")
    df[TARGET_COL]     = (rng.random(n_rows) < 0.3).astype(int)
    return df


def test_stored_metrics_are_comparable_only_on_the_same_split():
    assert pm.needs_rescore("abc", "abc") is False
    assert pm.needs_rescore("abc", "def") is True
    assert pm.needs_rescore(None, "abc") is True
    assert pm.needs_rescore("abc", None) is True


def test_file_sha256_identifies_the_split(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _split(seed=0).to_parquet(a)
    _split(seed=1).to_parquet(b)
    assert pm.file_sha256(a) == pm.file_sha256(a)
    assert pm.file_sha256(a) != pm.file_sha256(b)
    assert len(pm.file_sha256(a)) == 64


def test_score_onnx_on_split_matches_torch_predict(tmp_path):
    metadata    = _metadata()
    model, path = _export(tmp_path, metadata, seed=0)
    df          = _split()

    probs = pm.score_onnx_on_split(path, metadata, df)

    x   = torch.tensor(df[NUMERIC_FEATURE_COLS].to_numpy(dtype=np.float32))
    wh  = torch.tensor(df["warehouse_id"].map(metadata["warehouse_vocab"]).to_numpy(dtype=np.int64))
    sku = torch.tensor(df["sku_id"].map(metadata["sku_vocab"]).to_numpy(dtype=np.int64))
    expected = model.predict(x, wh, sku).squeeze(1).numpy()

    assert probs.shape == (len(df),)
    np.testing.assert_allclose(probs, expected, atol=1e-5)


def test_score_onnx_on_split_refuses_entities_outside_the_vocab(tmp_path):
    metadata = _metadata()
    _, path  = _export(tmp_path, metadata, seed=0)
    df = _split()
    df.loc[0, "warehouse_id"] = "W-009"
    with pytest.raises(ValueError, match="W-009"):
        pm.score_onnx_on_split(path, metadata, df)


# ── Metric gate and the offline (--local) promotion ──────────────────────────

_CHAMPION = {"auprc": 0.60, "auc_roc": 0.80, "brier": 0.15}


def test_compare_metrics_requires_a_win_on_all_three():
    better = {"auprc": 0.65, "auc_roc": 0.82, "brier": 0.14}
    passes, summary = pm.compare_metrics(_CHAMPION, better)
    assert passes and all(d["passes"] for d in summary.values())

    worse_brier = {"auprc": 0.65, "auc_roc": 0.82, "brier": 0.16}
    passes, summary = pm.compare_metrics(_CHAMPION, worse_brier)
    assert not passes
    assert summary["brier"]["passes"] is False and summary["auprc"]["passes"] is True


def _artifacts_dir(tmp_path, name: str, seed: int, metrics: dict, split_sha256: str | None):
    """A train.py-shaped artifacts directory: model.onnx, feature_metadata.json, evaluation_test.json."""
    d = tmp_path / name
    d.mkdir()
    metadata = _metadata()
    _, onnx_path = _export(d, metadata, seed=seed)
    onnx_path.rename(d / "model.onnx")
    (d / "feature_metadata.json").write_text(json.dumps(metadata))
    (d / "evaluation_test.json").write_text(json.dumps({"model": metrics, "split_sha256": split_sha256}))
    return d


def test_promote_local_uses_stored_metrics_on_the_same_split(tmp_path, capsys):
    champ = _artifacts_dir(tmp_path, "champion",   0, _CHAMPION, "a" * 64)
    chal  = _artifacts_dir(tmp_path, "challenger", 1, {"auprc": 0.70, "auc_roc": 0.85, "brier": 0.10}, "a" * 64)

    assert pm.promote_local(champ, chal, dataset_dir=tmp_path / "no-dataset-needed") is True
    out = capsys.readouterr().out
    assert "stored evaluation_test.json metrics" in out
    assert "Verdict: PROMOTE" in out


def test_promote_local_blocks_a_weaker_challenger(tmp_path, capsys):
    champ = _artifacts_dir(tmp_path, "champion",   0, _CHAMPION, "a" * 64)
    chal  = _artifacts_dir(tmp_path, "challenger", 1, {"auprc": 0.55, "auc_roc": 0.85, "brier": 0.10}, "a" * 64)

    assert pm.promote_local(champ, chal, dataset_dir=tmp_path) is False
    assert "Verdict: BLOCKED" in capsys.readouterr().out


def test_promote_local_rescores_both_models_when_splits_differ(tmp_path, capsys):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _split(seed=3).to_parquet(dataset / "test.parquet")
    challenger_hash = pm.file_sha256(dataset / "test.parquet")

    # Stored metrics would promote, but they are from different splits, so
    # both models are rescored on the challenger's split and the stored
    # numbers are ignored.
    champ = _artifacts_dir(tmp_path, "champion",   0, _CHAMPION, "b" * 64)
    chal  = _artifacts_dir(tmp_path, "challenger", 1, {"auprc": 0.99, "auc_roc": 0.99, "brier": 0.01}, challenger_hash)

    verdict = pm.promote_local(champ, chal, dataset_dir=dataset)
    out = capsys.readouterr().out
    assert isinstance(verdict, bool)
    assert "rescored locally on the challenger's test split (n=64" in out
    assert "0.9900" not in out   # stored challenger metrics were replaced


def test_promote_local_refuses_a_dataset_that_is_not_the_challengers_split(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _split(seed=3).to_parquet(dataset / "test.parquet")
    champ = _artifacts_dir(tmp_path, "champion",   0, _CHAMPION, "b" * 64)
    chal  = _artifacts_dir(tmp_path, "challenger", 1, _CHAMPION, "c" * 64)
    with pytest.raises(ValueError, match="not the split the challenger"):
        pm.promote_local(champ, chal, dataset_dir=dataset)


def test_local_model_info_needs_a_complete_artifacts_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.onnx"):
        pm.local_model_info(tmp_path)
