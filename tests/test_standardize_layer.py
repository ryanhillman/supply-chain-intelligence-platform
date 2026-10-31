"""
Tests for the model-owned standardization (ml_training/training/model.py).

The parquet splits hold raw feature units; DatasetBuilder fits a
StandardScaler on the train split and stores its mean_/scale_ in
feature_metadata.json; ShortagePredictor applies (x - mean) / scale as a
fixed first layer. These tests pin the contract that makes that safe:
the layer reproduces sklearn's transform, the statistics are buffers the
optimizer cannot touch, the zero-variance guard carries over, and the
values survive a state_dict round trip.
"""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from ml_training.training.config import TrainingConfig
from ml_training.training.model import ShortagePredictor, Standardize


def _train_matrix(n_rows: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = [
        rng.normal(2800, 900, n_rows),        # large mean (active_po_qty-like)
        rng.exponential(35, n_rows),          # skewed (avg_daily_sales_7d-like)
        rng.uniform(0.70, 1.00, n_rows),      # small scale (supplier_reliability-like)
        np.full(n_rows, 7.0),                 # constant column -> zero variance
    ]
    return np.column_stack(cols)


def test_layer_matches_sklearn_transform():
    x_train = _train_matrix()
    scaler  = StandardScaler().fit(x_train)
    layer   = Standardize(scaler.mean_, scaler.scale_)

    x_new    = _train_matrix(n_rows=300, seed=1)
    expected = scaler.transform(x_new)
    got      = layer(torch.tensor(x_new, dtype=torch.float32)).numpy()

    # float32 buffers: 1e-5 is the honest tolerance for means in the thousands.
    np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_zero_variance_column_uses_scale_one_and_stays_finite():
    scaler = StandardScaler().fit(_train_matrix())
    assert scaler.scale_[3] == 1.0            # sklearn's guard for std == 0
    layer = Standardize(scaler.mean_, scaler.scale_)
    out   = layer(torch.tensor(_train_matrix(n_rows=5), dtype=torch.float32))
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:, 3], torch.zeros(5))


def test_statistics_are_buffers_not_parameters():
    layer = Standardize([1.0, 2.0], [3.0, 4.0])
    assert list(layer.parameters()) == []
    assert set(layer.state_dict()) == {"mean", "scale"}


def test_non_positive_scale_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        Standardize([0.0], [0.0])


def test_model_buffers_survive_state_dict_round_trip():
    metadata = {
        "n_numeric_features": 4,
        "n_warehouses": 2,
        "n_skus": 3,
        "feature_mean":  [10.0, 20.0, 30.0, 40.0],
        "feature_scale": [1.0, 2.0, 3.0, 4.0],
    }
    config = TrainingConfig(hidden_dims=(8,), dropout_rates=(0.0,))
    model  = ShortagePredictor.from_metadata(metadata, config)
    state  = model.state_dict()

    # A freshly built model with different statistics takes the saved ones.
    other = ShortagePredictor.from_metadata({**metadata, "feature_mean": [0.0] * 4}, config)
    other.load_state_dict(state)
    assert torch.equal(other.standardize.mean, torch.tensor(metadata["feature_mean"]))
    assert torch.equal(other.standardize.scale, torch.tensor(metadata["feature_scale"]))

    # And the optimizer only ever sees the trunk, heads and embeddings.
    param_names = {n for n, _ in model.named_parameters()}
    assert not any(n.startswith("standardize.") for n in param_names)
