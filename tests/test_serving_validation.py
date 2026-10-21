"""
Regression test for model_serving/serving/container/main.py's vocab-range
validation, fixed in this branch: an out-of-range warehouse_id/sku_id
(e.g. warehouse_id: 99) reached the ONNX embedding Gather op at inference
time and raised there, landing in the generic
`except Exception -> HTTPException(500)` handler, instead of being
rejected as a 422 the way the documented contract (and FastAPI's own
pydantic validation) implies.

predictor.py's own ONNXPredictor.__init__ needs GCS + onnxruntime model
files, so this test swaps in a fake predictor exposing just the
n_warehouses/n_skus/predict() surface main.py actually uses.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

_CONTAINER_DIR = Path(__file__).parents[1] / "model_serving" / "serving" / "container"


@pytest.fixture()
def client():
    if str(_CONTAINER_DIR) not in sys.path:
        sys.path.insert(0, str(_CONTAINER_DIR))
    import main as main_module  # noqa: PLC0415 -- path-dependent import by design

    class FakePredictor:
        n_warehouses = 6
        n_skus = 20
        model_version = "test"
        n_numeric_features = 21

        def predict(self, numeric_features, warehouse_ids, sku_ids):
            n = len(warehouse_ids)
            return (
                np.full(n, 0.5, dtype="float32"),
                np.full(n, 0.5, dtype="float32"),
            )

    main_module._predictor = FakePredictor()
    return TestClient(main_module.app)


def _instance(warehouse_id=0, sku_id=0):
    return {"instances": [{"numeric_features": [0.0] * 21, "warehouse_id": warehouse_id, "sku_id": sku_id}]}


def test_in_range_request_succeeds(client):
    response = client.post("/predict", json=_instance(warehouse_id=0, sku_id=0))
    assert response.status_code == 200


def test_out_of_range_warehouse_id_returns_422_not_500(client):
    response = client.post("/predict", json=_instance(warehouse_id=99, sku_id=0))
    assert response.status_code == 422
    assert "warehouse_id" in response.json()["detail"]


def test_out_of_range_sku_id_returns_422_not_500(client):
    response = client.post("/predict", json=_instance(warehouse_id=0, sku_id=500))
    assert response.status_code == 422
    assert "sku_id" in response.json()["detail"]


def test_boundary_sku_id_equal_to_vocab_size_is_rejected(client):
    """sku_id must be strictly < n_skus (valid indices are 0..n_skus-1)."""
    response = client.post("/predict", json=_instance(warehouse_id=0, sku_id=20))
    assert response.status_code == 422


def test_negative_warehouse_id_still_rejected_by_pydantic(client):
    """Pre-existing validation (not part of this fix) must still work."""
    response = client.post("/predict", json=_instance(warehouse_id=-1, sku_id=0))
    assert response.status_code == 422
