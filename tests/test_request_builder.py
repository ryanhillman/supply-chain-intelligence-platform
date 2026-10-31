"""
Tests for model_serving/pipeline/request_builder.py.

RequestBuilder turns Bigtable FeatureRows into endpoint instances. Two
contracts are pinned: numeric features are passed through in raw units
(the ONNX graph standardizes them), and
stockout_risk_score is the closed-form Normal approximation computed here,
not a model output.
"""
import json

from model_serving.pipeline.feature_client import FeatureRow
from model_serving.pipeline.request_builder import RequestBuilder


def _metadata(tmp_path):
    (tmp_path / "feature_metadata.json").write_text(json.dumps({
        "warehouse_vocab": {"W-001": 0, "W-002": 1},
        "sku_vocab":       {"SKU-001": 0, "SKU-002": 1},
        "horizon_days":    7,
    }))
    return tmp_path


def _row(key, wh, sku, **features):
    return FeatureRow(row_key=key, warehouse_id=wh, sku_id=sku, features=features)


def test_raw_features_pass_through_and_risk_score_is_closed_form(tmp_path):
    builder = RequestBuilder(artifacts_dir=_metadata(tmp_path))
    rows = [
        _row("W-001#SKU-001", "W-001", "SKU-001",
             current_qty=0.0,   avg_daily_sales_7d=10.0, stddev_daily_7d=2.0),
        _row("W-002#SKU-002", "W-002", "SKU-002",
             current_qty=500.0, avg_daily_sales_7d=10.0, stddev_daily_7d=2.0),
        _row("W-009#SKU-001", "W-009", "SKU-001", current_qty=5.0),   # unknown warehouse
    ]
    requests = builder.build(rows)

    assert [r.row_key for r in requests] == ["W-001#SKU-001", "W-002#SKU-002"]

    out_of_stock, well_stocked = requests
    assert out_of_stock.stockout_risk_score == 1.0
    assert 0.0 <= well_stocked.stockout_risk_score < out_of_stock.stockout_risk_score

    # Raw units, NUMERIC_FEATURE_COLS order (current_qty is first), 21 values.
    assert well_stocked.instance["numeric_features"][0] == 500.0
    assert len(well_stocked.instance["numeric_features"]) == 21
    assert well_stocked.instance["warehouse_id"] == 1
    assert well_stocked.instance["sku_id"] == 1
