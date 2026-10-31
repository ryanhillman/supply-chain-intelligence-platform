"""
model_serving/pipeline/local_endpoint.py and the scorer's endpoint_url mode:
the batch scorer reaches the serving container over HTTP with the same
`.predict(instances=...).predictions` shape it reads from aiplatform.Endpoint.

The server here is a stdlib HTTP handler speaking the container's contract
(POST /predict {"instances": [...]} -> {"predictions": [...]}), so the test
needs neither Docker nor the container's dependencies.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from model_serving.pipeline.batch_scorer import BatchScorer
from model_serving.pipeline.local_endpoint import LocalEndpoint


class _FakeContainer(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_GET(self):
        self._reply(200, {"status": "ok", "model_version": "fake", "n_numeric_features": 21})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append(body)
        instances = body.get("instances", [])
        if any(len(i["numeric_features"]) != 21 for i in instances):
            self._reply(422, {"detail": "numeric_features must have 21 elements"})
            return
        self._reply(200, {"predictions": [{"shortage_prob_7d": 0.25 + 0.1 * k} for k in range(len(instances))]})

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture()
def container_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeContainer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _FakeContainer.seen = []
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


def _instance(k: int) -> dict:
    return {"numeric_features": [float(k)] * 21, "warehouse_id": 0, "sku_id": k}


def test_predict_speaks_the_vertex_contract(container_url):
    endpoint = LocalEndpoint(container_url + "/")   # trailing slash is tolerated

    response = endpoint.predict(instances=[_instance(0), _instance(1)])

    assert [p["shortage_prob_7d"] for p in response.predictions] == [0.25, 0.35]
    assert _FakeContainer.seen == [{"instances": [_instance(0), _instance(1)]}]
    assert endpoint.health()["n_numeric_features"] == 21


def test_http_errors_surface_the_container_detail(container_url):
    bad = {"numeric_features": [0.0] * 20, "warehouse_id": 0, "sku_id": 0}
    with pytest.raises(RuntimeError, match="HTTP 422.*21 elements"):
        LocalEndpoint(container_url).predict(instances=[bad])


def test_batch_scorer_uses_the_local_endpoint_without_vertex(container_url, monkeypatch):
    # The publisher client is constructed eagerly; point it at an emulator
    # address so no credentials are looked up (nothing is published here).
    monkeypatch.setenv("PUBSUB_EMULATOR_HOST", "localhost:1")

    scorer = BatchScorer(
        project_id="local-scip", region="us-central1",
        feature_client=None, request_builder=None,
        endpoint_url=container_url,
    )

    class Req:
        def __init__(self, k):
            self.instance = _instance(k)

    probs, latency_ms = scorer._score_batch([Req(0), Req(1), Req(2)])
    assert probs == [0.25, 0.35, 0.45]
    assert latency_ms >= 0.0


def test_batch_scorer_requires_exactly_one_endpoint(monkeypatch):
    monkeypatch.setenv("PUBSUB_EMULATOR_HOST", "localhost:1")
    with pytest.raises(ValueError, match="exactly one"):
        BatchScorer(project_id="p", region="r", feature_client=None, request_builder=None)
    with pytest.raises(ValueError, match="exactly one"):
        BatchScorer(project_id="p", region="r", feature_client=None, request_builder=None,
                    endpoint_resource_name="projects/p/locations/r/endpoints/1",
                    endpoint_url="http://localhost:8080")
