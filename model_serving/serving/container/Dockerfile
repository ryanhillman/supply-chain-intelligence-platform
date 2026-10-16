# SCIP Shortage Predictor — Vertex AI serving container
#
# Build:
#   docker build -t shortage-predictor-serving:local .
#
# Run locally (see smoke-test instructions in model_serving/serving/smoke_test.md):
#   docker run -p 8080:8080 \
#     -e MODEL_GCS_PATH=gs://<bucket>/models/shortage-predictor/<version> \
#     -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/key.json \
#     -v /path/to/scip-vertex-sa.json:/tmp/key.json:ro \
#     shortage-predictor-serving:local
#
# Push to Artifact Registry (after terraform apply provisions the repo):
#   docker tag shortage-predictor-serving:local \
#     us-central1-docker.pkg.dev/<project>/scip-serving/shortage-predictor-serving:<version>
#   docker push \
#     us-central1-docker.pkg.dev/<project>/scip-serving/shortage-predictor-serving:<version>
#
# This container does NOT embed model.onnx. The model is downloaded from GCS
# at startup using MODEL_GCS_PATH. The same image serves any model version.

FROM python:3.11-slim

# libgomp1 is required by onnxruntime-cpu for OpenMP thread parallelism.
# Without it, onnxruntime raises "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for container security hardening.
# /tmp is world-writable, so model download to /tmp/model still works.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py predictor.py ./

# Switch to non-root before CMD.
USER appuser

# Vertex AI default prediction port is 8080.
# Override with --port at deploy time if needed.
EXPOSE 8080

# workers=1: onnxruntime InferenceSession is not fork-safe.
# For higher throughput, scale via Vertex AI min/max replicas, not workers.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
