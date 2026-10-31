# Phase 5.2 — Serving Container Smoke Test

Two methods: **uvicorn** (no Docker required) and **Docker** (requires Docker build).
Run the uvicorn method first to verify the Python code before building the image.

---

## Prerequisites

1. The Phase 4 training artifacts exist at `ml_training/artifacts/`:
   - `model.onnx`
   - `model.onnx.data`
   - `feature_metadata.json`

2. These artifacts have been uploaded to GCS by `model_serving/scripts/register.py`
   (Phase 5.3). Until then, use a local path override for the uvicorn method.

3. `scip-vertex-sa.json` exists at `infrastructure/gcp/keys/scip-vertex-sa.json`.

---

## Method 1 — uvicorn (no Docker)

Runs the FastAPI app directly. Fastest way to verify logic before building the image.

### 1a. Install dependencies

```powershell
# From repo root
cd model_serving/serving/container
pip install -r requirements.txt
```

### 1b. Set environment variables

```powershell
$env:MODEL_GCS_PATH = "gs://supply-chain-platform-499920-scip-lake/models/shortage-predictor/v20260620_180918"
$env:GOOGLE_APPLICATION_CREDENTIALS = "infrastructure/gcp/keys/scip-vertex-sa.json"
```

### 1c. Start the server

```powershell
# From model_serving/serving/container/
$env:PYTHONIOENCODING = "utf-8"
uvicorn main:app --host 0.0.0.0 --port 8080
```

Expected startup log:
```
INFO  scip.serving — Container startup — loading ONNX model from GCS
INFO  predictor   — Downloaded gs://.../model.onnx -> /tmp/model/model.onnx (...)
INFO  predictor   — Downloaded gs://.../model.onnx.data -> /tmp/model/model.onnx.data (...)
INFO  predictor   — ONNXPredictor ready | version=v20260620_180918 | inputs=[...]
INFO  scip.serving — Container ready
INFO  uvicorn     — Application startup complete.
```

### 1d. Health check

```powershell
# New terminal
Invoke-RestMethod -Uri http://localhost:8080/health
```

Expected:
```json
{
  "status": "ok",
  "model_version": "v20260620_180918",
  "n_numeric_features": 21
}
```

### 1e. Prediction request

The 21 numeric features below are **all-zero raw values** (zero stock, zero sales)
— valid shape, not a meaningful entity. The check is: response has 2 instances, both `shortage_prob_7d`
and `stockout_risk_score` are finite floats in [0, 1].

```powershell
$body = @{
    instances = @(
        @{
            numeric_features = @(0.0) * 21
            warehouse_id     = 0
            sku_id           = 0
        },
        @{
            numeric_features = @(0.0) * 21
            warehouse_id     = 1
            sku_id           = 5
        },
        @{
            numeric_features = @(0.0) * 21
            warehouse_id     = 2
            sku_id           = 10
        }
    )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method POST `
    -Uri http://localhost:8080/predict `
    -ContentType "application/json" `
    -Body $body
```

Expected response shape:
```json
{
  "predictions": [
    {"shortage_prob_7d": <float in [0,1]>, "stockout_risk_score": <float in [0,1]>},
    {"shortage_prob_7d": <float in [0,1]>, "stockout_risk_score": <float in [0,1]>},
    {"shortage_prob_7d": <float in [0,1]>, "stockout_risk_score": <float in [0,1]>}
  ]
}
```

### 1f. Validation error check (schema guard)

Send only 20 features (should return HTTP 422, not 500):

```powershell
$bad_body = @{
    instances = @(
        @{
            numeric_features = @(0.0) * 20   # wrong: needs 21
            warehouse_id     = 0
            sku_id           = 0
        }
    )
} | ConvertTo-Json -Depth 5

try {
    Invoke-RestMethod -Method POST `
        -Uri http://localhost:8080/predict `
        -ContentType "application/json" `
        -Body $bad_body
} catch {
    $_.Exception.Response.StatusCode   # should print 422
    $_.ErrorDetails.Message            # should mention "21 elements"
}
```

---

## Method 2 — Docker (requires Docker build)

Run after Method 1 passes. Do not build until Phase 5.2 is approved.

```powershell
# Build (from model_serving/serving/container/)
docker build -t shortage-predictor-serving:local .

# Run
docker run --rm -p 8080:8080 `
    -e MODEL_GCS_PATH="gs://supply-chain-platform-499920-scip-lake/models/shortage-predictor/v20260620_180918" `
    -e GOOGLE_APPLICATION_CREDENTIALS="/tmp/key.json" `
    -v "${PWD}\infrastructure\gcp\keys\scip-vertex-sa.json:/tmp/key.json:ro" `
    shortage-predictor-serving:local
```

Then run steps 1d–1f above against `http://localhost:8080`.

### Verify non-root user

```powershell
docker run --rm --entrypoint whoami shortage-predictor-serving:local
# Expected: appuser
```

### Verify libgomp is present

```powershell
docker run --rm shortage-predictor-serving:local python -c "import onnxruntime; print(onnxruntime.__version__)"
# Expected: 1.19.2 (no libgomp error)
```

---

## Pass / Fail Criteria

| Check | Expected |
|---|---|
| Server starts | Log: "Container ready" |
| `/health` returns 200 | `{"status": "ok", ...}` |
| `/predict` with 3 instances returns 3 predictions | All `shortage_prob_7d` and `stockout_risk_score` in [0.0, 1.0] |
| `/predict` with 20 features returns 422 | Error detail mentions "21 elements" |
| Docker `whoami` returns `appuser` | Not root |
| Docker onnxruntime import succeeds | No libgomp error |

If `/predict` returns `shortage_prob_7d = 0.0` and `stockout_risk_score = 1.0`
for all instances with zero input, this is expected — it reflects the same
distribution shift documented in ADR-005 Decision 7. The pipeline is correct;
the values reflect model saturation on out-of-distribution input.
