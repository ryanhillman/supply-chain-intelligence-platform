# Running the platform locally (no GCP)

The whole pipeline also runs on one machine with no GCP project. Kafka and Spark
were always local, and this directory adds the pieces that stand in for the GCP
services.

| In GCP (June 2026)                | Locally                                                                              |
|-----------------------------------|--------------------------------------------------------------------------------------|
| GCS Bronze / Silver / checkpoints | `spark/data/` on the host (the `spark/` directory is mounted at `/opt/spark-app`)    |
| Bigtable feature store            | Bigtable emulator, `localhost:8086` on the host, `bigtable-emulator:8086` in Docker  |
| Pub/Sub prediction topic          | Pub/Sub emulator, `localhost:8085`                                                   |
| Vertex AI endpoint                | the same serving container, `docker run -p 8080:8080` with `MODEL_LOCAL_PATH`        |
| Vertex Model Registry + promotion | `deploy.py promote --local` between two artifact directories                         |

The Terraform, Vertex AI and GCS code paths are unchanged; the local run only
changes what the clients connect to. No service-account key is needed anywhere
in the local run.

## 1. Kafka and the event producer

```
docker compose -f kafka/docker-compose.yml up -d
```

This also creates the `kafka_scip-net` network that the other stacks join.

## 2. Emulators

```
docker compose -f local/docker-compose.emulators.yml up -d
python local/init_emulator.py
```

`init_emulator.py` creates the `inventory_features` table with its column
families and GC rules, and the `prediction-results` topic with the
`prediction-writer` subscription, matching what Terraform created in GCP. The
emulators are in-memory: rerun it after `docker compose restart`.

Host-side tools (scorer, monitor, dashboards) find the emulators through
environment variables, which the script prints:

```
# PowerShell
$env:BIGTABLE_EMULATOR_HOST = "localhost:8086"
$env:PUBSUB_EMULATOR_HOST   = "localhost:8085"
$env:GCP_PROJECT_ID         = "local-scip"
```

## 3. Bronze and Silver (Spark)

```
cp spark/.env.example spark/.env          # local paths and the emulator host are the defaults
docker build -t scip-spark:latest spark
docker compose -f spark/docker-compose.spark.yml up -d spark-bronze
```

Bronze writes Parquet under `spark/data/bronze/<topic>/...` every 30 seconds.
After a few minutes, run Silver once over everything in Bronze (stop Bronze
first if Docker Desktop is short on memory; the two jobs are 1.5 GB each):

```
docker compose -f spark/docker-compose.spark.yml stop spark-bronze
docker compose -f spark/docker-compose.spark.yml run --rm spark-silver-batch
```

Silver writes `spark/data/silver/` and one feature row per (warehouse, SKU) into
the Bigtable emulator. `spark-silver` (the streaming variant) is still there for
`up -d`, as in the June deployment.

## 4. Build the dataset and train

```
python ml_training/scripts/build_dataset.py --bronze-path spark/data/bronze --simulate-days 150 --output-dir ml_training/artifacts/dataset
python ml_training/scripts/train.py --dataset ml_training/artifacts/dataset --artifacts ml_training/artifacts
```

A few minutes of Bronze is far less than the 38 days the sampler needs, so
`build_dataset.py` tiles the events across simulated days the way it did
during development; 150 days leaves enough sample dates for the 7-day purge
gaps between the time splits (the default of 60 does not). The labels from a
capture this short are degenerate (no stockouts happen), so this trains a
plumbing check, not a model; the one-world replay replaces the tiling with a
simulated year. `train.py` writes `model.pt`, `model.onnx`,
`feature_metadata.json`, `evaluation_test.json`, `training_metrics.json` and
`drift_reference.json`.

## 5. Serve the model

```
docker build -t shortage-predictor-serving:local model_serving/serving/container
docker run -d --name scip-serving -p 8080:8080 `
    -e MODEL_LOCAL_PATH=/model -e MODEL_VERSION=local-dev `
    -v "${PWD}\ml_training\artifacts:/model:ro" shortage-predictor-serving:local
```

`Invoke-RestMethod http://localhost:8080/health` reports the version and the
feature width. The container is the one Vertex AI ran; only the model source
differs (`MODEL_LOCAL_PATH` instead of `MODEL_GCS_PATH`).

## 6. Score, write, monitor

```
python model_serving/scripts/deploy.py smoke-test --project local-scip --endpoint-url http://localhost:8080 --bigtable-instance scip-feature-store
python model_serving/scripts/score.py --project local-scip --endpoint-url http://localhost:8080 --model-version local-dev --write
python model_serving/scripts/monitor.py --project local-scip
python model_serving/scripts/show_predictions.py --project local-scip
python model_serving/scripts/show_feature_sample.py --project local-scip
```

`score.py` reads the feature rows from the emulator, calls the local container,
publishes one message per entity to the Pub/Sub emulator, and with `--write`
drains the subscription into the `predictions` column family. The monitor and
both dashboards read the same emulator.

## 7. Promotion gate, offline

```
python model_serving/scripts/deploy.py promote --local `
    --champion-dir ml_training/artifacts_v1 --challenger-dir ml_training/artifacts `
    --dataset ml_training/artifacts/dataset
```

Same three-metric gate as the Vertex AI path (AUPRC, AUC-ROC, Brier): the stored
`evaluation_test.json` metrics when both runs were evaluated on the same test
split, otherwise both models rescored on the challenger's split with
onnxruntime. Nothing is deployed; serve the winner by pointing
`MODEL_LOCAL_PATH` at its directory.
