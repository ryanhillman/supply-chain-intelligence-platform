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

## 1. Kafka and the replay

```
docker compose -f kafka/docker-compose.yml up -d
```

This creates the `kafka_scip-net` network the other stacks join, creates the
five topics (30-day retention), and starts the producer. The producer replays
365 simulated days (2025-10-01 to 2026-09-30, seed 42) from
`simulation/world.py`, about 200,000 events, and exits within a minute;
`docker logs scip-producer` ends with `Replay complete`. `REPLAY_START`,
`REPLAY_DAYS` and `SEED` in `kafka/docker-compose.yml` choose another world.
Replaying the same seed again re-sends identical events, which Silver and the
training sampler drop as duplicates.

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

Bronze reads the topics from the earliest offset and writes Parquet under
`spark/data/bronze/<topic>/event_type=/year=/month=/day=/`. It has caught up
when its log stops printing `write_bronze` lines, about 10 minutes for the
default replay on Docker Desktop. Then run Silver (stop Bronze first if Docker
Desktop is short on memory; each job takes up to 1.5 GB):

```
docker compose -f spark/docker-compose.spark.yml stop spark-bronze
docker compose -f spark/docker-compose.spark.yml run --rm spark-silver-batch
```

Silver computes one feature row per (warehouse, SKU) as of the latest day in
Bronze (2026-09-30 for the default replay), writes it to `spark/data/silver/`,
and writes the same rows into the Bigtable emulator. It takes about 5 minutes.
Pass `-e SILVER_AS_OF=YYYY-MM-DD` to compute an earlier date.

## 4. Build the dataset and train

```
python ml_training/scripts/build_dataset.py --bronze-path spark/data/bronze --output-dir ml_training/artifacts/dataset
python ml_training/scripts/train.py --dataset ml_training/artifacts/dataset --artifacts ml_training/artifacts
```

The sampler builds one row per (warehouse, SKU, day) from the replayed year,
with the same feature windows Silver uses, and labels each row from the
following 7 days of events. `train.py` writes `model.pt`, `model.onnx`,
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
