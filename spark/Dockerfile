FROM apache/spark:3.5.3
USER root

# ── Ivy cache dir (Kafka JARs downloaded at runtime via --packages) ──────────
RUN mkdir -p /opt/ivy2 && chmod -R 777 /opt/ivy2

# ── GCS connector JAR ─────────────────────────────────────────────────────────
# Pinned version for reproducibility. Placed in Spark's jars/ dir so it is
# automatically on the driver/executor classpath without --jars flag.
# Hadoop 3 variant required — apache/spark:3.5.3 ships with Hadoop 3.3.x.
ARG GCS_CONNECTOR_VERSION=2.2.22
RUN curl -fsSL \
    "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-${GCS_CONNECTOR_VERSION}.jar" \
    -o "/opt/spark/jars/gcs-connector-hadoop3-${GCS_CONNECTOR_VERSION}.jar"

# ── App directories ───────────────────────────────────────────────────────────
# Local data dirs kept for dev/fallback when GCS_BUCKET_NAME is not set.
# SA key dir mounted at runtime from infrastructure/gcp/keys/.
RUN mkdir -p /opt/spark-app/data/bronze \
             /opt/spark-app/data/checkpoints \
             /opt/spark-app/data/dlq \
             /opt/spark-app/keys \
    && chown -R 185:185 /opt/spark-app

COPY requirements.txt /opt/spark-app/requirements.txt
RUN pip install --no-cache-dir -r /opt/spark-app/requirements.txt

COPY . /opt/spark-app/
WORKDIR /opt/spark-app
USER 185

ENTRYPOINT ["/opt/spark/bin/spark-submit"]
