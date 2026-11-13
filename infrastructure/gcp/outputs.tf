###############################################################################
# Outputs — values needed by Spark jobs and CI/CD
###############################################################################

output "gcs_bucket_name" {
  description = "GCS data lake bucket name. Set GCS_BUCKET_NAME env var in spark/.env"
  value       = google_storage_bucket.data_lake.name
}

output "gcs_bucket_url" {
  description = "GCS bucket URI (gs://...). Use as base path in Spark jobs"
  value       = "gs://${google_storage_bucket.data_lake.name}"
}

output "bigtable_instance_id" {
  description = "Bigtable instance ID. Set BIGTABLE_INSTANCE_ID env var in spark/.env"
  value       = google_bigtable_instance.feature_store.name
}

output "bigtable_table_id" {
  description = "Bigtable table ID for inventory features"
  value       = google_bigtable_table.inventory_features.name
}

output "spark_sa_email" {
  description = "Service account email for Spark jobs. Use when creating SA key via gcloud"
  value       = google_service_account.spark_sa.email
}

output "bronze_path" {
  description = "GCS URI for bronze layer"
  value       = "gs://${google_storage_bucket.data_lake.name}/bronze"
}

output "silver_path" {
  description = "GCS URI for silver layer"
  value       = "gs://${google_storage_bucket.data_lake.name}/silver"
}

output "checkpoint_path" {
  description = "GCS URI for Spark checkpoints"
  value       = "gs://${google_storage_bucket.data_lake.name}/checkpoints"
}

###############################################################################
# Model serving outputs
###############################################################################

output "artifact_registry_url" {
  description = "Docker repository URL. Tag and push serving images here: {url}/{image}:{tag}"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.serving.name}"
}

output "vertex_ai_endpoint_resource_name" {
  description = "Full Vertex AI endpoint resource name. Pass to aiplatform.Endpoint() in Python."
  value       = google_vertex_ai_endpoint.shortage_predictor.id
}

output "prediction_results_topic" {
  description = "Pub/Sub topic name for prediction results. batch_scorer.py publishes here."
  value       = google_pubsub_topic.prediction_results.name
}

output "prediction_results_dlq_topic" {
  description = "Dead-letter topic name. Messages failing 5 deliveries land here."
  value       = google_pubsub_topic.prediction_results_dlq.name
}

output "vertex_sa_email" {
  description = "Service account email for the batch scorer and Vertex AI serving container."
  value       = google_service_account.vertex_sa.email
}

output "prediction_writer_sa_email" {
  description = "Service account email for the prediction writer."
  value       = google_service_account.prediction_writer_sa.email
}
