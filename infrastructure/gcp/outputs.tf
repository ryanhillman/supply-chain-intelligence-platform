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

output "spark_sa_key_command" {
  description = "Command to generate SA key (run manually — key NOT managed by Terraform)"
  value       = <<-EOT
    mkdir -p infrastructure/gcp/keys
    gcloud iam service-accounts keys create infrastructure/gcp/keys/scip-spark-sa.json \
      --iam-account=${google_service_account.spark_sa.email} \
      --project=${var.project_id}
  EOT
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

output "next_steps" {
  description = "Manual steps required after terraform apply"
  value       = <<-EOT
    1. Create SA key:
       gcloud iam service-accounts keys create infrastructure/gcp/keys/scip-spark-sa.json \
         --iam-account=${google_service_account.spark_sa.email}

    2. Update spark/.env:
       GCS_BUCKET_NAME=${google_storage_bucket.data_lake.name}
       BIGTABLE_INSTANCE_ID=${google_bigtable_instance.feature_store.name}
       BIGTABLE_TABLE_ID=${google_bigtable_table.inventory_features.name}
       GOOGLE_APPLICATION_CREDENTIALS=/opt/spark-app/keys/scip-spark-sa.json

    3. Verify bucket:
       gsutil ls gs://${google_storage_bucket.data_lake.name}

    4. Verify Bigtable:
       cbt -project ${var.project_id} -instance ${google_bigtable_instance.feature_store.name} ls
  EOT
}

###############################################################################
# Phase 5 Outputs
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
  description = "Service account email for the prediction writer Cloud Run job."
  value       = google_service_account.prediction_writer_sa.email
}

output "model_serving_next_steps" {
  description = "Manual steps required after Phase 5 terraform apply"
  value       = <<-EOT
    Phase 5 infrastructure provisioned. Next steps (do not run terraform apply yet):

    1. Create SA keys for Phase 5 service accounts:
       gcloud iam service-accounts keys create infrastructure/gcp/keys/scip-vertex-sa.json \
         --iam-account=${google_service_account.vertex_sa.email}
       gcloud iam service-accounts keys create infrastructure/gcp/keys/scip-prediction-writer-sa.json \
         --iam-account=${google_service_account.prediction_writer_sa.email}

    2. Configure Docker for Artifact Registry:
       gcloud auth configure-docker ${var.region}-docker.pkg.dev

    3. Build and push the serving container (Phase 5.2):
       docker build -t ${var.region}-docker.pkg.dev/${var.project_id}/scip-serving/shortage-predictor-serving:latest \
         model_serving/serving/container/
       docker push ${var.region}-docker.pkg.dev/${var.project_id}/scip-serving/shortage-predictor-serving:latest

    4. Register the model (Phase 5.3):
       python model_serving/scripts/register.py --bucket $GCS_BUCKET_NAME \
         --project ${var.project_id} \
         --image-uri ${var.region}-docker.pkg.dev/${var.project_id}/scip-serving/shortage-predictor-serving:latest \
         --sa-key infrastructure/gcp/keys/scip-vertex-sa.json

    5. Deploy the endpoint (Phase 5.3):
       python model_serving/scripts/deploy.py deploy \
         --project ${var.project_id} \
         --endpoint ${google_vertex_ai_endpoint.shortage_predictor.id} \
         --traffic 100 \
         --sa-key infrastructure/gcp/keys/scip-vertex-sa.json

    6. Verify Bigtable predictions column family:
       cbt -project ${var.project_id} -instance ${google_bigtable_instance.feature_store.name} \
         ls inventory_features
  EOT
}
