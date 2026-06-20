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
