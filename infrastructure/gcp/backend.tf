# Remote state stored in GCS.
#
# IMPORTANT: The state bucket must exist BEFORE running `terraform init`.
# Create it once manually:
#
#   gcloud storage buckets create gs://YOUR_PROJECT_ID-scip-tf-state \
#     --location=us-central1 \
#     --uniform-bucket-level-access
#
# Then replace YOUR_PROJECT_ID below and run `terraform init`.
# The bucket name cannot reference variables — Terraform limitation.

terraform {
  backend "gcs" {
    bucket = "YOUR_PROJECT_ID-scip-tf-state"
    prefix = "supply-chain-platform/terraform/state"
  }
}
