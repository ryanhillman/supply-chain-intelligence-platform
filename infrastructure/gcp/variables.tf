# ── Required ──────────────────────────────────────────────────────────────────

variable "project_id" {
  description = "GCP project ID. All resources are created in this project."
  type        = string
}

# ── Regional ──────────────────────────────────────────────────────────────────

variable "region" {
  description = "GCP region for the GCS bucket and Bigtable cluster."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for the Bigtable cluster node."
  type        = string
  default     = "us-central1-a"
}

# ── Environment ───────────────────────────────────────────────────────────────

variable "environment" {
  description = "Deployment environment. Controls force_destroy on GCS and deletion_protection on Bigtable."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod"
  }
}

# ── GCS ───────────────────────────────────────────────────────────────────────

variable "gcs_bucket_name" {
  description = "Override the GCS data lake bucket name. Defaults to '{project_id}-scip-lake'."
  type        = string
  default     = ""
}

variable "bronze_retention_days" {
  description = "Days before Bronze layer objects are deleted by the bucket lifecycle rule. Must exceed Silver's longest lookback (90 days of supplier and PO history) so a partition cannot expire mid-read; 120 leaves a month of slack."
  type        = number
  default     = 120

  validation {
    condition     = var.bronze_retention_days > 90
    error_message = "bronze_retention_days must exceed Silver's 90-day lookback window."
  }
}

# ── Bigtable ──────────────────────────────────────────────────────────────────

variable "bigtable_storage_type" {
  description = "SSD for low-latency online serving. HDD for batch-heavy workloads."
  type        = string
  default     = "SSD"

  validation {
    condition     = contains(["SSD", "HDD"], var.bigtable_storage_type)
    error_message = "bigtable_storage_type must be SSD or HDD."
  }
}

variable "bigtable_num_nodes" {
  description = "Number of Bigtable nodes in the single cluster. One node (~$0.65/hour, no free tier) is plenty for the 120-row demo table."
  type        = number
  default     = 1
}

# ── IAM ───────────────────────────────────────────────────────────────────────

variable "service_account_id" {
  description = "Account ID for the Spark service account (without @project.iam.gserviceaccount.com)."
  type        = string
  default     = "scip-spark-sa"
}

variable "vertex_deployer_email" {
  description = "Google account email of the human or CI identity that runs deploy.py (e.g. user@example.com). Granted roles/iam.serviceAccountUser on scip-vertex-sa. Leave empty to skip."
  type        = string
  default     = ""
}
