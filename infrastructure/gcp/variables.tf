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
  description = "Days before Bronze layer objects are deleted by lifecycle rule."
  type        = number
  default     = 90

  validation {
    condition     = var.bronze_retention_days >= 7
    error_message = "bronze_retention_days must be at least 7 to allow Silver reprocessing windows."
  }
}

# ── Bigtable ──────────────────────────────────────────────────────────────────

variable "bigtable_instance_type" {
  description = "DEVELOPMENT = single node, no SLA. PRODUCTION = minimum 3 nodes, 99.9% SLA."
  type        = string
  default     = "DEVELOPMENT"

  validation {
    condition     = contains(["DEVELOPMENT", "PRODUCTION"], var.bigtable_instance_type)
    error_message = "bigtable_instance_type must be DEVELOPMENT or PRODUCTION."
  }
}

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
  description = "Number of Bigtable nodes (PRODUCTION only; ignored for DEVELOPMENT)."
  type        = number
  default     = 1
}

# ── IAM ───────────────────────────────────────────────────────────────────────

variable "service_account_id" {
  description = "Account ID for the Spark service account (without @project.iam.gserviceaccount.com)."
  type        = string
  default     = "scip-spark-sa"
}
