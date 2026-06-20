###############################################################################
# Supply Chain Intelligence Platform — GCP Infrastructure
# Phase 3: GCS Data Lake + Bigtable Feature Store
###############################################################################

locals {
  bucket_name = var.gcs_bucket_name != "" ? var.gcs_bucket_name : "${var.project_id}-scip-lake"

  common_labels = {
    project     = "scip"
    environment = var.environment
    managed_by  = "terraform"
  }
}

###############################################################################
# API Enablement
# Must be enabled before any dependent resources are created.
# Each service API takes ~30s to activate on first enable.
###############################################################################

resource "google_project_service" "storage" {
  project            = var.project_id
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "bigtable" {
  project            = var.project_id
  service            = "bigtable.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "bigtable_admin" {
  project            = var.project_id
  service            = "bigtableadmin.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "iam" {
  project            = var.project_id
  service            = "iam.googleapis.com"
  disable_on_destroy = false
}

###############################################################################
# GCS Data Lake Bucket
# Single bucket, prefix-based organization:
#   bronze/    — raw Kafka events, 90-day lifecycle
#   silver/    — feature aggregates, permanent
#   checkpoints/ — Spark streaming checkpoints
#   dlq/       — dead-letter queue (Kafka DLQ is primary in Phase 3)
###############################################################################

resource "google_storage_bucket" "data_lake" {
  project  = var.project_id
  name     = local.bucket_name
  location = upper(var.region)

  uniform_bucket_level_access = true
  force_destroy               = var.environment == "dev" ? true : false

  versioning {
    enabled = false # Parquet files are immutable; versioning adds cost with no benefit
  }

  lifecycle_rule {
    condition {
      age            = var.bronze_retention_days
      matches_prefix = ["bronze/"]
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age            = 7
      matches_prefix = ["checkpoints/"]
    }
    action {
      type = "Delete"
    }
  }

  labels = local.common_labels

  depends_on = [google_project_service.storage]
}

###############################################################################
# Bigtable Instance
# DEVELOPMENT = single node, no replication, no SLA — use for dev/local testing
# PRODUCTION  = minimum 3 nodes per cluster, 99.9% SLA
###############################################################################

resource "google_bigtable_instance" "feature_store" {
  project             = var.project_id
  name                = "scip-feature-store"
  display_name        = "SCIP Feature Store"
  deletion_protection = var.environment == "prod" ? true : false

  cluster {
    cluster_id   = "scip-feature-store-c1"
    zone         = var.zone
    num_nodes    = var.bigtable_instance_type == "PRODUCTION" ? var.bigtable_num_nodes : null
    storage_type = var.bigtable_storage_type

    # autoscaling_config omitted — manual scaling, simpler for portfolio demo
  }

  instance_type = var.bigtable_instance_type
  labels        = local.common_labels

  depends_on = [
    google_project_service.bigtable,
    google_project_service.bigtable_admin,
  ]
}

###############################################################################
# Bigtable Table
# Row key: {warehouse_id}#{sku_id}
# Column families: inventory, velocity, demand, supplier, risk, lineage
###############################################################################

resource "google_bigtable_table" "inventory_features" {
  project       = var.project_id
  instance_name = google_bigtable_instance.feature_store.name
  name          = "inventory_features"

  # Split points based on warehouse_id prefix ranges.
  # Distributes load across tablet servers for range scans.
  split_keys = ["W-2", "W-4", "W-6"]

  column_family {
    family = "inventory"
  }

  column_family {
    family = "velocity"
  }

  column_family {
    family = "demand"
  }

  column_family {
    family = "supplier"
  }

  column_family {
    family = "risk"
  }

  column_family {
    family = "lineage"
  }

  depends_on = [google_bigtable_instance.feature_store]
}

###############################################################################
# Bigtable GC Policies
# One policy per column family. Mirrors the feature store design in
# docs/architecture/phase3-architecture.md.
#
# Rationale per family:
#   inventory  — max 1 version; realtime snapshot, older = stale
#   velocity   — max age 7d; rolling window, cells older than window are meaningless
#   demand     — max 1 version; forecasts are replaced not appended
#   supplier   — max age 90d; lead time history used in 90d rolling averages
#   risk       — max age 48h; stale risk scores are worse than no score
#   lineage    — max age 7d; audit window for replay debugging
###############################################################################

resource "google_bigtable_gc_policy" "inventory" {
  project         = var.project_id
  instance_name   = google_bigtable_instance.feature_store.name
  table           = google_bigtable_table.inventory_features.name
  column_family   = "inventory"
  deletion_policy = "ABANDON"

  max_version {
    number = 1
  }
}

resource "google_bigtable_gc_policy" "velocity" {
  project         = var.project_id
  instance_name   = google_bigtable_instance.feature_store.name
  table           = google_bigtable_table.inventory_features.name
  column_family   = "velocity"
  deletion_policy = "ABANDON"

  max_age {
    duration = "168h" # 7 days
  }
}

resource "google_bigtable_gc_policy" "demand" {
  project         = var.project_id
  instance_name   = google_bigtable_instance.feature_store.name
  table           = google_bigtable_table.inventory_features.name
  column_family   = "demand"
  deletion_policy = "ABANDON"

  max_version {
    number = 1
  }
}

resource "google_bigtable_gc_policy" "supplier" {
  project         = var.project_id
  instance_name   = google_bigtable_instance.feature_store.name
  table           = google_bigtable_table.inventory_features.name
  column_family   = "supplier"
  deletion_policy = "ABANDON"

  max_age {
    duration = "2160h" # 90 days
  }
}

resource "google_bigtable_gc_policy" "risk" {
  project         = var.project_id
  instance_name   = google_bigtable_instance.feature_store.name
  table           = google_bigtable_table.inventory_features.name
  column_family   = "risk"
  deletion_policy = "ABANDON"

  max_age {
    duration = "48h"
  }
}

resource "google_bigtable_gc_policy" "lineage" {
  project         = var.project_id
  instance_name   = google_bigtable_instance.feature_store.name
  table           = google_bigtable_table.inventory_features.name
  column_family   = "lineage"
  deletion_policy = "ABANDON"

  max_age {
    duration = "168h" # 7 days
  }
}

###############################################################################
# Service Account
# Used by Spark jobs (running in Docker or GKE) to authenticate to GCS/Bigtable.
# Key created manually via:
#   gcloud iam service-accounts keys create keys/scip-spark-sa.json \
#     --iam-account=<SA_EMAIL>
# Key is NOT managed by Terraform to prevent storage in state file.
###############################################################################

resource "google_service_account" "spark_sa" {
  project      = var.project_id
  account_id   = var.service_account_id
  display_name = "SCIP Spark Service Account"
  description  = "Used by Spark streaming jobs to read/write GCS and Bigtable"

  depends_on = [google_project_service.iam]
}

###############################################################################
# IAM Bindings
# Principle of least privilege: SA gets only what Spark needs.
#   storage.objectAdmin  — read/write/delete GCS objects (bronze, silver, checkpoints)
#   bigtable.user        — read/write Bigtable rows (not admin/DDL)
#   monitoring.metricWriter — emit custom metrics from Spark
#   logging.logWriter    — write structured logs to Cloud Logging
###############################################################################

resource "google_project_iam_member" "spark_gcs" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.spark_sa.email}"
}

resource "google_project_iam_member" "spark_bigtable" {
  project = var.project_id
  role    = "roles/bigtable.user"
  member  = "serviceAccount:${google_service_account.spark_sa.email}"
}

resource "google_project_iam_member" "spark_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.spark_sa.email}"
}

resource "google_project_iam_member" "spark_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.spark_sa.email}"
}
