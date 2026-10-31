###############################################################################
# Supply Chain Intelligence Platform — GCP Infrastructure
# Phase 3: GCS Data Lake + Bigtable Feature Store
# Phase 5: predictions column family (added to existing inventory_features table)
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
#   bronze/    — raw Kafka events, deleted after var.bronze_retention_days (120d)
#   silver/    — feature aggregates, permanent
#   checkpoints/ — Spark streaming checkpoints. No lifecycle rule: Spark bounds
#                its own checkpoint growth (minBatchesToRetain, log compaction),
#                and an age rule deleted the live offsets after a pause, so the
#                Bronze query restarted from "latest" and silently skipped the
#                Kafka data produced in between. Remove abandoned checkpoint
#                directories by hand.
#   dlq/       — dead-letter queue (newline-delimited JSON envelopes)
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

  labels = local.common_labels

  depends_on = [google_project_service.storage]
}

###############################################################################
# Bigtable Instance
# One PRODUCTION node. Google deprecated the DEVELOPMENT instance type
# ("Prefer PRODUCTION for all use cases"); a 1-node PRODUCTION instance costs
# the same (~$0.65/node-hour, no free tier) without the restrictions. This is
# the project's largest line item — destroy the instance between demo sessions
# (Silver is the source of truth; the next Silver run repopulates Bigtable).
###############################################################################

resource "google_bigtable_instance" "feature_store" {
  project             = var.project_id
  name                = "scip-feature-store"
  display_name        = "SCIP Feature Store"
  deletion_protection = var.environment == "prod" ? true : false

  cluster {
    cluster_id   = "scip-feature-store-c1"
    zone         = var.zone
    num_nodes    = var.bigtable_num_nodes
    storage_type = var.bigtable_storage_type

    # autoscaling_config omitted — manual scaling, simpler for portfolio demo
  }

  # instance_type is intentionally unset: the provider defaults to PRODUCTION
  # and the field is deprecated (DEVELOPMENT instances are being converted).
  labels = local.common_labels

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

  # Phase 5 — ML prediction outputs, kept separate from feature families.
  # Columns: shortage_prob_7d, stockout_risk_score, model_version,
  #          prediction_timestamp.
  # GC policy: max 3 versions (current + 2 prior rounds for rollback inspection).
  column_family {
    family = "predictions"
  }

  depends_on = [google_bigtable_instance.feature_store]
}

###############################################################################
# Bigtable GC Policies
# One policy per column family.
#
# Every feature family keeps exactly the latest cell (max_version = 1): Silver
# rewrites the whole row on each run and every reader takes the newest cell
# only, so older versions are never read. Age-based expiry (velocity 7d, risk
# 48h, lineage 7d, supplier 90d) was removed — this store is written in bursts
# around demo sessions, and max_age turned an idle store into a partially
# empty one that the scorer back-filled with defaults and scored as all-clear.
# Freshness is enforced by the scorer instead (lineage:silver_ts, see
# model_serving/pipeline/feature_client.py split_fresh()).
#
#   predictions — max 3 versions: current + 2 prior rounds for rollback
#                 inspection
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

  max_version {
    number = 1
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

  max_version {
    number = 1
  }
}

resource "google_bigtable_gc_policy" "risk" {
  project         = var.project_id
  instance_name   = google_bigtable_instance.feature_store.name
  table           = google_bigtable_table.inventory_features.name
  column_family   = "risk"
  deletion_policy = "ABANDON"

  max_version {
    number = 1
  }
}

resource "google_bigtable_gc_policy" "lineage" {
  project         = var.project_id
  instance_name   = google_bigtable_instance.feature_store.name
  table           = google_bigtable_table.inventory_features.name
  column_family   = "lineage"
  deletion_policy = "ABANDON"

  max_version {
    number = 1
  }
}

# Phase 5 — Keep 3 versions: current prediction + 2 prior rounds.
# Allows champion/challenger rollback inspection without unbounded growth.
resource "google_bigtable_gc_policy" "predictions" {
  project         = var.project_id
  instance_name   = google_bigtable_instance.feature_store.name
  table           = google_bigtable_table.inventory_features.name
  column_family   = "predictions"
  deletion_policy = "ABANDON"

  max_version {
    number = 3
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
#   storage.objectAdmin  — read/write/delete objects in the data-lake bucket only
#                          (bronze, silver, checkpoints, dlq); granted on the
#                          bucket, not the project, like the Vertex SA
#   bigtable.user        — read/write Bigtable rows (not admin/DDL)
#   monitoring.metricWriter — emit custom metrics from Spark
#   logging.logWriter    — write structured logs to Cloud Logging
###############################################################################

resource "google_storage_bucket_iam_member" "spark_gcs" {
  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.spark_sa.email}"
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
