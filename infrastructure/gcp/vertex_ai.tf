###############################################################################
# Vertex AI Infrastructure — Phase 5
# Artifact Registry, Vertex AI Endpoint, serving service account + IAM.
#
# Resources created here do NOT deploy any model or container. They provision
# the GCP scaffolding that phase5/scripts/register.py and deploy.py will target.
###############################################################################

###############################################################################
# API Enablement
###############################################################################

resource "google_project_service" "vertex_ai" {
  project            = var.project_id
  service            = "aiplatform.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifact_registry" {
  project            = var.project_id
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloud_run" {
  project            = var.project_id
  service            = "run.googleapis.com"
  disable_on_destroy = false
}

###############################################################################
# Artifact Registry
# Docker repository for serving container images.
# Image URI format: {region}-docker.pkg.dev/{project}/scip-serving/{image}:{tag}
###############################################################################

resource "google_artifact_registry_repository" "serving" {
  project       = var.project_id
  location      = var.region
  repository_id = "scip-serving"
  description   = "Docker images for SCIP ML serving containers"
  format        = "DOCKER"

  labels = local.common_labels

  depends_on = [google_project_service.artifact_registry]
}

###############################################################################
# Vertex AI Endpoint
# A dedicated endpoint for the shortage predictor model family.
# Model deployment and traffic splits are managed via phase5/scripts/deploy.py —
# not by Terraform — to avoid coupling model lifecycle to infrastructure apply.
###############################################################################

resource "google_vertex_ai_endpoint" "shortage_predictor" {
  project      = var.project_id
  location     = var.region
  name         = "shortage-predictor-endpoint"
  display_name = "SCIP Shortage Predictor"
  description  = "Serves the ShortagePredictor ONNX model for 7-day stockout risk scoring (ADR-005)"

  labels = local.common_labels

  depends_on = [google_project_service.vertex_ai]
}

###############################################################################
# Service Account — scip-vertex-sa
#
# Used by:
#   1. batch_scorer.py  — reads Bigtable features, calls endpoint.predict(),
#                          publishes results to prediction-results topic
#   2. Vertex AI serving container — reads model.onnx from GCS at startup
#
# Least-privilege rationale:
#   storage.objectViewer  — model artifact download only (not objectAdmin)
#   bigtable.user         — feature read + prediction write (row-level, not DDL)
#   pubsub.publisher      — publish to prediction-results only
#   aiplatform.user       — invoke endpoint.predict()
#   monitoring.metricWriter — emit PSI and scoring metrics
#   logging.logWriter     — structured log output
###############################################################################

resource "google_service_account" "vertex_sa" {
  project      = var.project_id
  account_id   = "scip-vertex-sa"
  display_name = "SCIP Vertex AI Service Account"
  description  = "Used by the batch scorer and Vertex AI serving container (Phase 5)"

  depends_on = [google_project_service.iam]
}

resource "google_project_iam_member" "vertex_gcs_reader" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.vertex_sa.email}"
}

resource "google_project_iam_member" "vertex_bigtable" {
  project = var.project_id
  role    = "roles/bigtable.user"
  member  = "serviceAccount:${google_service_account.vertex_sa.email}"
}

resource "google_project_iam_member" "vertex_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.vertex_sa.email}"
}

resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_sa.email}"
}

resource "google_project_iam_member" "vertex_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.vertex_sa.email}"
}

resource "google_project_iam_member" "vertex_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.vertex_sa.email}"
}
