###############################################################################
# Vertex AI Infrastructure — Phase 5
# Artifact Registry, Vertex AI Endpoint, serving service account + IAM.
#
# Resources created here do NOT deploy any model or container. They provision
# the GCP scaffolding that model_serving/scripts/register.py and deploy.py will target.
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
# Model deployment and traffic splits are managed via model_serving/scripts/deploy.py —
# not by Terraform — to avoid coupling model lifecycle to infrastructure apply.
###############################################################################

resource "google_vertex_ai_endpoint" "shortage_predictor" {
  project      = var.project_id
  location     = var.region
  name         = "shortage-predictor-endpoint"
  display_name = "SCIP Shortage Predictor"
  description  = "Serves the ShortagePredictor ONNX model for 7-day stockout risk scoring"

  labels = local.common_labels

  depends_on = [google_project_service.vertex_ai]
}

###############################################################################
# Service Account — scip-vertex-sa
#
# Used by:
#   1. batch_scorer.py  — reads Bigtable features, calls endpoint.predict(),
#                          publishes results to prediction-results topic
#   2. Vertex AI serving container — downloads model.onnx from GCS at startup
#      (gs://{bucket}/models/shortage-predictor/{version}/, set via
#      MODEL_GCS_PATH — see register_model.py)
#   3. register_model.py — uploads trained model artifacts to that same
#      models/shortage-predictor/ prefix when registering a new version
#
# Least-privilege rationale:
#   storage.objectAdmin, scoped by IAM condition to the models/ prefix only —
#     covers both the upload (register_model.py) and download (serving
#     container) paths without granting any access to the Bronze/Silver raw
#     data elsewhere in the same bucket.
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

resource "google_storage_bucket_iam_member" "vertex_gcs_lake" {
  bucket = google_storage_bucket.data_lake.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.vertex_sa.email}"

  condition {
    title       = "models-prefix-only"
    description = "Limits object access to models/ artifacts (upload on register, download on serve) — no access to Bronze/Silver raw data."
    expression  = "resource.name.startsWith(\"projects/_/buckets/${google_storage_bucket.data_lake.name}/objects/models/\")"
  }
}

# The condition above is evaluated per-object for get/create/delete, but
# NOT for list: GCS checks storage.objects.list against the bucket as the
# resource (there's no object yet to test the condition against), so a
# resource.name condition scoped to a prefix denies list_blobs() entirely
# — breaking both predictor.py's model download and register.py's upload.
# Grant list unconditionally at the bucket level instead; list only
# reveals object names/sizes/metadata, not Bronze/Silver content, so this
# doesn't undermine the models/-prefix restriction above.
resource "google_project_iam_custom_role" "vertex_gcs_list_only" {
  project     = var.project_id
  role_id     = "scipVertexGcsListOnly"
  title       = "SCIP Vertex GCS List Only"
  description = "Allows listing object names/metadata in the data lake bucket, without granting read access to object content. Needed because IAM conditions on resource.name are not evaluated for storage.objects.list (see vertex_gcs_lake)."
  permissions = ["storage.objects.list"]
}

resource "google_storage_bucket_iam_member" "vertex_gcs_list" {
  bucket = google_storage_bucket.data_lake.name
  role   = google_project_iam_custom_role.vertex_gcs_list_only.id
  member = "serviceAccount:${google_service_account.vertex_sa.email}"
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

# scip-vertex-sa must be able to act as itself when passed as the serving
# container SA in Vertex AI deploy calls (google.api_core INVALID_ARGUMENT
# otherwise). This is a self-referential actAs binding — the caller (SA key)
# and the specified service_account are the same identity.
resource "google_service_account_iam_member" "vertex_sa_self_act_as" {
  service_account_id = google_service_account.vertex_sa.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.vertex_sa.email}"
}

# Optional: grant a human deployer (e.g. the CI runner or dev user who runs
# deploy.py locally) permission to act as scip-vertex-sa. Set
# vertex_deployer_email in terraform.tfvars; leave empty to skip.
resource "google_service_account_iam_member" "vertex_sa_deployer" {
  count = var.vertex_deployer_email != "" ? 1 : 0

  service_account_id = google_service_account.vertex_sa.name
  role               = "roles/iam.serviceAccountUser"
  member             = "user:${var.vertex_deployer_email}"
}
