###############################################################################
# Pub/Sub Infrastructure — Phase 5
# prediction-results topic, prediction-writer subscription, dead-letter topic,
# prediction-writer service account + IAM.
#
# Decouples batch scoring from Bigtable writes: batch_scorer.py publishes;
# prediction_writer.py subscribes.
###############################################################################

###############################################################################
# API Enablement
###############################################################################

resource "google_project_service" "pubsub" {
  project            = var.project_id
  service            = "pubsub.googleapis.com"
  disable_on_destroy = false
}

###############################################################################
# Data source — project number (needed for Pub/Sub service agent IAM)
###############################################################################

data "google_project" "project" {
  project_id = var.project_id
}

###############################################################################
# Topics
###############################################################################

# Primary delivery topic. batch_scorer.py publishes one message per entity.
resource "google_pubsub_topic" "prediction_results" {
  project = var.project_id
  name    = "prediction-results"

  # 24h retention: one full scoring cycle has time to complete before messages expire.
  message_retention_duration = "86400s"

  labels = local.common_labels

  depends_on = [google_project_service.pubsub]
}

# Dead-letter topic. Messages that fail 5 delivery attempts are forwarded here
# for forensic inspection without blocking the primary subscription.
resource "google_pubsub_topic" "prediction_results_dlq" {
  project = var.project_id
  name    = "prediction-results-dead-letter"

  # 7d retention on the DLQ to allow manual investigation.
  message_retention_duration = "604800s"

  labels = local.common_labels

  depends_on = [google_project_service.pubsub]
}

###############################################################################
# Subscription
###############################################################################

resource "google_pubsub_subscription" "prediction_writer" {
  project = var.project_id
  name    = "prediction-writer"
  topic   = google_pubsub_topic.prediction_results.id

  # 60s ack deadline: prediction_writer.py writes a batch of 100 rows per pull.
  # A single MutateRows call to Bigtable completes in < 5s; 60s is generous.
  ack_deadline_seconds = 60

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.prediction_results_dlq.id
    max_delivery_attempts = 5
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  labels = local.common_labels
}

###############################################################################
# IAM — Pub/Sub service agent must publish to the DLQ topic and subscribe
# to the prediction-writer subscription to forward dead-letter messages.
# The service agent email is deterministic: service-{PROJECT_NUMBER}@gcp-sa-pubsub
###############################################################################

resource "google_pubsub_topic_iam_member" "pubsub_agent_dlq_publisher" {
  project = var.project_id
  topic   = google_pubsub_topic.prediction_results_dlq.name
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

resource "google_pubsub_subscription_iam_member" "pubsub_agent_dlq_subscriber" {
  project      = var.project_id
  subscription = google_pubsub_subscription.prediction_writer.name
  role         = "roles/pubsub.subscriber"
  member       = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-pubsub.iam.gserviceaccount.com"
}

###############################################################################
# Service Account — scip-prediction-writer-sa
#
# Used by prediction_writer.py (Cloud Run job) exclusively.
# Subscribes to prediction-results and writes the predictions column family
# to Bigtable. Separate from scip-vertex-sa to enforce least privilege:
# the writer does not need aiplatform.user or storage access.
###############################################################################

resource "google_service_account" "prediction_writer_sa" {
  project      = var.project_id
  account_id   = "scip-prediction-writer-sa"
  display_name = "SCIP Prediction Writer Service Account"
  description  = "Subscribes to prediction-results and writes Bigtable predictions family (Phase 5)"

  depends_on = [google_project_service.iam]
}

resource "google_project_iam_member" "prediction_writer_pubsub_subscriber" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.prediction_writer_sa.email}"
}

resource "google_project_iam_member" "prediction_writer_bigtable" {
  project = var.project_id
  role    = "roles/bigtable.user"
  member  = "serviceAccount:${google_service_account.prediction_writer_sa.email}"
}

resource "google_project_iam_member" "prediction_writer_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.prediction_writer_sa.email}"
}
