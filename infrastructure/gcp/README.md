# GCP Infrastructure

Terraform for the data-lake bucket, Bigtable, Artifact Registry, the Vertex AI endpoint, Pub/Sub and their service accounts.

After `terraform apply`, from this directory:

1. Create the service account keys. Terraform does not manage them, and `keys/` is gitignored.

   ```bash
   for sa in spark vertex prediction_writer; do
     gcloud iam service-accounts keys create "keys/scip-${sa//_/-}-sa.json" \
       --iam-account="$(terraform output -raw "${sa}_sa_email")"
   done
   ```

2. Copy `spark/.env.example` to `spark/.env` and fill in the project ID and bucket name.

3. Build and push the serving image.

   ```bash
   REPO="$(terraform output -raw artifact_registry_url)"
   gcloud auth configure-docker "${REPO%%/*}"
   docker build -t "$REPO/shortage-predictor-serving:latest" ../../model_serving/serving/container
   docker push "$REPO/shortage-predictor-serving:latest"
   ```

4. Register and deploy the model with `model_serving/scripts/register.py` and `deploy.py`; each script's docstring lists its commands.
