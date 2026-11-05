#!/usr/bin/env bash
# Pipeline validation, batch mode: checks the pipeline's outputs on GCP.
#
#   1. GCS Bronze        the five topic prefixes exist and hold Parquet files
#   2. Silver batch job  jobs/silver_batch.py reads Bronze, writes Silver and Bigtable
#   3. GCS Silver        each of the six warehouse partitions holds Parquet files
#   4. Silver schema     scripts/verify_silver_runner.py checks columns and nulls
#   5. Bigtable          row count, and the column families of one row
#
# Needs gsutil and cbt (gcloud SDK) on PATH, the scip-spark:latest image, and the
# Spark service-account key at infrastructure/gcp/keys/scip-spark-sa.json.
#
# Usage:
#   spark/scripts/validate_pipeline.sh [--project ID] [--bucket NAME]
#                                      [--instance ID] [--table ID] [--skip-batch-run]
#
# --project and --bucket default to GCP_PROJECT_ID and GCS_BUCKET_NAME, the keys in
# spark/.env. --skip-batch-run validates the Silver and Bigtable data already there.
set -uo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash on Windows: pass container paths through unchanged

project="${GCP_PROJECT_ID:-}"
bucket="${GCS_BUCKET_NAME:-}"
instance="scip-feature-store"
table="inventory_features"
skip_batch_run=false

while [ $# -gt 0 ]; do
  case "$1" in
    --project)        project="${2:?--project needs a value}";   shift 2 ;;
    --bucket)         bucket="${2:?--bucket needs a value}";     shift 2 ;;
    --instance)       instance="${2:?--instance needs a value}"; shift 2 ;;
    --table)          table="${2:?--table needs a value}";       shift 2 ;;
    --skip-batch-run) skip_batch_run=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$project" ] || { echo "Pass --project or set GCP_PROJECT_ID" >&2; exit 2; }
[ -n "$bucket" ]  || { echo "Pass --bucket or set GCS_BUCKET_NAME" >&2; exit 2; }

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
spark_dir="$repo_dir/spark"
key_path="$repo_dir/infrastructure/gcp/keys/scip-spark-sa.json"
bronze="gs://$bucket/bronze"
silver="gs://$bucket/silver"
failures=0

header() { printf '\n== %s ==\n' "$1"; }
info()   { echo "  [INFO] $1"; }

# check <exit status> <pass message> <fail message>
check() {
  if [ "$1" -eq 0 ]; then
    echo "  [PASS] $2"
  else
    echo "  [FAIL] $3"
    failures=$((failures + 1))
  fi
}

count_parquet() { gsutil ls -r "$1" 2>/dev/null | grep -c '\.parquet$'; }

# run_spark <container memory> <driver memory> <job> [job arguments]
# spark-submit in the scip-spark image, with spark/ and the key mounted where
# docker-compose.spark.yml mounts them.
run_spark() {
  local memory=$1 driver_memory=$2
  shift 2
  docker run --rm --memory "$memory" \
    -e GCP_PROJECT_ID="$project" \
    -e BIGTABLE_INSTANCE_ID="$instance" \
    -e BIGTABLE_TABLE_ID="$table" \
    -e BRONZE_BASE_PATH="$bronze" \
    -e SILVER_BASE_PATH="$silver" \
    -e GOOGLE_APPLICATION_CREDENTIALS=/opt/spark-app/keys/scip-spark-sa.json \
    -v "$spark_dir:/opt/spark-app" \
    -v "$key_path:/opt/spark-app/keys/scip-spark-sa.json:ro" \
    scip-spark:latest \
    --master "local[2]" \
    --properties-file config/spark-gcs.conf \
    --conf "spark.driver.memory=$driver_memory" \
    --conf spark.sql.shuffle.partitions=6 \
    "$@"
}

header "Pipeline validation"
info "Project  : $project"
info "Bronze   : $bronze"
info "Silver   : $silver"
info "Bigtable : $instance / $table"

if [ ! -f "$key_path" ]; then
  echo "  [FAIL] Service-account key not found at $key_path"
  exit 1
fi
if ! docker image inspect scip-spark:latest >/dev/null 2>&1; then
  echo "  [FAIL] scip-spark:latest not found; run: docker build -t scip-spark:latest spark/"
  exit 1
fi

header "Check 1: GCS Bronze"
for topic in demand.signals inventory.events purchase.orders.events shipments.events suppliers.events; do
  gsutil ls "$bronze/$topic/" >/dev/null 2>&1
  check $? "Bronze topic $topic/ exists" "Bronze topic $topic/ not found"
done
n=$(count_parquet "$bronze/**.parquet")
[ "$n" -gt 0 ]
check $? "$n Bronze Parquet files" "No Bronze Parquet files"

header "Check 2: Silver batch job"
if $skip_batch_run; then
  info "Skipped (--skip-batch-run)"
else
  info "Running jobs/silver_batch.py; this takes a few minutes ..."
  run_spark 768m 512m jobs/silver_batch.py
  check $? "silver_batch.py completed" "silver_batch.py failed"
fi

header "Check 3: GCS Silver"
for warehouse in W-001 W-002 W-003 W-004 W-005 W-006; do
  n=$(count_parquet "$silver/warehouse_id=$warehouse/**")
  [ "$n" -gt 0 ]
  check $? "warehouse_id=$warehouse: $n Parquet files" "warehouse_id=$warehouse: no Parquet files"
done

header "Check 4: Silver schema and nulls"
run_spark 512m 384m scripts/verify_silver_runner.py "$silver"
check $? "Silver schema validation passed" "Silver schema validation failed"

header "Check 5: Bigtable"
rows=$(cbt -project "$project" -instance "$instance" count "$table" 2>/dev/null | grep -E '^[0-9]+$' | tail -1)
if [ -n "$rows" ]; then
  [ "$rows" -ge 120 ]
  check $? "Bigtable rows: $rows (6 warehouses x 20 SKUs = 120)" "Bigtable rows: $rows, expected at least 120"
else
  check 1 "" "Could not read the Bigtable row count from cbt"
fi
sample=$(cbt -project "$project" -instance "$instance" lookup "$table" "W-001#SKU-001" 2>/dev/null)
for family in inventory velocity demand supplier risk lineage; do
  grep -q "$family:" <<<"$sample"
  check $? "Column family $family present in W-001#SKU-001" "Column family $family missing from W-001#SKU-001"
done

header "Summary"
if [ "$failures" -eq 0 ]; then
  echo "  ALL CHECKS PASSED"
  exit 0
fi
echo "  $failures CHECK(S) FAILED"
exit 1
