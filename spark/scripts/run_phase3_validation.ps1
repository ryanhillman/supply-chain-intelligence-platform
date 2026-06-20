<#
.SYNOPSIS
    Phase 3 validation — batch mode.

.DESCRIPTION
    Proves all Phase 3 outputs without requiring continuous streaming:

      1. GCS Bronze — verify all 5 topic prefixes and total file count
      2. GCS Silver — verify all 6 warehouse partitions and Parquet files
      3. Silver batch job — read Bronze GCS, write Silver + Bigtable (one shot)
      4. Silver schema validation — PySpark reads Silver and checks schema/nulls
      5. Bigtable row count — cbt count on inventory_features table

    Designed for Docker Desktop environments where running Bronze + Silver
    streaming jobs simultaneously exceeds available memory (>3 GB combined).
    The production architecture (silver_feature_engineering.py) is unchanged;
    this script uses silver_batch.py as a low-memory one-shot alternative.

    Prerequisites:
      - docker ps shows kafka is healthy
      - gsutil / cbt are on PATH (gcloud SDK)
      - scip-spark:latest image exists (docker images | grep scip-spark)
      - SA key exists at ../infrastructure/gcp/keys/scip-spark-sa.json

.PARAMETER ProjectId
    GCP project ID. Required: pass -ProjectId or set the GCP_PROJECT_ID
    environment variable (matches the key in spark/.env).

.PARAMETER BucketName
    GCS bucket name. Required: pass -BucketName or set the GCS_BUCKET_NAME
    environment variable (matches the key in spark/.env).

.PARAMETER SkipBatchRun
    Skip the silver_batch.py step. Use when Silver data already exists
    in GCS and you only want to validate existing outputs.

.EXAMPLE
    # Pass values explicitly:
    .\run_phase3_validation.ps1 -ProjectId my-gcp-project -BucketName my-gcp-project-scip-lake

.EXAMPLE
    # Or export env vars first (e.g. after sourcing spark/.env), then run:
    .\run_phase3_validation.ps1

.EXAMPLE
    .\run_phase3_validation.ps1 -ProjectId my-gcp-project -BucketName my-gcp-project-scip-lake -SkipBatchRun
#>

param(
    [string]$ProjectId   = "",
    [string]$BucketName  = "",
    [string]$Instance    = "scip-feature-store",
    [string]$Table       = "inventory_features",
    [switch]$SkipBatchRun
)

$ErrorActionPreference = "Stop"

# Resolve from environment when not passed explicitly.
# GCP_PROJECT_ID and GCS_BUCKET_NAME match the keys in spark/.env.
if (-not $ProjectId)  { $ProjectId  = $env:GCP_PROJECT_ID }
if (-not $BucketName) { $BucketName = $env:GCS_BUCKET_NAME }

if (-not $ProjectId) {
    throw "ProjectId is required. Pass -ProjectId <id> or set the GCP_PROJECT_ID environment variable."
}
if (-not $BucketName) {
    throw "BucketName is required. Pass -BucketName <name> or set the GCS_BUCKET_NAME environment variable."
}

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-Header([string]$text) {
    Write-Host ""
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
}

function Write-Pass([string]$msg) { Write-Host "  [PASS] $msg" -ForegroundColor Green  }
function Write-Fail([string]$msg) { Write-Host "  [FAIL] $msg" -ForegroundColor Red    }
function Write-Info([string]$msg) { Write-Host "  [INFO] $msg" -ForegroundColor Yellow }

$failures = 0

function Assert-True([bool]$condition, [string]$pass, [string]$fail) {
    if ($condition) { Write-Pass $pass }
    else            { Write-Fail $fail; $script:failures++ }
}

# ── Resolve paths ─────────────────────────────────────────────────────────────

$ScriptDir  = $PSScriptRoot                              # spark/scripts/
$SparkDir   = Resolve-Path (Join-Path $ScriptDir "..")   # spark/
$KeyPath    = Resolve-Path (Join-Path $ScriptDir "..\..\infrastructure\gcp\keys\scip-spark-sa.json")

$BronzePath = "gs://$BucketName/bronze"
$SilverPath = "gs://$BucketName/silver"

# ── Pre-flight ────────────────────────────────────────────────────────────────

Write-Header "Phase 3 Validation — Batch Mode"
Write-Info "Project  : $ProjectId"
Write-Info "Bucket   : $BucketName"
Write-Info "Bronze   : $BronzePath"
Write-Info "Silver   : $SilverPath"
Write-Info "Bigtable : $Instance / $Table"
Write-Info "SA key   : $KeyPath"
Write-Info "Run time : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

if (-not (Test-Path $KeyPath)) {
    Write-Fail "SA key not found at $KeyPath"
    exit 1
}

$sparkImage = docker images scip-spark:latest --format "{{.Repository}}:{{.Tag}}" 2>&1
if ($sparkImage -notmatch "scip-spark:latest") {
    Write-Fail "scip-spark:latest image not found — run: docker build -t scip-spark:latest spark/"
    exit 1
}

# ── Check 1: GCS Bronze ───────────────────────────────────────────────────────

Write-Header "Check 1 — GCS Bronze Layer"

$expectedTopics = @(
    "demand.signals",
    "inventory.events",
    "purchase.orders.events",
    "shipments.events",
    "suppliers.events"
)

foreach ($topic in $expectedTopics) {
    $ls = gsutil ls "$BronzePath/$topic/" 2>&1
    Assert-True ($LASTEXITCODE -eq 0) `
        "Bronze topic $topic/ exists" `
        "Bronze topic $topic/ NOT FOUND"
}

$bronzeSize = (gsutil du -sh "$BronzePath/**" 2>&1 | Select-String "MiB|KiB|GiB").Line
Write-Info "Bronze total size: $bronzeSize"

$parquetCount = (gsutil ls -r "$BronzePath/**.parquet" 2>&1 | Where-Object { $_ -match "\.parquet$" } | Measure-Object).Count
Assert-True ($parquetCount -gt 0) `
    "$parquetCount Bronze Parquet files found" `
    "No Bronze Parquet files found"

# ── Check 2: GCS Silver (pre-existing) ───────────────────────────────────────

Write-Header "Check 2 — GCS Silver Layer (pre-existing)"

$expectedWarehouses = @("W-001","W-002","W-003","W-004","W-005","W-006")

foreach ($wh in $expectedWarehouses) {
    $whParquet = (gsutil ls -r "$SilverPath/warehouse_id=$wh/**" 2>&1 |
        Where-Object { $_ -match "\.parquet$" } | Measure-Object).Count
    Assert-True ($whParquet -gt 0) `
        "warehouse_id=$wh : $whParquet Parquet files" `
        "warehouse_id=$wh : no Parquet files"
}

$silverSize = (gsutil du -sh "$SilverPath/**" 2>&1 | Select-String "MiB|KiB|GiB").Line
Write-Info "Silver total size: $silverSize"

# ── Check 3: Silver batch job ─────────────────────────────────────────────────

Write-Header "Check 3 — Silver Batch Job (read Bronze → write Silver + Bigtable)"

if ($SkipBatchRun) {
    Write-Info "Skipping batch run (-SkipBatchRun flag set)"
} else {
    Write-Info "Running silver_batch.py as one-shot container (memory: 768m) ..."
    Write-Info "This reads existing Bronze GCS Parquet and writes one Silver epoch."
    Write-Info "Expected runtime: 3–8 minutes."

    docker run --rm `
        --name scip-silver-batch `
        --memory 768m `
        -e GCP_PROJECT_ID=$ProjectId `
        -e BIGTABLE_INSTANCE_ID=$Instance `
        -e BIGTABLE_TABLE_ID=$Table `
        -e BRONZE_BASE_PATH="gs://$BucketName/bronze" `
        -e GOOGLE_APPLICATION_CREDENTIALS=/opt/spark-app/keys/scip-spark-sa.json `
        -v "${SparkDir}:/opt/spark-app" `
        -v "${KeyPath}:/opt/spark-app/keys/scip-spark-sa.json:ro" `
        scip-spark:latest `
        --master "local[2]" `
        --conf "spark.driver.memory=512m" `
        --conf "spark.sql.shuffle.partitions=6" `
        --conf "spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem" `
        --conf "spark.hadoop.fs.AbstractFileSystem.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS" `
        --conf "spark.hadoop.google.cloud.auth.service.account.enable=true" `
        --conf "spark.hadoop.google.cloud.auth.service.account.json.keyfile=/opt/spark-app/keys/scip-spark-sa.json" `
        --conf "spark.hadoop.fs.gs.outputstream.type=FLUSHABLE_COMPOSITE" `
        --conf "spark.hadoop.fs.gs.implicit.dir.infer.enable=true" `
        "jobs/silver_batch.py"

    Assert-True ($LASTEXITCODE -eq 0) `
        "silver_batch.py completed successfully" `
        "silver_batch.py FAILED (exit code $LASTEXITCODE)"
}

# ── Check 4: Silver schema + row count (PySpark) ──────────────────────────────

Write-Header "Check 4 — Silver Schema + Feature Completeness (PySpark)"

Write-Info "Running verify_silver_runner.py against $SilverPath ..."

docker run --rm `
    --name scip-silver-verify `
    --memory 512m `
    -e GOOGLE_APPLICATION_CREDENTIALS=/opt/spark-app/keys/scip-spark-sa.json `
    -v "${SparkDir}:/opt/spark-app" `
    -v "${KeyPath}:/opt/spark-app/keys/scip-spark-sa.json:ro" `
    scip-spark:latest `
    --master "local[2]" `
    --conf "spark.driver.memory=384m" `
    --conf "spark.sql.shuffle.partitions=6" `
    --conf "spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem" `
    --conf "spark.hadoop.fs.AbstractFileSystem.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS" `
    --conf "spark.hadoop.google.cloud.auth.service.account.enable=true" `
    --conf "spark.hadoop.google.cloud.auth.service.account.json.keyfile=/opt/spark-app/keys/scip-spark-sa.json" `
    "scripts/verify_silver_runner.py" `
    $SilverPath

Assert-True ($LASTEXITCODE -eq 0) `
    "Silver schema validation passed" `
    "Silver schema validation FAILED (exit code $LASTEXITCODE)"

# ── Check 5: Bigtable row count ───────────────────────────────────────────────

Write-Header "Check 5 — Bigtable Row Count"

Write-Info "Counting rows in $Instance / $Table ..."

$rowCount = (cbt -project $ProjectId -instance $Instance count $Table 2>&1 |
    Select-String "^\d+$").Line

if ($rowCount) {
    $count = [int]$rowCount
    Assert-True ($count -ge 120) `
        "Bigtable rows: $count (expected >= 120 for 6 warehouses x 20 SKUs)" `
        "Bigtable rows: $count — expected >= 120"
} else {
    Write-Fail "Could not parse Bigtable row count from cbt output"
    $script:failures++
}

# ── Sample row spot-check ─────────────────────────────────────────────────────

Write-Header "Check 5b — Bigtable Sample Row"

$sampleRow = cbt -project $ProjectId -instance $Instance lookup $Table "W-001#SKU-001" 2>&1
$families = @("inventory","velocity","demand","supplier","risk","lineage")

foreach ($fam in $families) {
    Assert-True ($sampleRow -match "${fam}:") `
        "Column family '$fam' present in W-001#SKU-001" `
        "Column family '$fam' MISSING from W-001#SKU-001"
}

# ── Summary ───────────────────────────────────────────────────────────────────

Write-Header "Phase 3 Validation Summary"

Write-Info "Bronze   : $BronzePath"
Write-Info "Silver   : $SilverPath"
Write-Info "Bigtable : $Instance/$Table — $rowCount rows"

if ($failures -eq 0) {
    Write-Host ""
    Write-Host "  ALL PHASE 3 CHECKS PASSED" -ForegroundColor Green
    Write-Host "  Phase 3 is complete. Ready for Phase 4 (PyTorch inference)." -ForegroundColor Green
    Write-Host ""
    exit 0
} else {
    Write-Host ""
    Write-Host "  $failures CHECK(S) FAILED" -ForegroundColor Red
    Write-Host ""
    exit 1
}
