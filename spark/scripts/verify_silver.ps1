<#
.SYNOPSIS
    Verify the Silver GCS layer after Phase 3 deployment.

.DESCRIPTION
    Checks that:
      1. GCS bucket and silver/ prefix are reachable
      2. Silver Parquet files exist and are partitioned correctly
      3. A PySpark validation job reads Silver and reports row counts,
         schema conformance, and null rates on key features

.PARAMETER ProjectId
    GCP project ID (defaults to $env:GCP_PROJECT_ID)

.PARAMETER BucketName
    GCS bucket name (defaults to $env:GCS_BUCKET_NAME)

.EXAMPLE
    .\verify_silver.ps1 -ProjectId my-project -BucketName my-project-scip-lake
#>

param(
    [string]$ProjectId  = $env:GCP_PROJECT_ID,
    [string]$BucketName = $env:GCS_BUCKET_NAME
)

$ErrorActionPreference = "Stop"

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

# ── Pre-flight ────────────────────────────────────────────────────────────────

Write-Header "Silver GCS Verification"

if (-not $ProjectId)  { Write-Fail "GCP_PROJECT_ID not set. Pass -ProjectId or set env var."; exit 1 }
if (-not $BucketName) { Write-Fail "GCS_BUCKET_NAME not set. Pass -BucketName or set env var."; exit 1 }

Write-Info "Project  : $ProjectId"
Write-Info "Bucket   : $BucketName"
Write-Info "Silver   : gs://$BucketName/silver"
Write-Info "Run time : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss UTC')"

# ── Check 1: Bucket reachable ─────────────────────────────────────────────────

Write-Header "Check 1 — GCS Bucket Reachable"

try {
    $bucketInfo = gsutil ls -b "gs://$BucketName" 2>&1
    Assert-True ($LASTEXITCODE -eq 0) `
        "Bucket gs://$BucketName is reachable" `
        "Cannot reach gs://$BucketName — check credentials and bucket name"
} catch {
    Write-Fail "gsutil error: $_"
    $failures++
}

# ── Check 2: Silver prefix exists ─────────────────────────────────────────────

Write-Header "Check 2 — Silver Prefix Exists"

$silverPath = "gs://$BucketName/silver"
$silverLs = gsutil ls "$silverPath/" 2>&1

Assert-True ($LASTEXITCODE -eq 0) `
    "silver/ prefix exists at $silverPath" `
    "silver/ prefix not found — has the Silver job run yet?"

if ($LASTEXITCODE -eq 0) {
    Write-Info "Top-level silver/ contents:"
    $silverLs | ForEach-Object { Write-Info "  $_" }
}

# ── Check 3: Warehouse partitions ─────────────────────────────────────────────

Write-Header "Check 3 — Warehouse Partition Structure"

$expectedWarehouses = @("W-001", "W-002", "W-003", "W-004", "W-005", "W-006")
$partitions = gsutil ls "$silverPath/" 2>&1 | Where-Object { $_ -match "warehouse_id=" }

Write-Info "Found $(($partitions | Measure-Object).Count) warehouse_id= partition(s)"

foreach ($wh in $expectedWarehouses) {
    $found = $partitions | Where-Object { $_ -match "warehouse_id=$wh" }
    Assert-True ($null -ne $found) `
        "Partition warehouse_id=$wh present" `
        "Partition warehouse_id=$wh MISSING"
}

# ── Check 4: Parquet files present ────────────────────────────────────────────

Write-Header "Check 4 — Parquet Files"

$parquetFiles = gsutil ls -r "$silverPath/**.parquet" 2>&1 | Where-Object { $_ -match "\.parquet$" }
$parquetCount = ($parquetFiles | Measure-Object).Count

Assert-True ($parquetCount -gt 0) `
    "$parquetCount Parquet file(s) found under silver/" `
    "No Parquet files found — Silver job may not have written any data"

if ($parquetCount -gt 0) {
    Write-Info "Sample files (first 5):"
    $parquetFiles | Select-Object -First 5 | ForEach-Object { Write-Info "  $_" }
}

# ── Check 5: Total GCS size ───────────────────────────────────────────────────

Write-Header "Check 5 — Silver Layer Size"

$duOutput = gsutil du -sh "$silverPath/" 2>&1
Write-Info "Silver layer size: $duOutput"
Assert-True ($LASTEXITCODE -eq 0) "gsutil du succeeded" "gsutil du failed"

# ── Check 6: PySpark schema + row count validation ────────────────────────────

Write-Header "Check 6 — PySpark Schema + Feature Completeness"

Write-Info "Running PySpark validation via Docker (scripts/verify_silver_runner.py)..."
Write-Info "(requires scip-spark image and SA key)"

$keysPath = Resolve-Path (Join-Path $PSScriptRoot "..\..\infrastructure\gcp\keys\scip-spark-sa.json") -ErrorAction SilentlyContinue

if (-not $keysPath) {
    Write-Info "SA key not found at infrastructure/gcp/keys/scip-spark-sa.json"
    Write-Info "Skipping PySpark validation — run manually after creating SA key"
} else {
    docker run --rm `
        -v "${PSScriptRoot}\..\:/opt/spark-app" `
        -v "${keysPath}:/opt/spark-app/keys/scip-spark-sa.json:ro" `
        -e "GOOGLE_APPLICATION_CREDENTIALS=/opt/spark-app/keys/scip-spark-sa.json" `
        --network kafka_scip-net `
        scip-spark:latest `
        --master "local[*]" `
        --conf "spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem" `
        --conf "spark.hadoop.google.cloud.auth.service.account.enable=true" `
        --conf "spark.hadoop.google.cloud.auth.service.account.json.keyfile=/opt/spark-app/keys/scip-spark-sa.json" `
        "/opt/spark-app/scripts/verify_silver_runner.py" `
        "$silverPath"

    Assert-True ($LASTEXITCODE -eq 0) `
        "PySpark Silver validation passed" `
        "PySpark Silver validation failed (exit code $LASTEXITCODE)"
}

# ── Summary ───────────────────────────────────────────────────────────────────

Write-Header "Summary"

if ($failures -eq 0) {
    Write-Host "  ALL CHECKS PASSED" -ForegroundColor Green
    Write-Host ""
    exit 0
} else {
    Write-Host "  $failures CHECK(S) FAILED" -ForegroundColor Red
    Write-Host ""
    exit 1
}
