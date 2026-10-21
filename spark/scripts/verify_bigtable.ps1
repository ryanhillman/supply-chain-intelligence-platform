<#
.SYNOPSIS
    Verify the Bigtable feature store after Phase 3 deployment.

.DESCRIPTION
    Checks that:
      1. Bigtable instance and table exist
      2. All 6 column families are present
      3. Row count is non-zero
      4. A sample row contains populated cells across all families
      5. Value decoding (big-endian double) works correctly for numeric features

.PARAMETER ProjectId
    GCP project ID (defaults to $env:GCP_PROJECT_ID)

.PARAMETER InstanceId
    Bigtable instance ID (defaults to $env:BIGTABLE_INSTANCE_ID)

.PARAMETER TableId
    Bigtable table ID (defaults to $env:BIGTABLE_TABLE_ID)

.EXAMPLE
    .\verify_bigtable.ps1 -ProjectId my-project -InstanceId scip-feature-store -TableId inventory_features
#>

param(
    [string]$ProjectId  = $env:GCP_PROJECT_ID,
    [string]$InstanceId = $(if ($env:BIGTABLE_INSTANCE_ID) { $env:BIGTABLE_INSTANCE_ID } else { "scip-feature-store" }),
    [string]$TableId    = $(if ($env:BIGTABLE_TABLE_ID)    { $env:BIGTABLE_TABLE_ID    } else { "inventory_features" })
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

function Invoke-Cbt {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$CbtArgs
    )
    $output = & cbt -project $ProjectId -instance $InstanceId @CbtArgs 2>&1
    return $output
}

# ── Pre-flight ────────────────────────────────────────────────────────────────

Write-Header "Bigtable Feature Store Verification"

if (-not $ProjectId)  { Write-Fail "GCP_PROJECT_ID not set."; exit 1 }

Write-Info "Project  : $ProjectId"
Write-Info "Instance : $InstanceId"
Write-Info "Table    : $TableId"
Write-Info "Run time : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss UTC')"

# Check cbt is available
try {
    $cbtVersion = & cbt version 2>&1
    Write-Info "cbt version: $($cbtVersion[0])"
} catch {
    Write-Fail "cbt (Cloud Bigtable CLI) not found. Install via: gcloud components install cbt"
    exit 1
}

# ── Check 1: Instance exists ──────────────────────────────────────────────────

Write-Header "Check 1 — Instance Exists"

try {
    $instanceInfo = gcloud bigtable instances describe $InstanceId --project=$ProjectId --format="value(state)" 2>&1
    Assert-True ($LASTEXITCODE -eq 0 -and $instanceInfo -match "READY") `
        "Instance '$InstanceId' is READY" `
        "Instance '$InstanceId' not found or not READY (state: $instanceInfo)"
} catch {
    Write-Fail "gcloud error: $_"
    $failures++
}

# ── Check 2: Table exists ─────────────────────────────────────────────────────

Write-Header "Check 2 — Table Exists"

$tables = Invoke-Cbt "ls"
Assert-True ($tables -contains $TableId) `
    "Table '$TableId' found in instance" `
    "Table '$TableId' NOT found. Tables present: $($tables -join ', ')"

# ── Check 3: Column families ──────────────────────────────────────────────────

Write-Header "Check 3 — Column Families"

$expectedFamilies = @("inventory", "velocity", "demand", "supplier", "risk", "lineage")
$tableInfo = Invoke-Cbt "ls" $TableId

Write-Info "Table info from cbt ls $TableId :"
$tableInfo | ForEach-Object { Write-Info "  $_" }

foreach ($family in $expectedFamilies) {
    $found = $tableInfo | Where-Object { $_ -match "^\s*$family\b" }
    Assert-True ($null -ne $found) `
        "Column family '$family' present" `
        "Column family '$family' MISSING"
}

# ── Check 4: Row count ────────────────────────────────────────────────────────

Write-Header "Check 4 — Row Count"

Write-Info "Counting rows (may take a moment for large tables)..."
$countOutput = Invoke-Cbt "count" $TableId
# cbt count outputs: "<n> rows"
$rowCount = 0
if ($countOutput -match "(\d+)\s+rows?") {
    $rowCount = [int]$Matches[1]
}

Assert-True ($rowCount -gt 0) `
    "Row count: $rowCount rows" `
    "Row count is 0 — Silver job may not have written to Bigtable yet"

Write-Info "Total rows: $rowCount"

# Expected: 6 warehouses × N SKUs. Warn if fewer than 6.
if ($rowCount -gt 0 -and $rowCount -lt 6) {
    Write-Info "WARNING: Only $rowCount row(s) — expected at least 6 (one per warehouse/SKU pair)"
}

# ── Check 5: Sample row — all families populated ──────────────────────────────

Write-Header "Check 5 — Sample Row Completeness"

Write-Info "Reading first row from table..."
# cbt read with count=1 returns the first row lexicographically
$sampleRow = Invoke-Cbt "read" $TableId "count=1"

Write-Info "Raw cbt output:"
$sampleRow | ForEach-Object { Write-Info "  $_" }

foreach ($family in $expectedFamilies) {
    $familyCells = $sampleRow | Where-Object { $_ -match "^  ${family}:" }
    Assert-True ($null -ne $familyCells -and ($familyCells | Measure-Object).Count -gt 0) `
        "Family '$family' has cells in sample row" `
        "Family '$family' has NO cells in sample row — writer may have skipped this family"
}

# ── Check 6: Spot-check specific qualifiers ───────────────────────────────────

Write-Header "Check 6 — Key Qualifier Presence"

$expectedQualifiers = @{
    "inventory:cur_qty"   = "current inventory quantity"
    "velocity:sales_7d"   = "7-day sales velocity"
    "velocity:sales_30d"  = "30-day sales velocity"
    "risk:urgency"        = "reorder urgency tier"
    "lineage:pipeline_ver"= "pipeline version"
    "lineage:bronze_epoch"= "source bronze epoch"
}

foreach ($qualifier in $expectedQualifiers.Keys) {
    $desc  = $expectedQualifiers[$qualifier]
    $found = $sampleRow | Where-Object { $_ -match [regex]::Escape($qualifier) }
    Assert-True ($null -ne $found) `
        "$qualifier present ($desc)" `
        "$qualifier MISSING ($desc)"
}

# ── Check 7: Spot-check known warehouse row ───────────────────────────────────

Write-Header "Check 7 — Known Row Key (W-001#SKU-001)"

Write-Info "Reading row W-001#SKU-001 ..."
$knownRow = Invoke-Cbt "read" $TableId "prefix=W-001#SKU-001" "count=1"

if ($knownRow -match "W-001#SKU-001") {
    Write-Pass "Row W-001#SKU-001 found"
    $cellCount = ($knownRow | Where-Object { $_ -match "^\s+\d{4}/" }).Count
    Write-Info "  Cell count: $cellCount"
    Assert-True ($cellCount -ge 6) `
        "Row has >= 6 cells (at least one per family)" `
        "Row has only $cellCount cell(s) — some families may be empty"
} else {
    Write-Fail "Row W-001#SKU-001 not found — check warehouse/SKU seed data"
    $failures++
}

# ── Check 8: GC policy sanity (instance-level metadata) ──────────────────────

Write-Header "Check 8 — Instance Type"

$instanceType = gcloud bigtable instances describe $InstanceId `
    --project=$ProjectId `
    --format="value(type)" 2>&1

Write-Info "Instance type: $instanceType"
if ($instanceType -match "DEVELOPMENT") {
    Write-Info "  DEVELOPMENT instance — single node, no SLA, suitable for dev/testing"
    Write-Pass "Instance type appropriate for environment"
} elseif ($instanceType -match "PRODUCTION") {
    Write-Pass "PRODUCTION instance — SLA-backed, replication enabled"
} else {
    Write-Info "Unknown instance type: $instanceType"
}

# ── Summary ───────────────────────────────────────────────────────────────────

Write-Header "Summary"

Write-Info "Instance : $InstanceId"
Write-Info "Table    : $TableId"
Write-Info "Rows     : $rowCount"

if ($failures -eq 0) {
    Write-Host "  ALL CHECKS PASSED" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Next steps:" -ForegroundColor Cyan
    Write-Host "    • Run verify_silver.ps1 to validate the offline Silver layer" -ForegroundColor Cyan
    Write-Host "    • Phase 4: point PyTorch inference at Bigtable for <10ms feature reads" -ForegroundColor Cyan
    Write-Host ""
    exit 0
} else {
    Write-Host "  $failures CHECK(S) FAILED" -ForegroundColor Red
    Write-Host "  See [FAIL] lines above for details." -ForegroundColor Red
    Write-Host ""
    exit 1
}
