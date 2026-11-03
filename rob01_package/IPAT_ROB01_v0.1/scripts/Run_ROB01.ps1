param(
    [Parameter(Mandatory=$true)][string]$CleanPath,
    [string]$ParentPath,
    [string]$CalibrationPath,
    [string]$CachePath,
    [string]$ResultsRoot = "results"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $ProjectRoot
try {
    $Clean = (Resolve-Path -LiteralPath $CleanPath).Path
    $Manifest = Get-Content -LiteralPath (Join-Path $Clean "exp06_manifest.json") -Raw | ConvertFrom-Json
    if (-not $ParentPath) { $ParentPath = $Manifest.parent }
    if (-not $CalibrationPath) { $CalibrationPath = $Manifest.calibration }
    if (-not $CachePath) { $CachePath = $Manifest.cache }
    foreach ($p in @($ParentPath, $CalibrationPath, $CachePath)) {
        if (-not (Test-Path -LiteralPath $p -PathType Container)) {
            throw "Input directory unavailable: $p. Supply its current path using the matching parameter."
        }
    }
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $Verify = Join-Path $ResultsRoot "rob01_runner_$Stamp"
    $Run = Join-Path $ResultsRoot "rob01_$Stamp"
    uv run python scripts/verify_rob01.py --output $Verify
    if ($LASTEXITCODE -ne 0) { throw "ROB-01 verification failed. Retain: $Verify" }
    $Receipt = Join-Path $Verify "rob01_runner_acceptance.json"
    $Check = Get-Content -LiteralPath $Receipt -Raw | ConvertFrom-Json
    if ($Check.status -ne "PASS") { throw "ROB-01 verifier did not pass." }
    uv run python scripts/run_rob01.py --parent $ParentPath --calibration $CalibrationPath --cache $CachePath --clean $Clean --verification $Receipt --output $Run
    if ($LASTEXITCODE -ne 0) { throw "ROB-01 failed. Retain output: $Run" }
    $Acceptance = Get-Content -LiteralPath (Join-Path $Run "rob01_acceptance.json") -Raw | ConvertFrom-Json
    if ($Acceptance.status -ne "PASS") { throw "ROB-01 acceptance did not pass." }
    Write-Host "ROB-01 verification evidence: $Verify"
    Write-Host "ROB-01 evaluation evidence: $Run"
    Start-Process (Join-Path $Run "plots\report.html")
} finally {
    Pop-Location
}
