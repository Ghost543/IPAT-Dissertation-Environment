param([string]$ProjectRoot = (Get-Location).Path)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'pyproject.toml'))) {
    throw 'Run from the dissertation project root, or supply -ProjectRoot.'
}
$relativeFiles = @(
    'src\ipat\data\bpi17.py',
    'src\ipat\model\bpi17.py',
    'scripts\build_bpi17_adapter.py',
    'tests\test_bpi17_adapter.py',
    'configs\bpi17_contract_v0.1.json'
)
# Check every destination before copying anything. Existing identical files are safe to reuse.
foreach ($relative in $relativeFiles) {
    $source = Join-Path $PSScriptRoot $relative
    $target = Join-Path $ProjectRoot $relative
    if (Test-Path -LiteralPath $target) {
        if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash) {
            throw "Different file already exists: $target. Preserve it and review the difference before installation."
        }
    }
}
foreach ($relative in $relativeFiles) {
    $source = Join-Path $PSScriptRoot $relative
    $target = Join-Path $ProjectRoot $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    if (-not (Test-Path -LiteralPath $target)) { Copy-Item -LiteralPath $source -Destination $target }
}
Write-Output 'BPI17 adapter installed: five files. Existing baseline/adaptive/retention modules unchanged.'
