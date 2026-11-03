# IPAT BPI17 contract and preflight v0.1

Read `DATASET_CONTRACT.md` first. `dataset_contract.json` is the machine-readable specification. Status: specified for adapter implementation; no processed-data or model freeze. The ZIP contains no source dataset and does not overwrite existing project modules.

The preflight reads gzip directly, checks source identity, assigns complete applications to chronological partitions or exclusions, and writes an offer manifest, JSON and optional PNG/SVG/PDF/offline HTML. Plots are produced from actual preflight output. Bundled `evidence/` is the verified development run; user-local replication goes to a separate directory.

PowerShell, from the existing dissertation project root, after downloading the ZIP to Downloads:

```powershell
$ipatArchive = Join-Path $env:USERPROFILE 'Downloads\IPAT_BPI17_Contract_v0.1.zip'
$ipatKit = Join-Path (Get-Location) 'IPAT_BPI17_Contract_v0.1'
if (Test-Path -LiteralPath $ipatKit) { throw 'Destination already exists. Choose a fresh directory to preserve previous evidence.' }
Expand-Archive -LiteralPath $ipatArchive -DestinationPath $ipatKit

uv run python -m unittest discover -s "$ipatKit\tests" -v

# Change only this path if the original archive is stored elsewhere.
$ipatSource = Join-Path $env:USERPROFILE 'Downloads\BPI Challenge 2017.xes.gz'
if (-not (Test-Path -LiteralPath $ipatSource)) { throw "Source archive not found: $ipatSource" }
$ipatOutput = Join-Path (Get-Location) 'results\bpi17_contract_v0.1_local'
if (Test-Path -LiteralPath $ipatOutput) { throw 'Output already exists. Use a fresh directory to preserve the previous run.' }
uv run --with matplotlib==3.10.8 python "$ipatKit\scripts\preflight_bpi17_contract.py" `
  --source "$ipatSource" `
  --contract "$ipatKit\dataset_contract.json" `
  --output-dir "$ipatOutput" `
  --plots
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed. Inspect the error before continuing.' }
Start-Process (Join-Path $ipatOutput 'contract_preflight.html')
```

`--with` supplies plotting support for this command without editing project dependency files. The parser and tests require only Python 3.11+ standard library. Use `uv run python ...` without `--plots` if figures are not needed. Keep using the project's intended environment; a uv VIRTUAL_ENV mismatch warning alone is not a failed preflight.

Expected measured split counts: train 19,071; validation 7,871; calibration 6,117; test 8,613; excluded 1,323. All 42,995 candidates remain represented in the CSV. The excluded cohort is longer and has a higher false-label proportion; it is a documented selection limitation, not missing accounting. Both classes remain in every admitted partition.

Send back `contract_preflight.json` and the test output for local evidence reconciliation. The manifest hash should match the bundled run for the same archive and contract. Plot file hashes need not match across rendering environments.

Files: contract Markdown/JSON; streaming preflight script; ten fixture tests; candidate/split CSV; measured JSON; integrated plots and HTML; validation receipt; provenance; package manifest. `MANIFEST.json` hashes all other delivered files and excludes itself. Do not treat archived JSON status=PASS as proof that your local command ran.
