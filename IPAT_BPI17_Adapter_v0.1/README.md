# IPAT BPI17 schema adapter v0.1

Read `ADAPTER_DESIGN.md` before running. This package adds a six-numerical/seven-categorical adapter and schema-specific model assemblers; it reuses existing frozen model components and forward methods. Installation copies five new files and refuses to overwrite a different existing file. No baseline, adaptive or retention source file is replaced.

Development validation: 20 adapter tests, 17 full-source/integration checks and 37 existing model regression tests passed. Local replication remains required. No EXP-05 predictive experiment was run. Full processed datasets are generated locally; they are not included in this package.

## PowerShell from the project root

The ZIP is expected in the project root, as requested:

```powershell
$ipatKit = Join-Path (Get-Location) 'IPAT_BPI17_Adapter_v0.1'
if (Test-Path -LiteralPath $ipatKit) { throw 'Choose a fresh extraction directory to preserve earlier files.' }
Expand-Archive -LiteralPath '.\IPAT_BPI17_Adapter_v0.1.zip' -DestinationPath $ipatKit
& "$ipatKit\install_adapter.ps1" -ProjectRoot (Get-Location).Path

uv run python -m unittest discover -s '.\tests' -p 'test_bpi17_adapter.py' -v 2>&1 |
    Tee-Object -FilePath '.\bpi17_adapter_local_tests.txt'
if ($LASTEXITCODE -ne 0) { throw 'Adapter tests failed. Inspect the log before continuing.' }

# Adjust only the source path if your original archive is stored elsewhere.
$ipatSource = '.\BPI Challenge 2017.xes.gz'
$ipatManifest = '.\results\bpi17_contract_v0.1_local\offer_split_manifest.csv'
$ipatOutput = '.\results\bpi17_adapter_v0.1_local'

uv run python '.\scripts\build_bpi17_adapter.py' `
    --source "$ipatSource" `
    --manifest "$ipatManifest" `
    --contract '.\configs\bpi17_contract_v0.1.json' `
    --output-dir "$ipatOutput" `
    --plots
if ($LASTEXITCODE -ne 0) { throw 'Adapter acceptance failed. Inspect the output before proceeding.' }
Start-Process (Join-Path $ipatOutput 'adapter_diagnostics.html')
```

If your local preflight output directory has a different name, point `--manifest` to its `offer_split_manifest.csv`. The validated contract kit's `evidence/offer_split_manifest.csv` is an equivalent reference only if its pinned SHA-256 matches. The builder rejects a different manifest or source. No gzip extraction is necessary. Existing output directories are rejected, preserving previous evidence.

The existing project must provide NumPy, PyTorch, matplotlib and the dependencies imported by the frozen IPAT package (including h5py/PyYAML). The test runner itself is standard-library unittest. Do not change your project Python constraint or replace model files with old copies to run this patch. The uv environment mismatch warning does not by itself mean tests failed.

Send back:
- `bpi17_adapter_local_tests.txt`
- `results/bpi17_adapter_v0.1_local/adapter_acceptance.json`
- If any discrepancy appears, `cache/preprocessing.json` and the error output.

## Generated outputs

- `cache/preprocessing.json`: training-only means/scales, counts, constant flags and vocabularies.
- `cache/train.npz`, `validation.npz`, `calibration.npz`, `test.npz`: packed raw numeric inputs, categorical IDs, temporal inputs, missingness, offsets, labels and separate metadata.
- `adapter_acceptance.json`: source/code/runtime hashes, 17 gates and measured numerical errors.
- `adapter_diagnostics.png`, `.svg`, `.pdf`, `.html`: measured missingness, categorical unknown rates, an observed/missing/padding example and fitting counts. HTML is self-contained and works offline.

Logical cache hashes compare names, shapes, dtypes and tensor bytes, independently of ZIP container metadata. Compare source/contract/manifest hashes exactly; review code/runtime versions and numerical differences if preprocessing hashes differ across environments. Do not replace evidence merely to make hashes equal.

## Reading the results

Development counts remain 19,071 / 7,871 / 6,117 / 8,613 admitted offers. Fitting uses 13,865 training applications and 187,305 unique source events; packing overlapping training prefixes produces 253,781 event occurrences. These counts measure different units and are not a contradiction.

Recorded test `org:resource` unknown rate is 28.0064% of packed prefix event occurrences. A single UNK category is intentionally used without fitting on held-out data. This loses unseen category identity and may affect future predictions, but there is no predictive evidence yet quantifying that effect. No vocabulary expansion, resource removal or new experiment is silently introduced.

Numerical missingness near 89% for five offer-specific channels is expected under event-local availability: those fields are absent on preceding events and present at offer creation. Do not interpret this as 89% of offers lacking their creation values or backfill them into the past.

The optimiser check performs one finite step only. The full build's identity/gamma-zero check is a limiting-case engineering test, not the primary IPAT training setting. Its validation dropout is zero; actual pilot defaults/budgets/seeds must be specified in EXP-05. The adapter requires no train/test resampling or sequence truncation.
