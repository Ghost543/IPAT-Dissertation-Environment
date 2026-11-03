# IPAT ROB-01 v0.1

Accepted protocol implementation: controlled numerical missingness and protected event deletion, frozen EXP-05 checkpoints and CAL-01 temperatures, anchored to reviewed EXP-06 predictions. No retraining or dependency upgrades. The supplied development evidence is numerical verification only; run the full synthetic verifier locally before real evaluation.

## Install from the project root (PowerShell)

Place IPAT_ROB01_v0.1.zip in the project root, then:

```powershell
Expand-Archive -LiteralPath .\IPAT_ROB01_v0.1.zip -DestinationPath .\rob01_package
uv run python .\rob01_package\IPAT_ROB01_v0.1\install_rob01.py --project .
if ($LASTEXITCODE -ne 0) { throw "ROB-01 installation failed" }
```

Use a new extraction directory if rob01_package already exists. Installer copies only ROB-01 additions and backs up any differing earlier ROB-01 files. EXP-05, CAL-01 and EXP-06 source/evidence remain unchanged. It requires the existing accepted environment and does not modify pyproject.toml, lockfiles or dependencies.

## Full local verification, then real evaluation

Replace the clean path with the directory containing your actual accepted exp06_manifest.json and exp06_acceptance.json. It must contain the complete run, including per-model predictions and plots, not just the two uploaded JSON summaries.

```powershell
.\scripts\Run_ROB01.ps1 -CleanPath ".\results\YOUR_ACCEPTED_EXP06_DIRECTORY"
```

This executes synthetic unit tests and all 324 synthetic checkpoint-condition evaluations first. It checks the verification receipt, then starts the real frozen evaluation. It obtains EXP-05, CAL-01 and cache paths from the EXP-06 manifest. If those directories moved, provide explicit current paths:

```powershell
.\scripts\Run_ROB01.ps1 -CleanPath "C:\path\to\exp06_run" `
  -ParentPath "C:\path\to\exp05_run" `
  -CalibrationPath "C:\path\to\cal01_run" `
  -CachePath "C:\path\to\accepted_cache"
```

Default outputs are fresh timestamped directories under this project's results folder. There is no automatic AppData redirect. If desired, -ResultsRoot explicitly selects another location. Keep the same environment/device used for the accepted runs. Do not add --active or upgrade packages in response to the pre-existing uv environment-path warning.

If PowerShell's execution policy blocks the wrapper, use the Python commands below under the existing policy; no policy change is required by this package.

## Individual commands

```powershell
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$RobVerify = Join-Path "results" "rob01_runner_$Stamp"
uv run python scripts/verify_rob01.py --output $RobVerify
if ($LASTEXITCODE -ne 0) { throw "ROB-01 verification failed; retain output" }
```

Then set $Clean to the accepted EXP-06 run directory and use its manifest paths:

```powershell
$Clean = "C:\path\to\accepted_exp06_run"
$M = Get-Content -LiteralPath (Join-Path $Clean "exp06_manifest.json") -Raw | ConvertFrom-Json
$RobRun = Join-Path "results" "rob01_$Stamp"
uv run python scripts/run_rob01.py `
  --parent $M.parent --calibration $M.calibration --cache $M.cache `
  --clean $Clean `
  --verification (Join-Path $RobVerify "rob01_runner_acceptance.json") `
  --output $RobRun
if ($LASTEXITCODE -ne 0) { throw "ROB-01 evaluation failed; retain output" }
Start-Process (Join-Path $RobRun "plots\report.html")
```

The real run repeats the frozen parent replay checks and zero-corruption clean checks before 324 stress evaluations. Each inference is itself repeated for exact determinism. The console displays progress; no runtime estimate is asserted. No auto-resume is implemented: retain failed/interrupted attempts and select a fresh output directory. Never manually edit acceptance files or bypass a failed integrity gate.

## Return for review

From the verification output: rob01_runner_acceptance.json and fixture_tests.txt.
From the real output: rob01_acceptance.json, rob01_summary.json and corruption_counts.csv. Share plots or report.html for interpretation. Keep the complete run directory because per-offer predictions, corruption plans and manifests provide replay evidence.

## Reading results

- Oriented NLL change = clean minus corrupt: negative means deterioration.
- Paired advantage = comparator oriented change minus T00 oriented change: positive favours comparator. Primary is raw T11-T00.
- Calibration deterioration = corrupt minus clean ECE: positive means worse calibration.
- Absolute corrupted NLL is reported alongside changes.
- Replicates, severities and families are averaged within each training seed before seed means; bootstrap resamples whole applications and conditions on the fixed corruption realisations/predictors.
- All length strata use original clean lengths. Missing values, observed zero and padding remain distinct.
- PASS concerns faithful execution, not evidence that IPAT is superior.

## Development verification limitation

This development workspace has no PyTorch. The included numerical tests and 324-condition saved-prediction report fixture were executed without any model checkpoints. The local verifier exercises real neural/tree checkpoint-loading and inference paths on synthetic data. No real BPI17 robustness result or local integration PASS is included or claimed.
