# EXP-04 — frozen-checkpoint corruption evaluation, v0.1

Protocols accepted by the investigator on 9 September 2026. Implementation release;
formal local evaluation and interpretation remain outstanding.

## Scientific contract

Evaluate the three EXP-03 models (baseline, retention-only and full IPAT), using
all ten validation-selected paired-seed checkpoints. No model retraining,
corruption-specific checkpoint selection, or adaptation of preprocessing.

Primary endpoint: raw NLL oriented change, `NLL_clean - NLL_corrupted`.
Average severities within each corruption family, then average the four family
means. Positive change means improvement; a positive paired difference favours
the first model. Report absolute clean/corrupted NLL, AUROC, accuracy and Brier
scores alongside changes. Three pre-specified paired comparisons use the EXP-03
secondary two-sided sign-flip test on midranks, with Holm correction across three
comparisons. Its null interpretation assumes symmetric independent seed
differences. Approximate t intervals describe training-seed uncertainty conditional
on the fixed test data and corruption realisations, not dataset uncertainty.

MATH-08 calibration: top-label ECE uses 15 equal-width bins on [0,1]; the final bin
includes 1. Empty bins contribute zero. A 15-bin equal-mass sensitivity analysis
uses stable sequence order to resolve ties; equal-confidence observations may
span bins. ECE change is `ECE_corrupted - ECE_clean`: positive means deterioration.
Both raw and temperature-scaled panels are saved. One positive temperature is
fitted by clean-calibration NLL per checkpoint, then fixed under corruption.
Accuracy and AUROC are checked for invariance under positive temperature scaling.
AUROC uses logits to avoid sigmoid saturation introducing artificial numerical
ties. Probability-dependent metrics use float64 sigmoid probabilities; NLL uses
a stable logaddexp calculation on signed logits.

Temperature bounds [0.01, 100] and optimisation tolerances are configurable
engineering defaults. Boundary solutions are flagged, not concealed or treated
as favourable scientific evidence. The optimiser includes T=1 among candidates
so fitting NLL cannot become worse. No calibration data enter the model optimiser.

DATA-01 files are reused byte-for-byte. Preflight reproduces arrays in memory
and compares them exactly; it does not regenerate or overwrite stored data.
Canonical splits are 8000/2000/2000/4000. Test conditions are clean plus:

| Family | Existing files | Interpretation |
|---|---|---|
| Missingness | test_missing_10/20/40.h5 | Target total missingness; clean already has 5% expected missingness |
| Deletion | test_delete_10/25.h5 | Random deletion, compacted positions; surviving original timestamps and gap fields retained |
| Numerical noise | test_noise_10/25.h5 | Gaussian noise scaled by observed training-feature standard deviation |
| Time jitter | test_timejitter_10/25.h5 | Multiplicative positive gap perturbation, reconstructed timestamps |

Each corruption starts from clean test data. Severities use different random
realisations, not nested masks. Original labels and generator truth are retained;
they are not recalculated from corrupted observations. Generator probabilities
are not interpreted as posterior probabilities given degraded observations.

MATH-02 scalar injectivity does not recover missing/deleted data and does not
prove end-to-end robustness. Retention and attention are descriptive diagnostics,
not causal evidence. Deletion may remove all signal events. Undefined per-sequence
ranking/conditional-mean diagnostics are NaN in NPZ, null when no eligible rows
remain in JSON, and have explicit eligibility/undefined counts. Signal attention
mass of zero when there is no signal is a meaningful mass, not a ranking score.

## Implementation structure

- `src/ipat/experiments/exp04_contract.py`: frozen condition enumeration, metrics,
  calibration fitting, aggregation and read-only data verification.
- `src/ipat/experiments/exp04.py`: parent verification, model loading, inference,
  per-checkpoint resume, exact first-seed replay and evidence persistence.
- `src/ipat/experiments/exp04_report.py`: plots, CSV tables and self-contained HTML.
- `scripts/run_exp04.py`: formal command-line entry point, preflight and report rebuild.
- `scripts/verify_exp04.py`: acceptance suite with machine-readable local evidence.
- `tests/test_exp04.py`: 25 engineering acceptance tests, including a small full
  end-to-end fixture. Its models/data are explicitly DEVELOPMENT_ONLY.
- `configs/experiments/exp04_corruption_retention_v0.1.yaml`: engineering defaults.

The patch adds EXP-04 files only. It does not replace the frozen model/data code,
EXP-03 scripts, project environment configuration, lockfile or existing results.

## Install from Windows PowerShell

Download `IPAT_EXP04_v0.1_patch.zip` into Downloads. Run this from your existing
`IPAT_Dissertation_Environment_v1` project root:

```powershell
$exp04Archive = Join-Path $env:USERPROFILE 'Downloads\IPAT_EXP04_v0.1_patch.zip'
if (-not (Test-Path -LiteralPath $exp04Archive)) { throw "Download the EXP-04 patch first." }
if (-not (Test-Path -LiteralPath '.\src\ipat\model\adaptive.py')) { throw "Run from the dissertation project root." }
Expand-Archive -LiteralPath $exp04Archive -DestinationPath . -Force
```

The patch relies on your existing accepted EXP-03 environment and dependencies,
including Matplotlib. Do not run `uv add`, upgrade PyTorch or regenerate datasets.
Formal preflight checks Python and scientific package versions against the parent
manifest. Extra EXP-04 source files are allowed, but every source path recorded
in the original parent manifest must still match. Windows path separators are
normalised only when resolving paths; the parent fingerprint remains unchanged.

## Acceptance, preflight and formal execution

```powershell
uv run python scripts/verify_exp04.py --output results/exp04/exp04_local_acceptance_v0.1.json
if ($LASTEXITCODE -ne 0) { throw 'EXP-04 acceptance failed; stop and share the output.' }

uv run python scripts/run_exp04.py --preflight-only
if ($LASTEXITCODE -ne 0) { throw 'EXP-04 preflight failed; stop and share the output.' }

uv run python scripts/run_exp04.py --resume
if ($LASTEXITCODE -ne 0) { throw 'EXP-04 execution stopped; inspect the error before resuming.' }

Start-Process '.\results\exp04\v0.1\exp04_report.html'
```

Default parent directory: `results/exp03/v0.1`. This must contain the original
manifest, summary and all 30 primary run directories, with checkpoints and
prediction archives. The summary supplied to this conversation alone is not
sufficient to evaluate models. If your existing parent folder differs, pass
`--parent-dir 'your\actual\exp03\folder'` to both preflight and execution.
Default data directory: `data/processed/ipat_syn_v0.1`.

If preflight already created an output folder, rerun it with `--resume`. A local
acceptance JSON is deliberately not overwritten: choose a fresh `--output` name
when recording another acceptance run. Completed per-checkpoint evaluation runs
are reused only after their hashes and experiment fingerprint pass. An interrupted
checkpoint evaluation is recomputed; model training is never repeated. Changing
configuration, source, inputs or environment requires a new EXP-04 output folder.
Keep the EXP-03 parent and original experiment evidence unchanged.

The familiar `VIRTUAL_ENV` warning means uv selected its configured project
environment. Do not switch to `--active` simply to remove the warning; use the
same accepted environment as EXP-03.

## Expected outputs

`results/exp04/v0.1/` contains:

- `exp04_corruption_retention_v0.1.json`: signed summary with every run, aggregate,
  confidence intervals, paired comparisons, eligibility counts and review flags.
- `exp04_report.html`: offline report with nine selectable figures and an
  interactive reliability explorer (model, seed, corruption); SVG download.
- `figures/`: nine figures, each in PNG, SVG and PDF.
- `metrics_by_seed.csv`, `macro_changes_by_seed.csv`: exact table exports.
- `seed_<seed>/<model>/`: clean-calibration logits; ten test-condition prediction
  archives containing raw logits, sequence IDs, labels and diagnostics; run receipt.
- `manifest.json`, `preflight.json`, `progress.json`, `report_manifest.json`:
  configuration, source/environment/data provenance and output hashes.

Formal primary inference covers 30 checkpoints x 10 conditions x 4000 sequences
= 1,200,000 test predictions, plus 60,000 clean-calibration predictions. The
first seed's three models repeat all ten conditions and calibration inference
for reproducibility, excluded from aggregate inference. Scaled metrics are derived
from the same saved logits, not an additional model forward pass.

HTML and plots are generated automatically. To rebuild presentation outputs
without inference:

```powershell
uv run python scripts/run_exp04.py --report-only
```

Share the formal summary and local acceptance JSON after the run. You may also
share the HTML for visual interpretation; keep checkpoint and prediction archives
locally for detailed audit. Successful execution remains
`EXECUTION_COMPLETE_REVIEW_REQUIRED`, not a favourable hypothesis verdict.

## Development validation and limitations

The package includes `exp04_development_acceptance_v0.1.json`: 25 tests passed,
zero skipped, including frozen corruption reproduction, known calibration
optimum, metric signs, family weights, checkpoint corruption detection, replay,
resume mismatch rejection and complete fixture report generation. Development
Python/package versions and source hashes are recorded in that file.

Static plots were visually inspected. `exp04_html_development_checks_v0.1.json`
records a Node DOM harness exercise of all 30 fixture reliability selections,
nine figure selectors and SVG download handling. This checks control logic and
embedded values; a full browser-rendered layout check was unavailable in the
build environment. Formal local replication remains required. No development
prediction values are offered as dissertation findings.

## Sources and authority

The investigator-approved protocol and frozen project Math Spec define the IPAT
claims and numerical conditions. External sources support the broader methods,
not IPAT-specific performance claims:

- Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness
  to common corruptions and perturbations. International Conference on Learning
  Representations. https://arxiv.org/abs/1903.12261
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of
  modern neural networks. Proceedings of Machine Learning Research, 70,
  1321–1330. https://proceedings.mlr.press/v70/guo17a.html
