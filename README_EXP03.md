# EXP-03 v0.1 — informative-event position sweep

Status: implementation supplied; local acceptance and formal experiment results pending.
The user approved mean within-triplet probability range and ten paired training seeds on
8 September 2026. No frozen MATH-01–08, DATA-01 or IMP-01–03 file is changed by this patch.

## What this experiment asks

When a designated informative event is moved early/middle/late, how much do predictions
change? Does IMP-02 retention-aware attention reduce this sensitivity relative to IMP-01,
and what is the incremental effect of IMP-03's adaptive representation conditional on
retention attention? These are hypotheses, not mathematical consequences of retention
or monotonic numerical mappings.

For triplet k, model m, seed s and position v, define p_hat = sigmoid(logit),
D_msk = max_v p_hat_mskv - min_v p_hat_mskv, and PS_ms = mean_k D_msk.
Lower PS means greater prediction stability, but a constant useless predictor also has
PS=0. Always interpret PS with predictive quality and the secondary logit range.
There is no favourable-effect or p-value completion gate.

MATH-03 includes explicit position/time; MATH-04 produces a source-level latent retention
propensity; MATH-05 uses S + gamma * centred log-retention + padding mask; MATH-06 remains
task-loss-only. None guarantees position invariance. Numerical injectivity stops at its
scoped representation boundary, not throughout the Transformer.

## Intervention and important limitations

Use the existing canonical DATA-01 train.h5 (8,000), val.h5 (2,000), and
position_sweep.h5 (1,000 triplets/3,000 sequences). Calibration and canonical test files
are not read or used for selection. This runner NEVER regenerates or overwrites data files.
It reconstructs the sweep in memory solely to verify the frozen generator contract.

The probe is the original first signal event among three signals; it is placed at
round(0.1*(T-1)), round(0.5*(T-1)), round(0.9*(T-1)) in zero-based indexing.
Its content and associated gap move with it. Other events retain their relative order;
insertion shifts their indices and timestamps are reconstructed by cumulative gaps.
Labels, true probability and stored generating logit are identical within each triplet.

Two limitations must remain in Chapter 3 and the results interpretation:

1. This is a sequence-location intervention with reconstructed timestamps, not a pure
   positional-encoding ablation and not a full permutation-invariance test.
2. Canonical training has temporal label weight 0.4; the sweep has weight 0.0.
   Within-triplet truth invariance is verified, but absolute sweep performance reflects
   this evaluation-profile shift. Do not claim a pure in-distribution position effect.

The synthetic signal membership is visibly associated with event type A/B. Diagnostic
ranking of these types is not evidence of discovering unobserved causal importance.
The task target is stochastic and depends on signed contributions from three signals;
perfect held-out accuracy is neither required nor expected as an acceptance condition.

## Components and controls

| File | Responsibility |
|---|---|
| src/ipat/experiments/exp03_contract.py | Data integrity, triplet pairing, metrics and seed-level analysis |
| src/ipat/experiments/exp03.py | Matched construction, training, checkpointing, evaluation, replay and resume |
| configs/experiments/exp03_position_sweep_v0.1.yaml | Recorded configurable engineering settings |
| scripts/run_exp03.py | PowerShell-compatible entry point, preflight and formal run |
| tests/test_exp03.py | 34 named unit/integration tests; explicit dependency skips |
| scripts/verify_exp03.py | Acceptance evidence JSON; treats any skip as incomplete |

All models share the same train-only preprocessing, baseline dimensions, explicit
shared initial parameter state, epoch-specific shuffled order, AdamW settings, maximum
epochs and validation-NLL stopping rule. Baseline/retention initial zero-effect forward
parity is checked. Full IPAT has a non-identity mixture initially; its predictions are
not required to equal IMP-02. Extra parameter counts are reported.

Ground-truth signal_mask, contribution and p_true are excluded from model input dictionaries.
Only training labels enter task BCE. Diagnostic attention/retention calculations occur
after the selected checkpoint is restored. No temperature scaling, MI, CKA or sweep loss
is used in training, checkpoint selection or tuning.

Same stopping RULE does not imply the same number of epochs. Consumed batch-order
prefixes are checked equal, and actual training trajectories/budgets are reported.

## Engineering defaults (not theory)

- Fixed training seeds: 20260908 through 20260917 inclusive. Calendar-like integers
  have no theoretical significance; they are a predeclared unique seed schedule.
- Maximum epochs 30; batch size 64; AdamW lr 0.001, weight decay 0.01.
- Validation NLL measured once per epoch. Strictly lower NLL selects a checkpoint;
  ties retain the earliest checkpoint. Patience 5, min_delta 0.0.
- Baseline dropout 0.10; gamma 1.0 fixed; rho 0.25; scaled-asinh alphas 0.25, 1, 4.
- CPU default, one intra-op thread, float32, deterministic algorithms, TF32 disabled.
- No scheduler, gradient clipping, diagnostic penalty or extra parameter tuning.
- Replay of all three models for the FIRST seed, additional to the 30 primary runs.
  Replay trajectories and selected states must match exactly; output atol defaults to 0.

Review defaults before the first formal run. If changing any configuration, use a new
output directory and document why. Do not select a setting based on sweep performance.
These are fixed-budget comparisons, not assertions of equal optimal tuning for each model.

## Metrics and statistical scope

Primary: mean within-triplet probability range. Secondary: mean logit range; AUROC,
accuracy at logit>=0, NLL and binary Brier by position, equal-position macro average,
and worst position (minimum AUROC/accuracy, maximum NLL/Brier).

Attention mass: mean over layers and heads, summed over valid query rows, divided by
valid query count. Record mass assigned to all signal keys and to the probe key, plus
uniform reference masses 3/T and 1/T. Padded query/key attention must be zero.

Retention diagnostics: valid signal/nuisance mean propensity, probe propensity,
centred-log-retention signal minus nuisance margin, and per-sequence signal-ranking
AUROC. Aggregate sequences equally within position. Probe propensity range is paired.
These are mechanism-alignment diagnostics, not causal explanations. Stronger claims
would require the frozen targeted-versus-random intervention and seed-stability evidence.

Aggregate within training seed first. Report every seed, mean, sample SD, median and
approximate 95% Student-t intervals. Intervals measure training-seed variability conditional
on the FIXED train/validation/sweep datasets; they do not include dataset-sampling uncertainty.
Do not treat 3,000 correlated triplet rows or 90,000 model predictions as independent seeds.

Three predeclared sensitivity contrasts: retention-baseline, full-retention, full-baseline.
Negative differences favour the first model on sensitivity. Secondary Wilcoxon-style
signed-rank inference enumerates all 2^n sign assignments over midranks, discarding exact
zero differences. This handles ties without mislabelling SciPy's ordinary exact method.
All-zero differences give p=1. Holm-adjust the three secondary p-values. These inference
assumptions require independent, symmetric seed differences under the null; n=10 does
not guarantee adequate power or reliable distributional inference. No favourable-result gate.

## Apply safely (Windows PowerShell)

Extract the ZIP into a temporary directory and copy its six new Python/configuration
files and this README into the matching locations under your existing project.
The archive contains no replacement model, dataset, environment or pyproject file.
If any EXP-03 destination file already exists, preserve it and compare before replacing.

From the project root, first run:

```powershell
uv run pytest tests/test_exp03.py -q
uv run python scripts/verify_exp03.py --output results/exp03/exp03_local_acceptance_v0.1.json
uv run python scripts/run_exp03.py --preflight-only --out-dir results/exp03/v0.1
```

Local gate: all 34 tests pass, none skipped; verifier JSON status PASS; preflight status
PREFLIGHT_PASS with canonical counts and exact generator reproduction. Send these outputs
for review before the 30-run experiment. Test fixtures train only tiny development data;
their results are never formal EXP-03 findings. The verifier refuses to overwrite existing
acceptance evidence; use a new filename for subsequent attempts.

After the local gates are reviewed:

```powershell
uv run python scripts/run_exp03.py --resume --out-dir results/exp03/v0.1
```

`--resume` is necessary after preflight because the manifest already exists. Use the same
command after an interruption. Completed model/seed runs are verified and reused; an
interrupted model run restarts from its original seed (NOT from a partially trained epoch).
Do not run concurrent processes into the same output directory. Never delete old evidence
to resolve a fingerprint mismatch: use a new directory and document the change.

Keep device/configuration/code/data/runtime unchanged between preflight, execution and
resume. Resume verifies input files, source files, preprocessing, configuration, runtime,
checkpoints, prediction archives and saved run summaries. It refuses mismatches.
No runtime guarantee is given before measuring your hardware. There are 30 independent
model/seed fits plus three reproducibility replays, not 33 independent experimental runs.

The uv VIRTUAL_ENV warning does not by itself invalidate the experiment. Retain the
configured project environment; do not switch to --active without intentionally changing
and documenting the runtime. The supplied patch does not alter your environment.

## Evidence and completion

Output directory contains manifest.json, preflight.json, progress.json, one run.json,
checkpoint.pt and predictions.npz per model/seed, three replay runs, and
exp03_position_sweep_v0.1.json. Prediction NPZ files use numeric/string arrays (no pickle).
Keep the entire directory, not just the summary. Raw predictions allow metrics to be audited.

Expected formal total: 30 runs and 90,000 primary predictions, plus separate replay evidence.
The final runner status is EXECUTION_COMPLETE_REVIEW_REQUIRED, not hypothesis PASS.
Freeze only after local tests, all execution gates and investigator interpretation pass.
Null/adverse scientific outcomes are retained. Lower sensitivity with poorer predictive
quality can reflect trivial prediction compression and is not automatically robustness.

## Verified sources and attribution

The endpoint, seed schedule, implementation and IPAT equations are project decisions;
the following sources do not introduce or validate IPAT:

- Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
  https://papers.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
- Liu, N. F., et al. (2024). Lost in the middle: How language models use long contexts.
  Transactions of the Association for Computational Linguistics, 12, 157–173.
  https://aclanthology.org/2024.tacl-1.9/ (Motivation in a different task setting.)
- Jain, S., & Wallace, B. C. (2019). Attention is not explanation. NAACL-HLT, 3543–3556.
  https://aclanthology.org/N19-1357/
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation.
  Journal of the American Statistical Association, 102(477), 359–378.
  https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
- PyTorch. Reproducibility. https://docs.pytorch.org/docs/stable/notes/randomness.html
  Reproducibility is scoped to a fixed runtime, not guaranteed across platforms/releases.
- SciPy. wilcoxon. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
  Ordinary exact-method p-values are not exact when ties/zeros alter the null distribution.

## Dated continuity addendum — 8 September 2026

This dated EXP-03 record supplements, but does not rewrite, the older writing context lock.
MATH-01–08, DATA-01, IMP-01–03, EXP-01 and EXP-02 remain frozen at their accepted states.
EXP-03's primary endpoint and ten-seed plan are accepted; implementation is delivered for
local validation. No EXP-03 performance, generalisation, robustness or superiority result
has yet been established. The live tracker remains the progress authority.

### Development validation actually performed

Python compilation/static syntax checks passed. Of 34 acceptance test methods,
19 metric/configuration checks passed, 9 DATA-01 checks were skipped because h5py
was unavailable, and 6 PyTorch integration checks were skipped because PyTorch/h5py
were unavailable. No failures occurred among executed checks. The verifier returned
INCOMPLETE_OR_FAILED (exit code 1), correctly refusing to count skips as acceptance.
See deliverables/exp03_development_checks_v0.1.json for exact evidence. No formal
model training, canonical HDF5 preflight or full PyTorch acceptance has been run here.
