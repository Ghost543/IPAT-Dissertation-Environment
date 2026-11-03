# EXP-01 — Matched Tiny-Sample Sanity Test v0.1

EXP-01 is the first **matched empirical training harness** for the frozen IMP-01 standard Transformer and IMP-02 retention-aware Transformer.

It is deliberately narrow: the question is whether both implementations can learn the **same deliberately tiny synthetic sample** under the same training protocol. It is **not** a generalisation experiment and must not be used to claim that one architecture is better than the other.

## Matched protocol

The harness fixes:

- the canonical 8,000-sequence `IPAT-SYN-v0.1` train split (debug-sized HDF5 files are rejected);
- the same deterministic 32-sequence subset;
- the same sample order/full batch;
- the same train-only preprocessing statistics;
- the same shared initial parameter state;
- dropout `0.0` for deterministic acceptance;
- AdamW with the same learning rate and weight decay;
- the same maximum step budget;
- the same stopping rule;
- the same model dimensions, representation, projection, residual blocks, pooling and classifier.

The retention-aware model adds only the frozen IMP-02 retention scorer and retention-logit path. Its scorer starts at `w_r=0, b_r=0`, so at step 0 and `gamma=1.0` its attention/output are mathematically identical to IMP-01 and are required to agree numerically within a pre-specified `1e-6` tolerance. Exact numerical equality at `gamma=0` remains the separate EXP-02 result.

The configured acceptance gate is:

```text
subset size      32
max steps        300
learning rate    0.003
weight decay     0.01
dropout          0.0
gamma             1.0
target loss      <= 0.05
target accuracy  1.0
```

`gamma=1.0` remains an engineering starting value, not a theoretically privileged or final experimental choice.

## Install

Extract over the dissertation repository root, preserving paths.

## Unit tests

```powershell
uv run pytest tests/test_exp01.py -q
```

These tests verify deterministic subset selection, exact shared initial state, the precise IMP-02-only parameter set, zero-effect initial forward parity within the frozen tolerance, one-step reproducibility and JSON evidence persistence.

## Run EXP-01

First ensure the canonical DATA-01 files exist. If your current HDF5 files were created with `--debug`, regenerate them without that flag:

```powershell
uv run python scripts/generate_ipat_syn.py
```

Then run:

```powershell
uv run python scripts/run_exp01.py
```

The run writes:

```text
results/exp01/exp01_tiny_sample_v0.1.json
```

The JSON records the sampled sequence IDs/indices, subset prevalence, preprocessing statistics, initial-state hash, exact initial parity, parameter counts, both training trajectories, stopping steps and retention contrast.

## Completion rule

EXP-01 passes only if **both** models reach:

\[
\mathcal{L}_{train}\le 0.05
\]

and

\[
\mathrm{Accuracy}_{train}=1.0
\]

within 300 matched full-batch optimisation steps.

Step-to-target is logged only as a descriptive debugging quantity. A lower step count here is not evidence of better generalisation, robustness or predictive performance.
