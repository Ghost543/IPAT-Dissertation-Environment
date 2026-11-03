# IMP-01 — Matched standard-attention Transformer baseline

## Purpose

IMP-01 establishes the control architecture that IMP-02 must match.  The baseline uses standard
scaled dot-product self-attention only.  It intentionally contains no retention branch and no
`gamma` parameter.

## Frozen matching contract

The following must remain identical between IMP-01 and IMP-02 unless a later task explicitly
changes them:

- DATA-01 inputs and split membership;
- train-only standardisation statistics;
- fixed-identity numerical representation interface;
- categorical embeddings;
- temporal representation;
- explicit missingness mask;
- MATH-03 semi-orthogonal event projection;
- number of heads/layers and feed-forward width;
- pre-LayerNorm residual block structure;
- masked-mean sequence pooling;
- classification head;
- optimiser/training protocol and seeds in matched experiments.

IMP-02 should replace only the attention-logit mechanism with the MATH-04/05 retention term while
preserving a `gamma=0` path that is numerically equivalent to this standard attention implementation.

## Architecture defaults

With `configs/model/baseline_v0.1.yaml`:

- six numerical features;
- each fixed numerical feature emits `[standardised scalar || learned 2-D expansion]`;
- event-type embedding: 4-D;
- nuisance-category embedding: 4-D;
- temporal block: standardised timestamp, standardised gap, relative position and four sine bases;
- missingness block: six explicit mask values.

Therefore:

```text
d_num  = 6 * (1 + 2) = 18
d_cat  = 4 + 4         = 8
d_time = 3 + 4         = 7
d_mask = 6
d_in   = 39
```

For four heads, MATH-03 gives:

```text
d_model = 4 * ceil(39 / 4) = 40
```

These hyperparameter values are engineering defaults.  The structural rule `d_model >= d_in` and
`n_heads | d_model` is the mathematical requirement.

## Preservation boundary

`u -> z_pre_norm` uses a trainable reduced-QR semi-orthogonal map with `W^T W = I` (up to floating
point precision).  The preservation claim stops at `z_pre_norm`; subsequent LayerNorm/Transformer
operations are not claimed to be injective.

## Installation

Copy this patch into the existing dissertation repository, preserving paths.  The project must
already be installable as a `src/` package.

## Unit tests

```powershell
uv run pytest tests/test_baseline.py -q
```

## Tiny-sample smoke overfit

After DATA-01 debug generation exists:

```powershell
uv run python scripts/smoke_baseline.py
```

This is an IMP-01 implementation acceptance check only.  The formal matched baseline/IPAT tiny
sample experiment remains EXP-01 after IMP-02 is implemented.

## Completion gate

IMP-01 is complete when:

1. baseline unit tests pass in `IPAT-DISS-ENV-v1`;
2. forward/backward on DATA-01 contains no NaN/Inf;
3. attention weights sum to one across valid source events and padding receives zero mass;
4. `W^T W` is identity within the declared numerical tolerance;
5. the model overfits the configured 32-sequence synthetic subset;
6. no retention/gamma parameters are present;
7. model/config/environment identifiers are recorded before the first formal experiment.
