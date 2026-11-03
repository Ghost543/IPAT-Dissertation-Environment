# IMP-03 — Adaptive Information-Preserving Representation v0.1

This patch implements the frozen MATH-02 numerical representation while retaining the exact
IMP-01/IMP-02 downstream widths and model-facing interfaces.

For each observed standardised scalar, the encoder computes

\[
p_{jk}=\operatorname{softmax}(a_j)_k,
\qquad
\bar\pi_{j0}=\rho+(1-\rho)p_{j0},
\qquad
\bar\pi_{jk}=(1-\rho)p_{jk}\quad(k>0),
\]

\[
g_{tj}=\sum_k\bar\pi_{jk}\psi_k(\widetilde x_{tj}),
\qquad
z^{(num)}_{tj}=[g_{tj}\;\|\;h_j(g_{tj})].
\]

The candidates are the identity and configurable scaled-asinh transforms:

\[
\psi_0(x)=x,
\qquad
\psi_k(x)=\frac{\operatorname{asinh}(\alpha_k x)}{\alpha_k},\quad \alpha_k>0.
\]

Each candidate is strictly increasing and the mixture satisfies
`dg_j/dx >= pi_j0 >= rho > 0`. This is the scoped scalar-preservation property; it is not an
end-to-end losslessness claim. The MATH-03 projection property still stops at `z_pre_norm`.

## Frozen interface decisions

- training-only numerical means and scales are reused unchanged;
- one learned candidate mixture is used per numerical feature;
- the identity candidate is mandatory and has weight at least `identity_floor`;
- missing numerical values and padded events remain neutral in the numerical block;
- missingness stays explicit in the separate event mask block;
- numerical output width remains `n_num * (1 + numeric_embedding_dim)`;
- categorical, temporal, projection, attention, residual, pooling and classifier interfaces are
  unchanged;
- full IPAT uses the frozen IMP-02 event-level, query-independent, head-shared retention branch;
- the v0.1 gradient objective remains task loss only.

`identity_floor: 0.25` and `asinh_scales: [0.25, 1.0, 4.0]` are configurable engineering
starting values, not theoretical constants or completed model-selection results.

## Files

- `src/ipat/model/adaptive.py`
- `src/ipat/model/__init__.py`
- `configs/model/full_ipat_v0.1.yaml`
- `tests/test_adaptive_representation.py`
- `scripts/verify_adaptive_representation.py`
- `scripts/smoke_full_ipat.py`

## Apply

Extract this patch over the dissertation repository root, preserving paths.

## Acceptance sequence

Run in PowerShell from the dissertation repository:

```powershell
uv run pytest tests/test_adaptive_representation.py -q
```

Expected: 15 passing IMP-03 cases.

Then run the invariant report:

```powershell
uv run python scripts/verify_adaptive_representation.py --output results/imp03/imp03_acceptance_v0.1.json
```

Required outcomes include:

- mixture weights are non-negative and sum to one;
- the identity weight never falls below `identity_floor`;
- the sampled mixture is strictly increasing and its derivative respects the identity-floor bound;
- the semi-orthogonal projection retains its Gram and reconstruction checks;
- the identity-only, `gamma=0` limit reproduces IMP-01 exactly;
- every IMP-02 shared trainable parameter has identical initial state under the same seed.

Finally, after canonical DATA-01 is present, run:

```powershell
uv run python scripts/smoke_full_ipat.py
```

The smoke test is an implementation/trainability gate only. It is not evidence of generalisation,
robustness, calibration or superiority.
