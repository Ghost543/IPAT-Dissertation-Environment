# IMP-02 — Retention-Aware Attention v0.1

This patch extends the frozen IMP-01 standard-attention Transformer while preserving the matched
representation, projection, width/depth, residual architecture, pooling and classifier.

The only architectural change is the attention-logit mechanism:

\[
L_{bhis}
=
S_{bhis}
+
\gamma\widetilde{\lambda}_{bs}
+
M^{pad}_{bs}.
\]

Retention is computed once per event from the pre-LayerNorm projected event representation:

\[
\eta_{bs}=w_r^\top z_{bs}+b_r,
\qquad
\lambda_{bs}=\log\sigma(\eta_{bs}),
\]

\[
\widetilde{\lambda}_{bs}
=
\lambda_{bs}
-
\frac{1}{T_b}\sum_{u\in V_b}\lambda_{bu}.
\]

## Frozen v0.1 choices

- one retention score per event;
- shared across attention heads;
- query-independent;
- computed from `z_pre_norm`, before LayerNorm;
- stable `torch.nn.functional.logsigmoid`;
- `gamma >= 0`;
- `gamma` is a fixed buffer, not a learnable model parameter;
- retention scorer initialises with `w_r = 0`, `b_r = 0`;
- raw `r = sigmoid(eta)` is diagnostic only, not a calibrated probability or explanation.

`gamma: 1.0` in the config is an engineering starting value for smoke testing, not a theoretical
constant or final experimental choice.

## Files

- `src/ipat/model/retention.py`
- `src/ipat/model/__init__.py`
- `configs/model/retention_v0.1.yaml`
- `tests/test_retention_attention.py`
- `scripts/verify_retention.py`
- `scripts/smoke_retention.py`

## Install

Extract the patch over the dissertation repository root, preserving paths.

## Acceptance suite

Run:

```powershell
uv run pytest tests/test_retention_attention.py -q
```

Expected: 14 passing retention-specific tests.

Then run the exact matched limiting-case check on DATA-01:

```powershell
uv run python scripts/verify_retention.py
```

The required result is exact parity at `gamma=0` for:
- full-model forward output;
- hidden states and attention;
- all shared IMP-01 parameter gradients;
- zero gradient into the retention scorer.

Then check trainability:

```powershell
uv run python scripts/smoke_retention.py
```

This uses the same 32-sequence tiny-sample acceptance idea as IMP-01. It is an implementation smoke
test, not a generalisation result and not yet the formal paired EXP-01 analysis.

## Independent validation before handoff

The patch was checked outside the user's frozen dissertation environment:

- `14/14` retention-specific unit tests passed.
- `38/38` combined DATA-01 + IMP-01 + IMP-02 tests passed.
- `verify_retention.py` reported exact `gamma=0` forward and shared-gradient parity.
- Tiny-sample trainability reached 100% training accuracy with loss < 0.05.

These are development checks only. IMP-02 should remain below 100% until the same acceptance commands
pass in `IPAT-DISS-ENV-v1`.

## Main implementation invariants

1. `gamma=0` reproduces IMP-01 exactly.
2. Equal retention values reproduce IMP-01 for any non-negative gamma.
3. Zero-effect retention initialisation reproduces IMP-01 even when gamma is non-zero.
4. Padding receives zero attention and padded query outputs are zeroed.
5. Centred log-retention sums to zero over valid events.
6. The analytic MATH-05 derivatives with respect to gamma and eta are unit-tested.
7. `gamma` is not an optimiser parameter in v0.1.
8. Non-uniform retention with positive gamma changes the attention distribution as intended.
