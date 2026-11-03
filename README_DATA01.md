# DATA-01 — IPAT-SYN-v0.1

This patch implements the frozen synthetic validation design for IPAT.

## Included

- `src/ipat/data/synthetic.py` — canonical generator, ground truth, persistence and corruptions.
- `configs/data/ipat_syn_v0.1.yaml` — configurable canonical defaults.
- `scripts/generate_ipat_syn.py` — reproducible clean/corruption/position-sweep generation.
- `tests/test_synthetic_data.py` — DATA-01 acceptance tests.

## Apply to the existing repository

Copy the patch contents into the repository root, preserving paths.

## Acceptance test

```powershell
uv run pytest tests/test_synthetic_data.py -q
```

## Fast debug generation

```powershell
uv run python scripts/generate_ipat_syn.py --debug
```

Default output:

```text
data/processed/ipat_syn_v0.1/
```

## Canonical generation

```powershell
uv run python scripts/generate_ipat_syn.py
```

## Ground truth retained

Each event stores exact `signal_mask` (g_bs), signed `contribution` (a_bs), latent `u`, full numerical values, observed/masked numerical values, temporal variables and categorical values. Each sequence stores the exact generator logit, Bayes probability `p_true`, and sampled label `y`.

Feature generation and label sampling use separate RNG streams, so changing the label seed cannot change features or generator truth.

## Completion gate

Do not mark DATA-01 complete until the acceptance suite and `--debug` generation pass in the frozen dissertation environment, and the generated summary has been inspected for sequence length, prevalence, missingness and exactly three signal events per sequence.
