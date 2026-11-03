# IMP-03 v0.1.1 precision-gate correction

This narrow correction responds to the Windows dissertation-environment failure at
`maximum_weight_sum_error <= 1e-7`.

The frozen MATH-02 routing equation and `AdaptiveNumericalEncoder` are unchanged. The former
decimal threshold was below float32 epsilon at one, so valid non-uniform softmax mixtures could
fail because of normal floating-point rounding. The verifier and corresponding unit test now use
and record the tight dtype-aware engineering tolerance
`8 * torch.finfo(weights.dtype).eps`.

Apply the archive over the repository root, then rerun:

```powershell
uv run pytest tests/test_adaptive_representation.py -q
uv run python scripts/verify_adaptive_representation.py --output results/imp03/imp03_acceptance_v0.1.json
```

Do not mark IMP-03 complete until both commands pass and the JSON evidence has been inspected.
