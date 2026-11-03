# EXP-05 protocol acceptance and implementation record

The user accepted EXP05_Controlled_Pilot_Protocol_v0.1 on 13 September 2026 and authorised implementation. The original proposal is retained byte-for-byte (SHA256 8fb8fb2531c51f640fffc028e25875451113c3700e52b02f84fa832858b72140); its historic “proposed” wording is superseded by this acceptance record, not rewritten.

The frozen MATH specifications and BPI17 adapter remain unchanged. The new runner implements the approved four Transformer arms, GRU, GBDT and prevalence dummy, three seeds, twenty neural epochs, effective batch 64, validation-NLL checkpoint selection and natural class frequencies. Calibration and test predictive loading is prohibited. EXP-05 is not complete until local runner checks, source review, resource preflight, full execution and interpretation have passed their applicable gates.

Development evidence is packaged in evidence/. The synthetic integration fixture uses one epoch, 25 generated training offers and 12 generated validation offers, and cannot establish BPI17 predictive performance. The source-review template is deliberately NOT_VERIFIED. An HTTP 403 from the official repository does not establish a licence or lack of permission. Source/version/terms evidence is still required before the formal pilot.

Implementation defaults are engineering choices, not new mathematical claims. The train-only resource preflight uses disposable seed 999, fifty effective-batch updates per neural arm, with the longest training prefix included in its first batch. It neither loads validation nor retains models for the pilot. Its purpose is device/microbatch feasibility; it provides no research hypothesis evidence.

Resume identity covers configuration, preprocessing/source identifiers, module hashes and runtime/device. Exact recovery is checked within the development environment. Cross-version or cross-device bitwise equivalence is not claimed. RSS is sampled process memory (50 ms), not incremental model memory; if a run is interrupted, the resumed process peak does not reconstruct the previous process's peak. Preserve logs of every invocation. GPU allocation peaks are recorded separately.

This is a progress supplement to IPAT_Dissertation_Writing_Context_Lock_v1.0. The canonical mathematical handoff remains authoritative for mathematics; the live tracker and this dated acceptance record carry later progress. Do not read fixture PASS as EXP-05 completion.

## Implementation evidence sources

- PyTorch reproducibility documentation: https://docs.pytorch.org/docs/stable/notes/randomness.html — reproducibility scope, deterministic execution and RNG controls; verified 13 September 2026. It does not promise equality across releases/platforms.
- scikit-learn HistGradientBoostingClassifier documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html — estimator options and missing-value handling; verified 13 September 2026. The fixed 200-iteration budget is our accepted engineering choice.
- scikit-learn log-loss documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html — negative log-likelihood definition; verified 13 September 2026. The runner computes binary NLL directly from logits to avoid unnecessary probability clipping.
- Further methodological references and IPAT-specific mathematical derivations are retained in the accepted protocol. No external source is presented as evidence of an IPAT experimental result.
