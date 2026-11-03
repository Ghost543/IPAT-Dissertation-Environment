# EXP-06: held-out clean-test evaluation protocol v0.1

Date: 20 September 2026. Status: **proposed for protocol review; implementation and test evaluation have not started**. EXP-06 is the proposed identifier for this stage, not a claim that a tracker row has been updated. This package contains a protocol and a machine-readable specification, not an experiment runner.

## 1. Purpose and research questions

EXP-05 selected model checkpoints on validation data. CAL-01 fitted one temperature for each frozen checkpoint on separate calibration data. Those are development and fitting results. EXP-06 will assess the resulting predictors on the reserved later-period test applications, without further fitting or selection.

The primary question is whether full IPAT (T11) has lower raw predictive NLL than the matched Transformer (T00) on the admitted test offers. Secondary questions examine the separate representation and retention components, the GRU and GBDT references, whether frozen temperature scaling improves held-out probabilistic scores, and how performance varies by class and sequence length.

This stage supplies an independent predictive evaluation relative to the existing fitting and checkpoint-selection samples. It is still one retrospective dataset, one chronological split, a bounded training budget, and three training seeds. Audit statistics for the test cohort were inspected during data preparation; do not call all test information previously unseen. No test predictive results have been supplied for this protocol.

Temperature scaling is conventional post-processing (Guo et al., 2017). It is not an IPAT innovation. Raw architectural comparisons and calibrated pipeline comparisons will be reported separately so that their interpretations remain distinct. A successful evaluation may find no IPAT benefit, worse calibrated test scores, or stronger performance from an external reference.

## 2. Frozen inputs and cohort

Reuse the accepted BPI17 contract, train-fitted preprocessing, EXP-05 selected checkpoints, and all 18 CAL-01 temperatures. Keep every admitted offer and its complete prefix. Do not retrain, select another epoch, fit another temperature, average temperatures, rebalance labels, truncate long histories, change thresholds, or add features.

| Partition | Offers | Role in this stage |
|---|---:|---|
| Training | 19,071 | Existing preprocessing and dummy prevalence are reused; no fitting |
| Validation | 7,871 | Replay selected checkpoints against saved predictions only |
| Calibration | 6,117 | Replay against CAL-01 saved logits and verify temperatures only; no refitting |
| Test | 8,613 | Frozen clean predictive evaluation |

Partition boundaries and exclusions remain those of the accepted contract: test offer-creation timestamps are at or after 20 October 2016 UTC; all offers from applications crossing partition boundaries were excluded. Check application disjointness across all four partitions from frozen ID metadata. Verify the test cache's expected count and logical hash before evaluation. A discrepancy is a provenance failure, not permission to rebuild the split.

The target remains recorded `Accepted` at offer creation, false=0 and true=1, using the inclusive creation prefix and the contract's feature exclusions. It is not certified final loan outcome or later `O_Accepted`. The trace-static feature convention, unresolved CreditScore-zero meaning and non-random crossing-application exclusions remain limitations. Identifiers are alignment/grouping metadata, never model inputs.

All 18 arm/seed pairs are mandatory:

| Arm | Numerical mapping | Sequence mechanism |
|---|---|---|
| T00 | Fixed identity | Standard attention |
| T10 | Adaptive identity plus scaled-asinh | Standard attention |
| T01 | Fixed identity | Retention-aware attention |
| T11 | Adaptive identity plus scaled-asinh | Retention-aware attention |
| GRU | Frozen reference encoder | Frozen recurrent reference |
| GBDT | Frozen prefix summary features | Frozen gradient-boosted tree reference |

Seeds are 1001, 1002 and 1003. Keep duplicate GBDT outputs and disclose their dependence; they are not independent stochastic replications. The dummy uses the unchanged training prevalence, 13234/19071, and receives no new calibration.

## 3. Comparisons and weighting specified before test inference

For offer i, label y and frozen logit l, use stable binary NLL in nats:

    loss(y,l) = logaddexp(0,l) - y*l
    p_raw = sigmoid(l)
    p_scaled = sigmoid(l/T_arm,seed)

Report scores per checkpoint first. Aggregate scores across the three seeds with an arithmetic mean and sample standard deviation (ddof=1). This is not an ensemble: do not average probabilities before computing the reported model score.

Each offer has equal weight in primary NLL. Also report application-macro NLL: first average offer losses within each application, then average the applications. This sensitivity uses a different weighting target, without changing the trained models.

**Primary endpoint and contrast:** raw offer-mean test NLL; per-seed delta is `L_raw(T11) - L_raw(T00)`, then averaged across seeds. Negative favours full IPAT. Report the magnitude, all three seed values and the descriptive paired application-cluster interval. No post-hoc non-inferiority margin or significance threshold will be introduced.

**Secondary comparisons:**

1. Raw NLL for T10, T01, GRU and GBDT minus T00, and every arm versus the unchanged dummy.
2. Calibrated NLL for each comparator minus calibrated T00. These assess complete calibrated pipelines, not solely the architectural mechanisms.
3. Within each arm/seed, scaled minus raw NLL, Brier score, fixed-bin ECE and equal-mass ECE. Negative favours scaling for these metrics. Unlike fitting NLL, test NLL is allowed to worsen.
4. Raw factorial contrasts: representation `L10-L00`; retention `L01-L00`; full `L11-L00`; interaction `L11-L10-L01+L00`. Compute within seed before averaging. The interaction is specific to this trained comparison and NLL scale; it is not proof of universal synergy.

No seed, model or reported endpoint is removed because its result is unfavourable. Three seeds do not support a strong confirmatory claim. Descriptive intervals are not multiplicity-adjusted hypothesis tests, and absence of an observed benefit is not proof of equivalence.

## 4. Calibration, discrimination and class metrics

Report NLL, Brier score, top-label ECE, equal-mass ECE, accuracy, balanced accuracy, macro-F1, class-wise precision/recall/F1, confusion matrices, AUROC and one-versus-rest average precision for both classes. NLL and Brier assess probabilistic quality and are not pure calibration errors (scikit-learn developers, n.d.).

Inherit the frozen metric definitions: Brier is mean `(p-y)^2`; top-label confidence is `max(p,1-p)` and correctness uses `p>=0.5`. ECE uses 15 equal-width bins over [0,1], left-closed/right-open except the last includes 1; empty bins contribute zero. Equal-mass sensitivity stably sorts confidence in canonical cache order and splits into 15 approximately equal groups, preserving the existing split-tie convention. This is top-label calibration, not a class-1 probability reliability curve.

Undefined metrics are JSON null/NA with support and a reason, not zero. Keep the natural test prevalence. The fixed 0.5 decision rule assigns exact ties to class 1. Check decisions derived from saved raw and scaled probabilities and logits for agreement. Positive temperature scaling should preserve decisions; any discrepancy requires numerical investigation, not a claim that recall improved. AUROC/AP are invariant in exact arithmetic; record numerical ties and any observed differences rather than imposing false exact equality after finite-precision saturation.

Do not infer minority-class success from aggregate accuracy or top-label ECE. Report both classes explicitly. Scaling cannot address a minority recall deficit while this decision rule remains fixed.

## 5. Length diagnostics and uncertainty

Keep strata 1–10, 11–20, 21–40 and 41+ events. For every stratum report total offers, distinct applications, counts by label, raw/scaled NLL, Brier and class recall. Include full-length histogram and empirical cumulative distribution, retaining the tail. The existing contract recorded 243 test offers in the 41+ stratum (94 class 0 and 149 class 1); verify this against the frozen cache rather than treating it as a new result. No length-dependent thresholds or calibrators are fitted.

Stratum plots show every seed or clearly labelled seed means, with support adjacent to the curves. Small strata are descriptive; their point estimates do not prove a causal sequence-length effect. Missing class support produces gaps, not interpolated recall values. Overall scores can hide short-prefix minority errors and long-prefix difficulties, so both breakdowns are required.

For NLL comparisons, use 2,000 paired application-cluster bootstrap draws with PCG64 seed 60601, a new predeclared engineering setting. Sort the unique test application IDs; each draw samples the same number of applications with replacement, carrying all offers of each selected application with their multiplicities. Reuse the identical draws for all arms, seeds, raw/scaled comparisons and the raw factorial interaction.

Recompute offer-mean NLL with the resampled offer-count denominator, then average per-seed differences. Use percentile 2.5 and 97.5 bounds with linear quantile interpolation. Do not resample offers independently, count seeds as new applications, or substitute a best-seed interval. Preserve the draw-index hash. Intervals are descriptive, conditional on these fitted checkpoints and the observed test cohort; they omit model-selection, training and temperature-estimation uncertainty and do not model temporal dependence between applications. Other metrics and length strata receive point estimates and support in v0.1, not invented uncertainty bands.

## 6. Execution and acceptance conditions

Run in the accepted dissertation environment, matching the recorded EXP-05 software, device, thread settings and source hashes. Additional evaluation modules must not change frozen model, adapter, metric or calibration code. Preflight verifies both accepted summaries, their acceptance records, manifests, checkpoint identities/epochs, preprocessing, cache hashes and per-run temperature files. Bind each temperature to its checkpoint hash and arm/seed. Synthetic runner evidence and formal dataset evidence remain separate.

Validate the new runner on synthetic fixtures before opening the real test cache for inference. Tests must cover all six checkpoint paths, wrong checkpoint/temperature/cache rejection, ID/label alignment, whole-application bootstrap multiplicities, independent small metric calculations, absent-class handling, unchanged inputs/models, exact tie decisions, and report exports. Test fitting/selection APIs must not be reachable from the evaluator.

Before test inference, replay validation and calibration logits against the saved parent outputs with maximum absolute logit error <=1e-6 and unchanged class decisions. Do not call the temperature solver. Once provenance and replay pass, load the test split through a dedicated evaluation path; do not weaken the original EXP-05 or CAL-01 partition-access guards.

For each checkpoint, produce one raw logit per test offer and derive scaled logits using its frozen T. Repeat test inference to verify deterministic replay in the same environment. Require finite logits, probabilities and defined score values, all expected IDs exactly once, and exact alignment across runs. Record file hashes before and after evaluation and neural state hashes before and after forward passes. GBDT checkpoint bytes must remain unchanged and repeat predictions must agree.

| Acceptance gate | Required evidence |
|---|---|
| Protocol binding | Version, protocol bytes and configuration hashed before test inference |
| Parent binding | Accepted EXP-05/CAL-01 lineage, matching checkpoints, temperatures and preprocessing |
| Cohort integrity | 8,613 offers, expected logical hash, unique offer IDs, split/application separation and full prefixes |
| Frozen execution | No training, calibration fitting, threshold fitting or new checkpoint selection; source/input/state invariants |
| Reproducibility | Parent replay within tolerance with unchanged decisions; deterministic repeated test inference |
| Coverage | All 18 pairs, both probability variants, dummy and every required metric/stratum |
| Numerical validity | Stable finite scores, decisions preserved, explicit undefined metrics and numerical-tie diagnostics |
| Reporting | Tables and plots recomputed from saved predictions; all exports complete and consistent |

PASS means valid completed evaluation, never IPAT superiority or guaranteed calibration improvement. Scientific outcomes are recorded separately. Any gate failure prevents PASS; retain partial outputs and write a failure record. After test predictions exist, any protocol, code or cohort correction must be documented with the previous attempt retained; never erase an unfavourable completed run or relabel a changed analysis as the original test.

## 7. Planned outputs and visualisations

The future implementation will save root `exp06_manifest.json`, `exp06_summary.json`, `exp06_acceptance.json`, metrics/comparison CSVs, a console log and an input/output checksum inventory. Each arm/seed folder will contain `test_predictions.npz` and `result.json`. Prediction arrays include offer/application IDs, labels, lengths, raw/scaled logits and probabilities, plus the checkpoint and temperature provenance. Persist bootstrap draw indices for reproducibility.

Generate plots from saved outputs, exporting PNG, PDF and SVG plus a self-contained offline `plots/report.html`:

1. Raw and calibrated NLL comparisons: every seed, means and unchanged dummy reference.
2. Paired NLL deltas and application-cluster intervals, with zero reference and clear direction.
3. Raw 2 × 2 component table and interaction values; no exaggerated colour-scale interpretation.
4. Within-model raw-to-scaled NLL/Brier/ECE changes, with every seed and both ECE conventions.
5. Raw/scaled top-label reliability and bin support, using prespecified display seed 1001.
6. Confusion matrices and class precision-recall curves for seed 1001; explain unchanged decisions and use test class prevalence reference lines.
7. Length-stratum losses and class recall, support counts, complete histogram and cumulative distribution.

Every panel names the test partition, version, metric direction and seed scope. Avoid a single crowded overlay of every arm, seed and calibration state. Display-seed figures illustrate the fixed choice; tables retain all seeds. There are no new training curves. Evaluation elapsed time may be logged operationally, but is not a new architecture-efficiency benchmark; the EXP-05 timing outliers remain unresolved.

## 8. Progression and change control

The next work item after protocol acceptance is an additive runner, synthetic verifier and report generator. Real test execution follows local verification. Downloads and extraction instructions will assume ZIP files in the project root, and outputs will default to a fresh timestamped project `results` directory. No silent redirection to AppData and no automatic dependency upgrades.

Clean-test evaluation will provide the clean reference for later robustness work. Keep the same individual temperatures under any later corruption. If a later robustness analysis uses these test applications, freeze corruption types, severities, randomisation and comparisons before inspecting their corruption results; choices motivated by these clean-test outcomes must be disclosed as exploratory. No clean-test result authorises retuning and then presenting the same holdout as fresh confirmation.

This stage does not prove end-to-end information preservation, retention interpretability, causal importance, robustness, or population-wide calibration. The preservation theorem remains scoped before LayerNorm. Report the limited retrospective cohort and all negative findings explicitly.

## 9. Evidence lineage and references

The accompanying configuration and evidence-binding file record SHA-256 values from the reviewed EXP-05/CAL-01 summaries and acceptance files. The 30 August Context Lock progress section is historical; its mathematics and the later accepted contracts govern this protocol. No historical evidence files have been changed.

All budgets, comparison hierarchy, bootstrap settings and output requirements here are project design choices specified before this test evaluation; they are not claimed to be prescribed by the following sources.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In D. Precup & Y. W. Teh (Eds.), *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321–1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

scikit-learn developers. (n.d.). *Probability calibration*. Retrieved September 20, 2026, from https://scikit-learn.org/stable/modules/calibration.html

The first source establishes temperature scaling as existing post-processing methodology; the second explains why NLL/Brier do not isolate calibration. Neither supplies evidence of IPAT performance on BPI17. The project retains its existing environment and metric definitions rather than adopting a newer library implementation.
