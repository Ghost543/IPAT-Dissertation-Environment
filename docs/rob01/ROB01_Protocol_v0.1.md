# ROB-01 v0.1 — Controlled BPI17 robustness evaluation

Status: protocol accepted by the user on 23 September 2026; implementation supplied separately from user-local execution evidence. No real ROB-01 results exist in this package.

## Purpose and scope

Evaluate whether frozen full IPAT (T11) loses less predictive performance than its matched baseline T00 under incomplete observations. This is a subsequent, deliberately specified stress test on the already inspected EXP-06 held-out cohort. Design was fixed after clean-test review; it is not a new untouched confirmatory sample. Results do not establish universal robustness, causal event importance, prospective lending suitability or end-to-end information preservation.

Reuse the six frozen arms T00, T10, T01, T11, GRU and GBDT, training seeds 1001/1002/1003, EXP-05 preprocessing/checkpoints and individual CAL-01 temperatures. Reuse all 8,613 test offers from 6,356 applications. Predict the original recorded Accepted attribute of each inclusive offer-creation prefix. Never retrain, refit preprocessing, refit temperatures, select checkpoints or optimise thresholds. Threshold remains 0.5. Loss remains offer-weighted. GBDT is a summary-feature reference, not a matched architectural ablation; identical tree predictions across training seeds are not independent stochastic evidence.

## Registered corruption grid

Two separate corruption families are applied to the same clean observations; they are never composed.

- Additional numerical missingness probabilities: 0.10, 0.20, 0.40 of originally observed source values. Existing missingness is retained. RequestedAmount is one eligible static value per application, broadcast consistently; the other five numerical channels are event-level source observations. A selected observation receives missing_mask=true and raw finite-zero storage. An observed zero is eligible like any observed value, but is never automatically interpreted as missing. Categorical/time channels remain unchanged.
- Event deletion probabilities: 0.10, 0.25, 0.40 of eligible original historical events. Protect the original first event and EVERY O_Create Offer event, including earlier offer boundaries. Keep original source order and surviving elapsed timestamps. Rebuild valid prefix lengths and zero-based positions. Compute each surviving preceding-event gap by summing intervening original gaps in float64, then cast to the original model-input dtype; first gap remains zero. This respects the existing float32 interface and avoids cancellation from differencing rounded elapsed timestamps. No future events enter any prefix.

Three corruption replicates (1,2,3) are fixed independently of training seeds. Bernoulli selection probabilities are not exact quotas; requested and realised rates and denominators are always recorded. No-eligible cases remain in evaluation and have a null realised rate. No drop/replacement of rare cases. Existing missing values/protected events are excluded from the respective denominators. Numerical noise and temporal jitter are outside this version.

## Source consistency and reproducibility

The packed cache repeats source events in overlapping offer prefixes. Reconstruct the longest covered prefix per application and require every shorter prefix to agree exactly on all six raw fields: numerical values, missingness, categorical IDs, elapsed time, gap and position. Validate static RequestedAmount consistency and the numeric/categorical schema order. Source identity is (application ID, original position) in the validated complete prefix; IDs are audit metadata, not predictor features.

Selection is keyed SHA-256: UTF-8 JSON array ["ROB01-v0.1",70101,family,replicate,application_id,original_position,channel], compact separators and ASCII escaping; first eight digest bytes interpreted as unsigned big-endian / 2^64. RequestedAmount uses position=-1; deletion channel=-1. Selected iff uniform < requested probability. Severity, model arm and training seed never enter the key. This provides nested severity masks and identical corruptions across models and overlapping offers. Save each source plan (sorted application IDs, source offsets, selected masks and kept-event flags) and its hash.

Rebuild every offer prefix from this shared plan. Compute tree summaries anew from the same corrupted inputs. Batch padding remains the frozen collation operation, with validity separate from missingness. The 18 corruption conditions × 18 frozen predictors yield 324 corrupted model evaluations. Raw/scaled probabilities share each forward pass. Additional clean replays and repeated-inference checks are verification workload, not new experimental fits.

## Clean binding and leakage isolation

Bind the reviewed EXP-06 summary and acceptance by SHA-256. Verify EXP-06 manifest/checksums, parent EXP-05/CAL-01 evidence, checkpoint/temperature bindings, preprocessing, split disjointness and the accepted runtime using the existing frozen preflight. Verify zero corruption gives identical packed arrays for both families and all replicates. Replay all clean predictions using the same selected checkpoints; maximum absolute logit error <=1e-6 and identical decisions. Anchor deterioration to saved EXP-06 clean logits after successful replay, rather than replacing the clean reference with newly rounded results. Every corrupted inference is repeated exactly and verifies unchanged model state, checkpoint and inputs. All parent data/source files are checked again before PASS.

## Estimands and aggregation

Frozen MATH-07: oriented change Delta_rob = d_M (M_corrupt - M_clean). For NLL, Delta_rob = NLL_clean - NLL_corrupt. Negative means deterioration; positive means improvement. Primary comparison A = Delta_rob(T11) - Delta_rob(T00); positive favours T11. Primary uses raw probabilities. Scaled comparisons are secondary.

Compute paired losses per offer, within each training seed, corruption family, severity and corruption replicate. Average the three corruption replicates, then average severities within each family, then give each family equal weight. Only then summarise the three training seeds. The condition grid is not 324 independent replicates. Show absolute clean/corrupt NLL alongside changes. No superiority requirement in the execution gate and no non-inferiority/equivalence claim without a separate design.

Use 2,000 shared application-cluster bootstrap draws (PCG64 seed 70102), sampling applications with all their offers. Preserve matched clean/corrupt/model observations and offer weighting in each draw. Report descriptive percentile 95% intervals conditional on frozen models and these fixed corruption realisations. They do not estimate the full distribution of retraining or arbitrary corruption draws. Report individual training-seed effects; no significance test or multiple-comparison discovery claim. Store the resampling indices. No ratio metric is needed in this scoped implementation.

Calibration deterioration CD = ECE_corrupt - ECE_clean; positive means worse calibration. Fixed 15 equal-width top-label ECE bins over [0,1], equal-mass 15-bin sensitivity. Keep each clean-fitted temperature fixed under every condition. NLL and binary Brier mean((p-y)^2) are proper scores, not pure calibration errors. Also report accuracy, balanced accuracy, macro F1, classwise precision/recall, confusion matrices, AUROC and AP per class; undefined metrics remain explicit null. Preserve original clean length groups 1–10,11–20,21–40,41+ regardless of deletion. Record both original and corrupted lengths.

## Acceptance conditions

1. Local synthetic verifier passes and its code/dependency hashes match the real runner.
2. Parent lineage, reviewed clean evidence, source schema, cohort and checkpoint/temperature bindings pass.
3. Zero transformations are exact identities; all 18 clean checkpoint replays meet the fixed tolerance/decision checks.
4. Shared source masks are deterministic and nested; repeated prefixes and static features remain consistent.
5. Missingness/zero/padding and protected event/time rules pass synthetic tests; actual corruption and no-eligible cases are recorded.
6. All 324 corrupted evaluations are finite, aligned, repeated exactly and saved. No incomplete grid can pass.
7. Saved predictions reproduce global and original-length-stratified metrics; positive temperatures preserve fixed-threshold class decisions, with numeric failures reported rather than suppressed.
8. Reports/CSV/PNG/PDF/SVG/offline HTML are generated; every condition including adverse results remains in tables.
9. All frozen inputs and executable dependencies are unchanged at completion. Exceptions/interruptions leave failure evidence and never PASS.

## Outputs and interpretation

rob01_manifest.json; zero_replay.json; per-condition plan.npz/audit.json; per-model predictions.npz/result.json; bootstrap_draws.npz; metrics.csv; length_metrics.csv; corruption_counts.csv; comparisons.csv; rob01_summary.json; checksums.json; rob01_acceptance.json; console.txt; seven figure families and a self-contained report.html. Diagnostic plots summarise specified conditions; CSV/JSON retain the full grid.

Do not change design in response to emerging corruption results. Bugs affecting comparisons require an identified corrected runner and fresh output, preserving failed attempts. Clean-test evidence remains unchanged. ROB-01 results remain prospective until local acceptance and review. The preservation theorem remains scoped before LayerNorm; no corruption outcome proves recovery of deleted or missing information.
