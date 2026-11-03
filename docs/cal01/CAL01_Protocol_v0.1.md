# CAL-01: frozen-model temperature fitting v0.1

Date: 15 September 2026. Status: implementation prepared following the instruction to proceed after EXP-05 review. No BPI17 calibration results have been produced here. CAL-01 is a new milestone designation; it was not an existing numbered row in the tracker read for this task.

## Purpose and research link

EXP-05 established useful learning relative to a prevalence dummy but inconclusive incremental IPAT benefit. Its confidence curves looked broadly aligned on validation, with weak minority-class recall. CAL-01 now prepares the post-hoc calibration required by frozen MATH-06/MATH-08: can a simple adjustment align confidence more closely with observed outcomes after model weights are fixed?

Without this step, later reliability results would describe raw confidence alone. Applying the same calibration method to all comparator arms helps distinguish representational/attention differences from a conventional post-processing adjustment. Temperature scaling itself is established methodology, not a novel IPAT contribution. It changes confidence, not the learnt representation or attention mechanism.

CAL-01 fits calibrators and checks their engineering validity. Its fitting-sample scores cannot establish out-of-sample calibration benefit. The reserved test partition remains for a subsequent separately specified clean evaluation and robustness study. Model configuration and checkpoint selection are closed for this evaluation branch before calibration fitting. No model is selected or retuned using calibration outcomes; a later architecture revision is a separate development branch.

## Mathematical operation

For a frozen binary logit l, define p_T = sigmoid(l/T), T > 0. Fit one temperature per arm and seed using equal-offer mean calibration NLL:

    L(T) = mean_i log(1 + exp(-(2*y_i - 1)*l_i/T)).

Use stable logaddexp and float64 for this detached calculation. T > 1 softens confidence; T < 1 sharpens it; T = 1 is the raw predictor. Example: a raw p=0.9 becomes p=0.75 when T=2, because sigmoid(log(9)/2)=0.75. This is an illustration, not an observed BPI17 result.

Positive scaling preserves the sign and ordering of logits. At the fixed 0.5 rule, logit zero still predicts class 1. Accuracy, class recall and confusion matrices should be unchanged. AUROC/AP are also invariant under strictly monotone scaling in exact arithmetic, although converting extreme logits to probabilities can create numerical ties. CAL-01 does not claim improved discrimination from scaling, and explicitly checks actual probability-threshold decisions for numerical discrepancies. It cannot correct the minority-class recall limitation identified in EXP-05.

The solver uses inverse temperature beta=1/T. The NLL is convex in beta, with derivative mean(l*(sigmoid(beta*l)-y)). Bisection of this monotone derivative determines an interior solution. The numerical bounds T in [0.05,20] are predeclared engineering settings, not theoretical limits. Search runs at most 100 iterations with beta-interval tolerance 1e-12. Both bounds and T=1 are explicit candidate solutions. Fit NLL may not exceed raw fit NLL by more than 1e-10. A bound solution is reported as BOUNDARY; all-zero logits are UNIDENTIFIABLE and retain T=1. Neither condition is silently treated as an interior optimum or fixed by widening bounds after seeing test results. They create NEEDS_REVIEW readiness status while preserving all outputs.

## Frozen inputs and scope

Reuse all 18 EXP-05 selected predictors: T00, T10, T01, T11, GRU and GBDT, each with seeds 1001,1002,1003. Keep duplicate GBDT results and disclose their dependence. Fit each independently, with no best-arm/seed selection and no new neural training. The existing training-prevalence dummy remains a fixed reference from EXP-05; it is not converted into a calibration-prevalence estimator here.

Parent summary SHA-256: ae291bc6688d0c85a287a76a7c448c54c1f6f76ee1ac55aecaa0c1c63c49b2e3.
Parent configuration digest: 26ba05156779b11cb03f739c40a3471fd6f32c3db4d0549fda4f8b2701456edf.

| Partition | CAL-01 role |
| --- | --- |
| Train | No loading or fitting; existing train-only preprocessing reused |
| Validation: 7,871 offers | Checkpoint replay against saved EXP-05 predictions; no temperature fitting |
| Calibration: 6,117 offers | Generate frozen logits and fit T; report clearly labelled fitting diagnostics |
| Test: 8,613 offers | No predictive loading/evaluation |

Use the frozen cache hashes and preprocessing logical hash from the parent manifest. Preserve natural class frequencies, complete prefix lengths, label semantics, application separation and equal-offer weighting. The recorded Accepted attribute is not certified final loan outcome. No class weighting, resampling, new thresholds, corruption or calibration methods are introduced.

Restore the parent software/device/thread settings and require its environment and recorded source hashes to agree. New CAL-01 modules are separate additions. A changed environment/source stops execution rather than silently rebinding evidence. Neural checkpoints must have the parent manifest identity and selected epoch. Saved validation IDs, labels, prefix lengths and NLL must match the reviewed record. Replayed validation logits must be within absolute 1e-6 with zero relative tolerance and identical class decisions; report the actual error. This is a numerical replay gate, not a changed predictive metric. Repeat calibration inference exactly. Hash input checkpoints before and after processing and check neural state tensors remain unchanged. Tree checkpoint files are read-only and must come from the user's own verified parent run.

## Acceptance conditions, defined before local execution

1. The reviewed parent acceptance, summary and configuration are bound; all 18 parent results agree with the summary.
2. Preprocessing and validation/calibration logical hashes match. Calibration has both labels and unique offers; no validation/calibration application overlap.
3. All parent checkpoints reproduce validation predictions within the recorded gate; calibration replay is exact.
4. Model weights and checkpoint files remain unchanged. No network optimiser runs.
5. Every fit has finite positive T and finite logits/NLL, preserves decisions, and does not worsen its fitting objective beyond numerical tolerance.
6. All 18 fits and their solver statuses are reported. Boundary/unidentifiable solutions require review before later test evaluation.
7. Tables and PNG/PDF/SVG/HTML reports derive from saved values. A complete summary and acceptance file are written only after all fits and exports finish.
8. Independent local synthetic verification succeeds, including numerical fixtures, cache/checkpoint mismatch rejection, all six checkpoint paths across three seeds, replay, immutability and plots.

No decrease in ECE/Brier is required to pass execution gates: NLL is the fitting objective, and these other diagnostics can worsen. A reduction in calibration-fitting NLL is expected from optimisation and is not independent scientific confirmation.

## Outputs and interpretation

Per arm/seed: temperature.json with raw/scaled metrics, solver status and provenance; calibration_predictions.npz with raw/scaled logits, labels, offer/application IDs and lengths. Root: cal01_manifest.json, cal01_summary.json, cal01_acceptance.json. Use a new timestamped output directory; never overwrite EXP-05 or a failed attempt. This small post-processing runner starts a new attempt after interruption; it does not retrain parent models.

Five figure families are generated automatically: all seed temperatures; raw-to-scaled NLL/Brier/fixed-bin ECE/equal-mass ECE changes; before/after reliability; bin support; temperature objective profiles. Reliability, bin support and profiles use prespecified seed 1001, while temperature/metric plots show all seeds. Each figure states calibration FIT scope. Top-label ECE retains 15 fixed equal-width bins over [0,1], with 15 equal-mass groups as sensitivity, inherited from the frozen metric module. Length-specific raw/scaled metrics are retained in JSON. No uncertainty interval from reusing the fitting sample is promoted as independent calibration evidence.

Do not call reduced fit ECE a generalisation result, change model selection after this step, refit T on corrupted/test data, interpret T as retained information, or claim improved class recall from positive temperature scaling. Freeze temperatures for the subsequent evaluation. Timing outliers from EXP-05 remain unexplained; CAL-01 makes no efficiency claim.

## Evidence status and literature

Development: numerical and report checks are recorded separately. The complete PyTorch integration verifier must run in the user's existing dissertation environment before real CAL-01 execution. No synthetic fixture is BPI17 evidence. Local completion should be reviewed using cal01_runner_acceptance.json, fixture_tests.txt, cal01_acceptance.json and cal01_summary.json.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In D. Precup & Y. W. Teh (Eds.), Proceedings of the 34th International Conference on Machine Learning (Vol. 70, pp. 1321–1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html

This source supports temperature scaling as a post-processing calibration method. The BPI17 split, numerical bounds, provenance checks and CAL-01 sequencing are project design choices. Frozen MATH-06/MATH-08 and the accepted dataset contract govern the scientific restrictions.
