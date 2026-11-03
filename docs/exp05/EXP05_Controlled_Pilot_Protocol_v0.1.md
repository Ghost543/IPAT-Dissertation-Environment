# EXP-05 controlled pilot protocol v0.1

Date: 11 September 2026. Status: **proposed protocol for review; no pilot training or results**. New numerical budgets and reference-model choices below are engineering proposals. Frozen MATH-01–08, the BPI17 contract and validated adapter remain authoritative. This document supplements the historical Context Lock; it does not rewrite it or retrospectively claim approval of new decisions.

## 1. Scientific purpose and scope

EXP-05 asks whether the validated IPAT components can learn from a real event-sequence dataset under a controlled, affordable training budget, and what their individual and joint contributions look like. A passing adapter establishes faithful inputs and interface behaviour; it does not establish useful prediction. A pilot bridges that gap before expensive tuning or broader reliability studies.

The primary comparison is a 2 × 2 architectural ablation. Without the component-only arms, a difference between a standard Transformer and full IPAT could not be attributed separately to numerical representation or retention-aware attention. GRU and gradient-boosted tree references test whether the Transformer family is useful relative to simpler alternatives, but do not isolate IPAT mechanisms.

The task predicts the **recorded Accepted attribute of an offer from its inclusive creation prefix**. It is a retrospective benchmark. It is not certified prediction of later O_Accepted, final loan outcome or prospective lending decisions. All target, exclusion and feature-availability qualifications in the contract apply.

Hypotheses are prospective: (H1) models train stably; (H2) learned models improve validation NLL over a training-prevalence dummy; (H3) representation and/or retention yield favourable paired differences against the matched Transformer; (H4) their joint effect may depart from the sum of individual effects. H3/H4 may fail. Failure is reportable scientific evidence, not a reason to change the success criterion after seeing results.

## 2. Evidence boundary and partition roles

| Partition | Offers | Role in EXP-05 |
|---|---:|---|
| Train | 19,071 | Fit preprocessing already frozen; optimise model parameters |
| Validation | 7,871 | Select checkpoints and report exploratory pilot comparisons |
| Calibration | 6,117 | Reserved; no temperature fitting in this pilot |
| Test | 8,613 | Reserved; no predictive evaluation or model selection in this pilot |

These counts and logical hashes come from locally accepted adapter evidence. The contract excludes every application whose offer dates span multiple partitions. Retain that policy and all 41,672 admitted offers. No new stratification, resampling, clipping or prefix truncation.

**Protocol choice:** clean validation offer-mean NLL is the pilot's primary reported score. Because the same validation set selects checkpoints, it is a development estimate with selection optimism, not an unbiased final generalisation estimate. The later frozen study will use the reserved test set for its primary held-out result. Audit statistics for calibration/test have already been inspected; do not claim those datasets are wholly unseen. Their predictive outcomes have not been used for model selection here.

This choice avoids consuming a test evaluation and then tuning against it during pilot revision. Train-only fitting and separation of selection from test assessment follow the general leakage principles in the scikit-learn guidance [1].

## 3. Frozen mathematics and its computational meaning

For observed numerical channel j at event t, use training-only mu_j and positive s_j:

\[
\widetilde x_{tj}=(x_{tj}-\mu_j)/s_j,\quad
p_{jk}=\operatorname{softmax}(a_j)_k,
\]
\[
\bar\pi_{j0}=\rho+(1-\rho)p_{j0},\qquad
\bar\pi_{jk}=(1-\rho)p_{jk}\;(k>0),
\]
\[
\psi_{j0}(x)=x,\quad\psi_{jk}(x)=\operatorname{asinh}(\alpha_k x)/\alpha_k,
\quad g_j(x)=\sum_k\bar\pi_{jk}\psi_{jk}(x).
\]

The logits a_j are learned per channel, not functions of the current scalar. Consequently the frozen mixture has g'_j(x) >= rho > 0 in real arithmetic. The identity floor prevents the scalar path from becoming arbitrarily flat. Positive scaled-asinh candidates provide smooth compression while maintaining order. Their scales and rho are design settings, not a theorem of predictive superiority.

The explicit numerical coordinate and assembled event interface remain:

\[
z^{num}_{tj}=[g_j(\widetilde x_{tj})\Vert h_j(g_j(\widetilde x_{tj}))],\quad
u_t=[z^{num}_t\Vert z^{cat}_t\Vert z^{time}_t\Vert m_t],
\]
\[
z_t=v_t(W_z u_t+b_z),\qquad W_z^\top W_z=I.
\]

Here v_t is validity and m_t represents numerical missingness; neither is a target. For a valid event, the projection admits u_t=W_z^T(z_t-b_z). Preservation is scoped to the assembled representation before LayerNorm. Missing values, UNK categories and potentially coincident learned categorical embeddings do not support universal raw-input recoverability.

Retention and attention retain MATH-04/05:

\[
r_s=\sigma(w_r^\top z_s+b_r),\quad
\widetilde\lambda_s=\log r_s-\operatorname{mean}_{u\in valid}\log r_u,
\]
\[
P_{is}=\operatorname{softmax}_s\left(q_i^\top k_s/\sqrt{d_h}
+\gamma\widetilde\lambda_s+M_s\right).
\]

Compute log r with stable logsigmoid. Gamma is fixed, not learned. Retention is a relative source-event prior, shared across heads and independent of query in v0.1. It is not a calibrated probability of information preservation or a causal explanation. Gamma zero/equal retention recover standard attention under matched inputs and weights; trained trajectories of different arms need not remain equal.

For offer i with label y_i and model logit l_i, the sole gradient objective is:

\[
\mathcal L_{task}=\frac1N\sum_i[\operatorname{softplus}(l_i)-y_i l_i].
\]

This is binary NLL in nats. Minibatches approximate the equal-offer objective, with each admitted training offer included once per epoch. Report full-partition loss by summing per-offer losses then dividing by N, not averaging unequal batch means. AdamW decay remains an optimiser setting, not an added term in this displayed objective. Calibration/retention/MI/CKA diagnostics do not backpropagate into the model. Raw deterministic continuous I(X;Z) is not a primary score. No deferred auxiliary loss is enabled.

## 4. Model arms and controlled comparisons

| Arm | Numerical representation | Sequence mechanism | Purpose |
|---|---|---|---|
| T00 | Frozen baseline fixed-identity encoder | Standard attention | Matched control |
| T10 | Adaptive identity + scaled-asinh | Standard attention | Representation contribution |
| T01 | Fixed-identity encoder | Retention attention, gamma=1 | Attention contribution |
| T11 | Adaptive identity + scaled-asinh | Retention attention, gamma=1 | Full IPAT |
| GRU | Same fixed-identity event assembler and projection | One unidirectional GRU layer | Recurrent reference |
| GBDT | Fixed prefix summaries from allowed fields | Histogram gradient boosting | Tabular reference |
| Dummy | No event features | Constant train prevalence | Minimal probabilistic reference |

The fixed-identity encoder retains its existing learned expansion; 'fixed' refers to the scalar mapping, not freezing all embedding parameters. All four Transformer arms reuse the validated BPI17 classes, categorical/time/mask inputs, projection constraint, pre-norm blocks and masked-mean pooling. Only the two designated mechanisms differ. Shared parameter tensors must be equal at initialisation, verified by names and values rather than trusting a seed alone. Arm-specific parameters are recorded. Parameter totals and runtime may differ: this is an architectural intervention with a matched backbone, not a claim of exactly equal capacity or compute.

Define per-seed differences using validation NLL (lower is better):

\[
\Delta_R=L_{10}-L_{00},\quad\Delta_A=L_{01}-L_{00},\quad
\Delta_F=L_{11}-L_{00},\quad
\mathcal I=L_{11}-L_{10}-L_{01}+L_{00}.
\]

Negative delta is favourable. Negative interaction I means the combined NLL reduction exceeds the additive expectation on this metric scale. It is a descriptive interaction of trained pipelines under this budget, not proof of causal synergy or universal novelty. For example, hypothetical values 0.60, 0.58, 0.57, 0.54 produce I=-0.01; these are an arithmetic illustration, not experimental results.

GRU uses projected input width 60, hidden width 60, one layer, zero recurrent dropout, the last valid hidden state, LayerNorm and a binary head. Packed sequences or equivalent valid-state selection must prevent padding from changing predictions. It shares raw feature access, but not the Transformer architecture, pooling or parameter count.

GBDT uses HistGradientBoostingClassifier, log_loss, 200 iterations, learning_rate 0.05, max_leaf_nodes 15, min_samples_leaf 20, l2_regularization 0, max_bins 255, categorical_features=None, early_stopping=False and class_weight=None. These are explicit modest pilot settings. The estimator's automatic early stopping can otherwise create an internal split; disable it to preserve our declared partition policy [2]. Use all 200 iterations, without choosing a tree count from validation.

GBDT summaries, fixed before labels are inspected for modelling: each numeric channel's last observed value, mean, population standard deviation, minimum, maximum and observed fraction over the prefix; all-unobserved value statistics are NaN and fraction zero. Each categorical field contributes normalised frequencies for all training-vocabulary IDs including MISSING/UNK but excluding PAD, plus a one-hot final-event category over the same vocabulary. Temporal summaries are prefix length, elapsed duration, and mean/std/min/max gap (including the first zero gap). No identifiers or forbidden fields enter summaries. This compresses order, so the tree comparison cannot isolate attention. It is permitted only for the external reference, not the IPAT preservation path. Test summary provenance and missing-value treatment before fitting.

Dummy probability is q_train=13234/19071, fixed across partitions. At threshold 0.5 it always predicts the majority class; this makes clear why accuracy alone is insufficient. Compute scores on validation, never fit its probability to validation prevalence.

## 5. Proposed bounded training budget

All choices in this section are engineering defaults to review and then lock before the first predictive run; none is claimed optimal.

| Setting | Proposed value / rule |
|---|---|
| Seeds | 1001, 1002, 1003; all reported |
| Neural epochs | Exactly 20 complete epochs; no performance-based early termination |
| Effective batch | 64 offers; final incomplete batch retained |
| Neural optimiser | AdamW, learning rate 0.001, betas (0.9,0.999), epsilon 1e-8, decay 0.01 on all trainable parameters |
| Schedule | Constant learning rate; no scheduler or warm-up |
| Gradient clipping | Global norm 1.0; record unclipped norm and clipped fraction |
| Precision | Float32; no AMP; explicitly disable TF32 on CUDA for the paired study |
| Backbone | Two layers, four heads, d_in=59, d_model=60, feed-forward width 240 |
| Dropout | 0.10 for all Transformer arms; existing locations retained |
| Adaptive settings | rho=0.25, alpha=(0.25,1,4); existing zero routing-logit initialisation |
| Embeddings | Numeric learned expansion width 2; categorical width 4 per field |
| Retention | Fixed gamma=1 for T01/T11; existing scorer initialisation |
| Validation | Full validation each epoch in eval mode; no gradients |
| Checkpoint | Lowest full validation NLL among epochs 1–20; exact tie selects earlier epoch; also report epoch-20 results |
| Workers | num_workers=0 initially for the Windows runner |
| Device | Resolve once before all runs, record hardware/software; same device and precision for paired neural runs |

There are 298 batches per epoch at batch 64, giving 5,960 updates per neural run. Four Transformer arms × three seeds produce 12 fits; GRU adds three fits. Total neural budget is 89,400 updates. This is arithmetic from the proposed budget, not a runtime measurement. GBDT uses one fit per seed; if predictions coincide, disclose that and do not treat duplicates as independent stochastic evidence. Dummy requires one prevalence calculation.

Use an explicit precomputed offer permutation per seed/epoch, shared across neural arms. Use separate RNG streams for model initialisation, data order and training dropout; reset the training stream immediately before each paired Transformer run. Preserve same call order at shared dropout sites. Save shared initial tensors or verify exact shared hashes for each seed. Evaluation/plotting must not advance training RNG state.

Before the experiment, benchmark 50 train-only updates plus a longest-prefix batch at batch 64 with a separate disposable seed 999; report timing/memory, then discard those weights. This preflight selects only feasible device/microbatch settings, not model performance. If memory requires smaller microbatches, accumulate sum-reduced losses divided by the actual effective-batch offer count and clip/step once per effective batch. Apply the resolved policy to all paired arms. Do not silently truncate long prefixes, skip offers, drop a seed, switch precision or give a winning arm extra epochs. If the budget is infeasible, revise and version the protocol before predictive validation results are opened.

For interruption recovery, persist model, optimiser, all RNG states, epoch, batch offset, data permutation and protocol/source hashes. A runtime failure is recorded; it is not a poor-performance seed to remove. Bugs affecting comparisons require a new runner version and rerun of all affected paired arms.

## 6. Metrics, imbalance, length and uncertainty

Primary: uncalibrated validation offer-mean NLL. Secondary: balanced accuracy, macro F1, classwise precision/recall, confusion matrix, AUROC, AP for each class, Brier, ECE, time and memory. Decision is yhat=1 if probability >=0.5; no threshold search. Binary Brier convention is mean (p-y)^2, not the doubled two-column sum.

Class weighting/SMOTE/resampling are disabled by the frozen contract. They change training exposure or loss weighting and would confound this first architectural comparison. Natural imbalance is addressed through evaluation: show per-class support and scores, compare to the prevalence dummy, and never interpret a high accuracy in isolation.

Use length strata 1–10, 11–20, 21–40 and 41+, with offer and application counts and class counts in every stratum. Report NLL, balanced accuracy and class recall. Undefined metrics are JSON null with a reason and displayed NA; do not silently substitute zero. A one-class stratum has no two-class AUROC or balanced accuracy under our convention; class recall is NA where that class has no support. Precision is NA if no instances are predicted for that class. For macro F1, compute both fixed labels; absent-class/undefined cases must be explicit rather than library-default dependent.

Sequence-length skew stays intact. Long sequences require more padding and quadratic attention work, and may have different class composition. Performance versus length is descriptive, not proof that length caused the difference. Report both class counts and per-class behaviour alongside length plots. Do not normalise sequence lengths into an artificial Gaussian distribution or drop the tail.

Primary weighting remains per offer. Supplement application-macro NLL: average losses within each application, then average applications. This checks whether applications producing many offers dominate the primary result without changing the training objective.

Report all seed values, mean, sample standard deviation and paired deltas. Three seeds are a feasibility compromise, not a power calculation; do not run a confirmatory significance claim. General benchmark variability motivates reporting repetitions rather than selecting a favourable run [3]. No default p-values or best-seed headline.

Optional uncertainty display is fixed here as 2,000 paired application-cluster bootstrap draws with RNG seed 50501. Resample validation applications with replacement, carrying all their offers and using the same draw across arms/seeds; recompute offer-weighted NLL and seed-mean deltas. Show percentile 2.5/97.5 bounds only as descriptive intervals conditional on these trained checkpoints and the observed holdout. They exclude full training/model-selection uncertainty and temporal dependence between applications; they are not a substitute for more seeds or a final test set.

MATH-08 ECE uses top-label confidence max(p,1-p) and correctness; 15 fixed equal-width bins over [0,1], with left-closed/right-open bins except the final bin includes 1. Empty bins contribute zero. Equal-mass 15-bin sensitivity uses stable confidence sorting, then 15 approximately equal-size groups; document split ties. Do not confuse this with class-1 probability calibration. All ECE is detached; NLL and Brier assess probabilistic quality, not pure calibration error.

No temperature fitting in EXP-05. Later, after model freezing, fit temperature on clean calibration data only and retain it for test/corruption evaluation, consistent with MATH-08. Temperature scaling is a post-processing method discussed by Guo et al. [4]; it does not justify tuning architecture on calibration results. Robustness, MI estimation and substantive retention interventions remain separate milestones. A mean retention weight is not evidence of event usefulness.

## 7. Acceptance and progression gates — before implementation

| Gate | Required evidence |
|---|---|
| Provenance | Frozen source/contract/manifest/logical hashes checked; local adapter acceptance retained; DATA-02 terms/version resolved before training |
| Protocol | Exact budget, seeds, software versions, reference features and device policy recorded before runs |
| Leakage isolation | Runner fits on train, selects only on validation; calibration/test predictive paths disabled; negative tests detect access violations |
| Matching | Equal shared initial tensors; paired offer schedules; limiting-case forward/shared-gradient checks; padding invariance |
| Metrics | Independent small hand-computed NLL/Brier/threshold/binning checks; correct offer-weighting; undefined-metric fixtures; paired cluster resampling preserves applications |
| Execution | Every planned seed/arm completes its declared budget, or failures explicitly prevent COMPLETE status; no non-finite loss/gradient/predictions |
| Checkpoints | Selection verified from recorded validation histories; optimiser/RNG resume equivalence checked in controlled fixture; checkpoints reload to reproduce saved predictions |
| Audit trail | Every admitted validation offer has one prediction per selected model/seed, correct ID/label alignment; hashes/config/device/elapsed/memory recorded |
| Outputs | Tables and plots derive from saved results and agree numerically; no synthetic placeholders represented as results |

Scientific outcome is separate from these gates. A valid negative comparison can complete EXP-05. Describe learning as useful only if it improves over the training-prevalence dummy on validation NLL; report which arms and seeds do so. If no neural model improves, investigate optimisation/task accessibility before expanding. If full IPAT loses to baseline while references learn, record the negative finding and inspect paired components; do not redefine the metric or call the gate a numerical failure. 'Competitive' remains a descriptive comparison of NLL deltas, variability and resource costs, not an unstated non-inferiority threshold.

No automatic broad HPO, extra datasets or robustness claims follow from a successful command. Review underfitting, overfitting, minority-class collapse, length-stratum support, UNK behaviour and cost before deciding the next milestone.

## 8. Planned implementation and output contract

After conceptual/protocol review, add a small EXP-05 runner using the existing adapter and four model classes. Separate split loading and hash validation, paired training, GRU/reference summarisation, metric evaluation, and plotting. Avoid editing frozen model modules. New reference models and metric helpers require their own contract tests. Proposed files are src/ipat/experiments/exp05.py, src/ipat/experiments/exp05_metrics.py, scripts/run_exp05_pilot.py, scripts/plot_exp05_pilot.py, tests/test_exp05.py and configs/exp05_pilot_v0.1.json. These names describe future work; they are not runnable deliverables in this protocol package.

Planned saved outputs: protocol/config hashes; environment and code manifest; per-seed initial/shared hashes; epoch histories; selected and final checkpoints; per-offer validation predictions; aggregate JSON/CSV metrics; acceptance JSON; original console log; runtime/memory measurements. Outputs must never overwrite an earlier run.

Plots generated by experiment code, with PNG/PDF and an offline HTML report:

1. Train and validation NLL versus epoch, each seed visible; selected checkpoints marked. Distinguish optimisation-time train loss from full eval-mode train NLL.
2. Paired NLL comparison/delta plot with dummy reference and all seeds.
3. The 2 × 2 representation/attention table and interaction values.
4. Per-class recall, precision-recall curves with prevalence baselines, confusion matrices.
5. Length-stratum NLL/recall with class support; show both histogram and cumulative length distribution to expose the tail.
6. Reliability diagrams with bin counts and uncalibrated NLL/Brier/ECE.
7. Runtime/peak memory and model size, identifying tree/GRU reference differences.
8. Detached routing/retention trajectories labelled diagnostic, without implying information percentage or causality.

Every figure must include partition, metric direction, version and seed scope. No smoothing that hides failed runs; no test curves in pilot outputs. No artificial data should appear in a research-results panel.

## 9. Current evidence, source review and change control

Completed evidence: local adapter 20/20 tests and 17/17 integration checks; all four reported logical cache hashes agree with development. These are prerequisite engineering results only. No EXP-05 model has been trained or evaluated in this task.

Outstanding DATA-02 item: the official BPI2017 repository page returned HTTP 403 during this protocol check, and the terms DOI could not be retrieved. This attempt does not verify or invalidate permission. Preserve the source-review gate; do not assert a licence or exact repository version. A full accessible official record/terms document is needed to close it. Retrospective feature availability and CreditScore-zero meaning remain scoped limitations even after terms are verified.

Protocol changes receive a new version and rationale before affected runs. The frozen contract's historical status strings remain unchanged bytes; the separate local validation record documents the later freeze. Do not rewrite the old evidence to make hashes agree. The Context Lock's old progress section is historical; the current tracker and later validation decisions govern progress.

## 10. Credible source basis and claim attribution

The equations and IPAT-specific restrictions above come from the frozen project Math Spec and are explicitly our design/derivation, not attributed to these papers. Numerical budgets, reference-feature definitions and pilot partition use are new protocol proposals, not literature-established necessities.

[1] scikit-learn developers. (n.d.). *Common pitfalls and recommended practices*. Retrieved September 11, 2026, from https://scikit-learn.org/stable/common_pitfalls.html#data-leakage . Supports train-only preprocessing and leakage prevention, not our exact split dates.

[2] scikit-learn developers. (n.d.). *HistGradientBoostingClassifier*. Retrieved September 11, 2026, from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html . Supports estimator behaviour, missing-value handling and automatic early-stopping behaviour; our chosen budget is independent.

[3] Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Sepah, N., Raff, E., Madan, K., Voleti, V., Kahou, S. E., Michalski, V., Serdyuk, D., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). *Accounting for variance in machine learning benchmarks* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2103.03098 . Supports attention to benchmark variability; three seeds are our feasibility choice.

[4] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321–1330). Proceedings of Machine Learning Research. https://proceedings.mlr.press/v70/guo17a.html . Supports temperature scaling as post-processing; makes no result claim about IPAT or BPI17.
