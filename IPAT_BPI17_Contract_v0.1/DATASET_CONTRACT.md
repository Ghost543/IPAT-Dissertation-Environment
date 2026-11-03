# IPAT BPI17 dataset contract v0.1

Specification date: 10 September 2026. Status: **specified for adapter implementation, not a processed-data freeze**. This document and `dataset_contract.json` define the next implementation. `evidence/contract_preflight.json` measures their consequences. No EXP-05 model has been trained by this package.

## Purpose and scope

A dataset contract makes each prediction example, target, observation boundary, feature, split and evaluation population explicit. Otherwise apparently matched models can answer different questions or benefit from future information. This public pilot asks whether the frozen IPAT representation and retention mechanisms remain trainable and competitive on a real event-sequence benchmark under matched reliability evaluation. It cannot establish universal information preservation or banking deployment suitability.

The unit is one offer, represented by its application history through that offer's `O_Create Offer` event, inclusive. Earlier offers within the same application can appear in this history. Events after this boundary cannot be inputs. Preserve original event order; this archive has strictly increasing timestamps within applications. Fail on future archives with ties/decreases rather than silently inventing a sorting policy.

The binary target is the **recorded `Accepted` attribute at offer creation**: false=0, true=1. This follows the target extraction in the [EBES authors' preprocessing](https://github.com/On-Point-RND/EBES/blob/main/preprocess/bpi_17.py). The reference main branch was reviewed on 10 September 2026, but its commit is not pinned; this is not exact EBES benchmark reproduction.

The audit found 19,334/42,995 disagreements between that attribute and the later presence of `O_Accepted` for the offer. This establishes non-equivalence, not which interpretation is correct or that records are erroneous. Do not describe our target as final granted loan, eventual customer acceptance, or the later event without additional source validation. Do not require a later terminal event for inclusion: that would change this target's cohort.

The archive is retrospective. Trace attributes and offer creation attributes may have been populated or revised later. Prefix restriction prevents using later event records, but cannot prove the historical availability of retrospectively stored values. Label maturity times are not certified. Therefore this contract supports a **retrospective chronological holdout**, not an as-of deployment simulation. The absence of future-record leakage is narrower than a proof of operational feature availability.

## Source identity and provenance

File: `BPI Challenge 2017.xes.gz`; 29,658,747 bytes. Uploaded archive SHA-256:

`183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249`

Complete measured source: 31,509 applications; 1,202,267 events; 42,995 offers. The previous audit includes raw integrity checks and independent counts. The new preflight independently parses the compressed XML and checks the pinned hash/counts. The archive is read directly; no extraction or source modification is required.

User-supplied attribution: Boudewijn van Dongen, Eindhoven University of Technology, first online 6 February 2017, under the linked [4TU General Terms of Use](https://doi.org/10.4121/resource:terms_of_use). Full terms, exact official repository version and a publisher-provided checksum remain unverified. A link or locally computed checksum does not independently establish these. The related older dataset DOI in the supplied description must not be cited as this archive's identifier. Preserve this distinction in the dissertation.

## Chronological partitions and related offers

Partition by each offer's creation timestamp, converted to UTC:

| Partition | Interval | Role |
|---|---|---|
| Train | before 1 July 2016 | Fit preprocessing and model parameters |
| Validation | 1 July to before 1 September 2016 | Select checkpoints and pre-specified model settings |
| Calibration | 1 September to before 20 October 2016 | Fit post-hoc temperature after model freezing |
| Test | 20 October 2016 onwards | Final held-out evaluation |

Exact-boundary timestamps enter the later interval. The first two boundaries are new calendar-based engineering choices, not mathematical necessities or inherited frozen decisions. October follows the reference code's test date. They were specified before examining model outcomes. Inspecting class counts here is a data-validity audit, not permission to optimise split boundaries against test performance.

If an application's offers occupy more than one interval, exclude **all** its offers from admitted partitions. Record every excluded offer, provisional partition and reason in `offer_split_manifest.csv`. Do not move future offers into training, split a shared application history between train and test, or duplicate offers to meet proportions. No label stratification or resampling is applied.

This rule trades coverage for strict grouping and time separation. The measured exclusions are not random: 512 applications/1,323 offers (3.08% of candidate offers) are excluded; their mean prefix length is 22.65 compared with 13.11 overall. Their false-label proportion is 38.93% compared with 29.91% overall. The contract therefore estimates performance on the admitted cohort. It must not be presented as covering all 42,995 offers without qualification. Keep the excluded cohort available for a separately specified future sensitivity study, not opportunistic test-set reinsertion.

## Features and preprocessing

| Type | Fields |
|---|---|
| Numerical (6) | RequestedAmount; FirstWithdrawalAmount; NumberOfTerms; MonthlyCost; CreditScore; OfferedAmount |
| Categorical (7) | Action; org:resource; concept:name; EventOrigin; lifecycle:transition; LoanGoal; ApplicationType |
| Temporal | Elapsed time, preceding-event gap and position through the frozen temporal interface |
| Masks | Explicit observed/missing values, separately from event validity/padding |

RequestedAmount, LoanGoal and ApplicationType are trace attributes broadcast within each prefix under the retrospective benchmark convention. Other attributes are taken only from their own event. Do not backward-fill creation values onto preceding events. Do not feed Accepted, Selected, EventID, OfferID or application/offer identifiers to any model; IDs are provenance/grouping metadata only. In particular, this exclusion applies to attributes on earlier offer events as well as the target event.

For observed numerical channel j, retain MATH-02:

\[\widetilde x_{tj}=(x_{tj}-\mu_j)/s_j,\qquad s_j>0.\]

Fit the population mean and standard deviation on observed **training** values only. Count each unique source event covered by admitted training prefixes once when fitting event attributes; count static trace values once per training application. This engineering convention avoids repeated overlapping prefixes implicitly reweighting preprocessing statistics. Loss weighting remains per offer. Deduplication for statistics uses source application ID and event index, never feature equality.

Use scale 1.0 for constant observed training channels; otherwise max(std, 1e-8). The scale floor is a configurable numerical default, not theory. Persist mean, std, actual scale, observation counts and constant-channel flags. Fail if a numerical channel has no observed training values; do not fit its statistics on held-out data. No clipping, winsorisation or coarse binning on the preservation path.

Missing numerical cells receive a zero placeholder **after** standardisation and an explicit missingness flag. Observed raw zero is standardised as an observed value with the opposite flag. The missingness convention must be mapped to the existing frozen interface and tested, not guessed. Padding uses a separate validity mask. Preserve CreditScore=0 as recorded: it appears in 27,735 offers, but sentinel semantics remain unknown. This does not assert that zero is a genuine credit score.

Categorical vocabularies are fitted on training data only, in deterministic lexical order. Reserve PAD=0, MISSING=1, UNK=2, then observed categories from 3. These are adapter defaults, not MATH necessities. Unknown held-out categories map to UNK, a deliberate loss of raw category identity; categorical preservation claims remain conditional on the encoded domain and distinct embedding rows. Store vocabularies and counts.

Compute elapsed time and preceding-event gap in seconds from source timestamps (first gap zero); use the existing frozen normalised time/gap/position encoder with training-only fitted scales. The exact temporal tensor convention must be checked against the installed frozen code before adapter acceptance. Absolute calendar timestamp is split/provenance metadata, not an added model input.

## Sequence-length skew and the preservation claim

Measured candidate prefix lengths: min 6, median 10, mean 13.1135, 95th percentile 30, 99th percentile 55, max 139. A right tail means most examples are short while a small number have much longer histories. It does not imply corrupt records, nor require a Gaussian length distribution. Length is a count of events, not a numerical channel to transform until the histogram looks symmetric.

Keep every admitted prefix intact. Dynamic right padding extends examples only to the maximum length in their batch; the frozen validity mask prevents padding from contributing to attention and aggregation. Shuffle offers with a recorded seed; each admitted offer appears once per epoch. Use identical offer order and batches across the four matched Transformer arms. Batch size is configurable and must accommodate the longest prefix. If memory is inadequate, adjust batch size/accumulation under a documented matched policy; do not silently truncate. Efficient attention kernels may alter actual memory use, so do not infer measured runtime from the length histogram.

Scaled-asinh acts on observed numerical values, not on the number of events:

\[g_j(x)=\bar\pi_{j0}x+\sum_{k>0}\bar\pi_{jk}\operatorname{asinh}(\alpha_k x)/\alpha_k.\]

Under frozen channel-wise routing independent of x, non-negative mixture weights and identity floor rho>0,

\[g'_j(x)=\bar\pi_{j0}+\sum_{k>0}\bar\pi_{jk}/\sqrt{1+\alpha_k^2x^2}\ge\rho.\]

This proves scalar injectivity in the mathematical model; it does not prove robustness to long histories or recovery of missing observations. MATH-03 retains `[g || h(g)]`, concatenates categorical/temporal/missingness representations and requires semi-orthogonal projection. A larger real-data schema may require a larger common model width: output dimension must be at least assembled input dimension for W^T W=I to be possible. Check this explicitly and apply one matched width to the four Transformer arms. Preservation still stops before LayerNorm.

## Label imbalance, objective and evaluation

Candidate labels: 30,136 true (70.09%), 12,859 false (29.91%). An always-true rule would achieve 70.09% accuracy on the full candidate set while recalling none of the false cases. This is an arithmetic illustration, not a held-out experimental result.

The pilot retains natural frequencies and unweighted per-offer cross-entropy:

\[L=-\frac1N\sum_{i=1}^N[y_i\log p_i+(1-y_i)\log(1-p_i)].\]

Here p_i is predicted probability of recorded Accepted=true. Use logits/stable cross-entropy in the model. This task-only objective follows MATH-06. NLL assesses probabilistic predictions, not only thresholded accuracy; see the [official log-loss definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html). A successful implementation is not required to demonstrate IPAT superiority.

No class weighting or oversampling is used initially. They can be legitimate for a separately defined cost-sensitive objective, but are not automatic corrections for this observed ratio. Derivation: if q=P(Y=1|X=x), weighted population loss is -w1*q*log(p)-w0*(1-q)*log(1-p). Setting its derivative to zero gives p*=w1*q/[w1*q+w0*(1-q)], generally different from q. Weighting therefore changes the target of ideal probability fitting unless corrected; it must not be introduced casually into a calibration-focused comparison. This is a mathematical derivation, not an empirical claim about trained IPAT.

Primary: clean held-out offer-mean NLL. Secondary: balanced accuracy, macro F1, per-class precision/recall, confusion matrix, AUROC, AP for each class, Brier, ECE, runtime/peak memory. Balanced accuracy is mean class recall, so the always-true illustration scores 0.5 when both classes are present; see [official balanced-accuracy documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).

For AP, evaluate true using p and false using 1-p; an all-tied constant score has AP equal to the evaluated class's prevalence. Do not interpret minority-class AP using the majority prevalence or a universal 0.5 baseline. AP is a ranking summary, not calibration; see [official AP documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Fit the constant-probability dummy using training prevalence only, then evaluate it on held-out data; see [DummyClassifier's prior strategy](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html).

Use a fixed class threshold of 0.5. Record undefined metrics as NA with counts/reasons. Any cost-based threshold selection belongs on validation data in a separate documented protocol. Retain MATH-08: top-label ECE with 15 equal-width bins, 15 equal-mass bins sensitivity; fit temperature once on clean calibration after model freezing. Keep test outcomes out of parameter, temperature and threshold fitting.

Report results in pre-specified length strata 1–10, 11–20, 21–40 and 41+ events, including counts by label, NLL and class recall. There are only 243 admitted test offers in the longest stratum (94 false/149 true); uncertainty will be greater there. Report this rather than overclaiming from aggregate accuracy. Length strata describe performance variation; they do not establish that length causally explains it.

Each offer contributes once to primary loss. Applications with more offers consequently have more weight. Also report application-macro NLL as a supplementary estimand. For data-sampling uncertainty, resample parent applications with all their offers; independent-offer resampling would ignore shared histories. Paired training seeds assess optimisation variability and are not substitutes for independent applications. Final seed counts/budgets belong in EXP-05 and are not invented here.

## Components, gates and next implementation

The supplied preflight consists of one streaming parser, application-level partition assignment, manifest writer and report/plot renderer. It does not change existing models or build tensors. Ten tests exercise boundaries, timezones, crossing applications, long prefixes, target definition, IDs, invalid timestamps/labels, joint counts and hash rejection. Full-source gates and comparison to the prior audit are recorded separately in the validation receipt.

The adapter should have four small responsibilities: (1) extract immutable prefix examples and provenance; (2) fit and persist training-only preprocessing; (3) encode the frozen numerical/categorical/time/missingness schema; (4) collate dynamic batches with the existing validity mask. Reuse installed interfaces, not a guessed replacement architecture.

Before DATA-03 is marked complete, demonstrate: exact source/manifest identity; disjoint applications; complete exclusion accounting; target/prefix provenance; no forbidden inputs or future/backfilled attributes; held-out perturbations cannot alter training statistics/vocabularies; zero/missing/padding distinction; correct 6/7 channel counts; temporal interface compatibility; feasible semi-orthogonal dimensions; padding invariance; matched shared initialisation; identity/gamma-zero forward and shared-gradient parity; deterministic artefact hashes; and user-local replication. Resolve or explicitly document source/availability limitations. The preflight PASS alone cannot satisfy the adapter gates.

Theoretical properties above come from the frozen handoff and explicit derivations. Dates, schema encoding IDs, statistic weighting and batching are design choices. Future benefits for minority labels, long histories, robustness or calibration are hypotheses. The current empirical evidence is descriptive source/preflight measurement only.
