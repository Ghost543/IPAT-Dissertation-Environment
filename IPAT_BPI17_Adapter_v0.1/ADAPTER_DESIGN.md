# BPI17 schema adapter v0.1

The adapter turns the locally validated offer-prefix contract into model-ready tensors. Its scientific purpose is to make the real-data comparison faithful to the frozen IPAT definition: identical examples and preprocessing must reach each model variant, and the only intended variant differences remain numerical adaptation and retention-aware attention. This is data/implementation validation, not EXP-05 predictive evidence.

## Connection to the frozen mathematics

MATH-01 separates numerical, categorical, temporal and missingness information. MATH-02 uses training-only statistics:

\[\widetilde x_{tj}=(x_{tj}-\mu_j)/s_j,\quad s_j>0.\]

The existing `FixedIdentityNumericalEncoder` and `AdaptiveNumericalEncoder` already perform this operation. Therefore the adapter emits **raw observed values**, together with fitted means/scales and masks. Emitting standardised values here would incorrectly apply standardisation twice. Missing raw placeholders are finite zero; the encoders replace the resulting standardised coordinate with zero when missing or padded.

An illustrative channel with mean 100 and scale 20 encodes observed raw zero as -5, with missing=false. A missing value becomes standardised zero, with missing=true. A padded event has valid=false. These are three distinct states, even if storage placeholders happen to be zero. This illustration is not a measured BPI17 statistic.

The adaptive module retains the frozen identity-plus-scaled-asinh mixture and identity floor. Neither fitting nor batching adds clipping, winsorisation or a nonlinear preprocessing stage. The scalar derivative guarantee applies to observed standardised inputs under the frozen mapping; it cannot recover observations absent from the source.

MATH-03 assembles

\[u_t=[z_t^{num}\Vert z_t^{cat}\Vert z_t^{time}\Vert m_t],\qquad z_t=p_t(W_zu_t+b_z),\quad W_z^TW_z=I.\]

Here p_t is event validity, separate from the numerical missingness vector m_t. The new assembler retains the existing numerical/temporal modules and substitutes seven explicit categorical embeddings for the two synthetic categorical fields. Each model uses the same schema and embedding sizes.

At the existing defaults: numerical width=6*(1+2)=18; categorical width=7*4=28; temporal width=3+4=7; missingness width=6. Hence assembled width=59. The inherited rule rounds up to the next multiple of four heads, giving model width=60. Output width at least input width is necessary for this semi-orthogonal projection; four-dimensional categorical embeddings and four heads are engineering defaults, configurable together across matched variants.

The actual frozen temporal implementation computes elapsed-time and gap standardisation, explicit relative position, and sine features:

\[\tau_t=(e_t-\mu_e)/s_e,\quad d_t=(\Delta_t-\mu_\Delta)/s_\Delta,\quad q_t=t/\max(L-1,1).\]

\[z_t^{time}=[\tau_t,d_t,q_t,\sin(2^0\tau_t),\ldots,\sin(2^{K-1}\tau_t)].\]

The adapter supplies elapsed seconds from the application's first event, preceding-event gap seconds (first gap zero), and zero-based t. The temporal encoder computes the remaining operations, using training-fitted scales. L is each offer prefix's actual valid length, not the padded batch length. This is retrospective classification at the offer boundary: the full prefix length is known at prediction time.

## Four implementation components

1. `ipat.data.bpi17`: verifies the pinned archive, contract and split-manifest hashes; checks offer boundaries/targets against source; extracts only schema allowlisted fields. Each admitted application's source events are stored once, up to its last admitted offer. Excluded applications remain absent from model caches and present in the earlier manifest.
2. `fit_preprocessing`: uses training applications only. Observed numerical values on overlapping prefixes are counted once per unique source event. Static RequestedAmount is counted once per training application. Temporal statistics also use unique covered training events, an explicit engineering convention. Population standard deviation, scale floor and constant-channel handling follow the dataset contract. No observed training values is an error. Vocabularies use training categories in lexical order.
3. `PackedOffers` and collation: caches concatenate complete prefixes with offsets; indexing recovers one offer. Each offer contributes one target. Batches right-pad only to their maximum length. Inputs, labels and ID metadata are returned separately, so identifiers cannot accidentally become model fields.
4. `ipat.model.bpi17`: adds a seven-field categorical encoder and assembler. Four small model subclasses inherit the existing model forward methods, attention blocks, projection, temporal encoder, numerical encoders and classifier structure. Their constructors retain shared parameter creation order; no old module is overwritten.

## Input contract

| Key | Shape | Type | Meaning |
|---|---|---|---|
| x_obs | B,L,6 | float32 | Raw numerical values; finite zero placeholders |
| missing_mask | B,L,6 | bool | True means numerically unobserved |
| categorical | B,L,7 | int64 | Training-vocabulary IDs |
| timestamp | B,L | float32 | Elapsed seconds, not absolute date |
| delta | B,L | float32 | Preceding-event gap seconds |
| position | B,L | int64 | Zero-based index; padding -1 |
| valid_mask | B,L | bool | True means real event |
| y (separate) | B | float32 | Recorded Accepted label, false=0/true=1 |
| metadata (separate) | B dictionaries | strings | Application/offer IDs for audit and grouping |

Categorical IDs are PAD=0, MISSING=1, UNK=2; observed training categories start at 3. Mapping multiple unseen categories to UNK is deliberately lossy, so there is no promise of recovering unseen raw category identity. Raw numerical parsing/fitting uses float64 and model storage uses float32: mathematical injectivity is not a claim of bitwise recovery of all raw decimal source strings. Preservation remains scoped to the represented interface, before LayerNorm.

`Accepted`, `Selected`, identifiers and arbitrary unlisted fields are never model features, including on earlier events. No creation values are backward-filled. RequestedAmount/LoanGoal/ApplicationType remain trace-static under the documented retrospective source convention. Historical availability and CreditScore zero semantics remain unresolved source limitations; this adapter does not certify them.

## Acceptance conditions set before validation

Data fixtures must demonstrate unique-event/statistic weighting, unchanged fitting under held-out perturbations, future-event exclusion, feature allowlisting, exact target-boundary provenance, zero/missing/padding separation, known/missing/unknown categories, no double standardisation, temporal values, constant/empty channel behaviour, rejection of invalid values/scales, complete-prefix roundtrips, deterministic caches and non-mutating collation.

Model fixtures must demonstrate common dimensions and exact shared initialisation; exact identity-plus-gamma-zero logits/attention/shared-gradient parity; padding invariance within declared float32 tolerance; frozen temporal convention; one-time numerical scaling; semi-orthogonality; and a finite backward/optimiser step. The single step establishes an executable gradient path and parameter update, not learning effectiveness, convergence, generalisation or superiority.

Full-source acceptance adds admitted count checks, application separation, manifest-to-cache provenance, cache roundtrip/deterministic hashes, positive training scales, no unknown training categories, and model checks on actual training prefixes including the longest prefix. No test outcomes are used for model selection. The builder records software versions, module hashes, numerical errors and tolerances. Fixture output remains separate evidence from the JSON report.

Freeze processed-data acceptance only after these checks also pass in the user's project. EXP-05 training remains a separate task with its own fixed budgets/seeds and outcome analysis. Negative comparative findings would not invalidate a correctly implemented adapter.

## Evidence and sources

Project-specific definitions: canonical IPAT Writing Context Lock MATH-01/02/03/06/08; inspected IMP-01/02/03 source; locally validated BPI17 contract and source manifest. These are the authority for IPAT design decisions, rather than external publications.

Training-only fitting follows the [official scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage): preprocessing parameters must be learned on training data and reused on held-out data. The grouping, fitting weights and schema adapter layout are explicit IPAT engineering choices. No performance claim is derived from that documentation.

The source target convention is grounded in the [EBES authors' BPI17 preprocessing](https://github.com/On-Point-RND/EBES/blob/main/preprocess/bpi_17.py). Our grouped four-way split differs and is not an exact reproduction of that benchmark. Source attribution/terms/version qualifications from the dataset contract remain in force.
