# ROB-01 implementation record

User accepted the two-family controlled protocol on 23 September 2026. This additive implementation binds the reviewed EXP-06 evidence, reuses the accepted inference/metrics/IO components and does not edit prior modules. The runner uses the existing EXP-06 parent replay functions, exact repeat-inference checks and atomic output writer.

Operational choices now fixed: SHA-256 keyed source selection seed 70101; corruption replicates 1/2/3; application-cluster bootstrap PCG64 seed 70102 with 2,000 draws. The registered families/severities, source-unit denominator, first/offer-event protection, frozen temperatures and original-length strata follow the accepted concept. Descriptive bootstrap intervals condition on both fixed predictors and the three corruption realisations.

Source reconstruction validates exact consistency across overlapping packed prefixes. Labels and IDs remain metadata. Numerical missingness masks RequestedAmount once per application and event-local fields once per source cell. Deletion protects all creation events and the first event. Gaps sum original intervening represented gaps in float64 then cast once. Original timestamps survive. Each source plan is saved; predicted inputs are hashed per condition. GBDT summaries are recomputed by the frozen prefix_summaries implementation.

Numerical development verification and report rendering are documented in evidence/. PyTorch is unavailable in the development runtime, so full checkpoint integration remains a required user-local step. verify_rob01.py runs the complete synthetic 18-predictor ×18-condition grid using the same evaluation driver. Only that local verification can create rob01_runner_acceptance.json for the real runner. The development receipt cannot satisfy the real-run gate.

No robustness outcomes from the real test cohort were computed during implementation. The synthetic report is labelled as such on every figure. Do not interpret any synthetic numerical result as scientific evidence.
