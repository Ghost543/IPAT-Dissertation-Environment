# EXP-06 v0.1 implementation record

21 September 2026. The user accepted the supplied EXP-06 protocol and authorised implementation. The original 20 September protocol/configuration bytes are retained unchanged, including their historical proposed-status wording, so their review hashes remain valid. This record supplies the later acceptance status. No live tracker update is claimed.

The implementation adds four experiment modules, two commands, one unit-test file, the approved configuration and documentation. It does not replace EXP-05/CAL-01, models, metrics, preprocessing or their partition-access guards. It reuses the accepted EXP-05 atomic writer with Windows sharing-violation retries for JSON and NPZ output.

The primary contrast, cohort, 18 arm/seed pairs, separate temperatures, threshold, metrics, weighting, 2,000 application-cluster draws with seed 60601, display seed and reporting scope match the approved protocol. No fitting, epoch selection, threshold search or corruption is added to real evaluation. Full test prefixes and class proportions are retained. All 18 validation and calibration replays must pass before test inference.

Numerical/report development validation can be performed without PyTorch by loading only detached modules through an isolated namespace. That does not exercise the normal package initialisers, checkpoint restoration, CUDA, the formal parent preflight or real data. These limitations are explicit in the development evidence. The included local verifier uses normal package imports, all five neural arm paths and the GBDT path across three seeds; it is mandatory before the real runner. The verifier fits only a small synthetic GBDT and uses initialised synthetic neural checkpoints and known positive fixture temperatures. Its outputs are never BPI17 findings.

The runtime verifies the complete approved configuration and protocol, reviewed summary and acceptance hashes, parent environment, all selected checkpoint hashes and CAL-01 source hashes. Its own local verifier receipt is bound to evaluation modules, verification code, tests, protocol and configuration. Input data, checkpoint files and frozen source modules receive before/after hashes. The formal acceptance file is written last, after plots and checksums, so a partial run cannot be mistaken for an accepted result.

The report is standalone: its SVG images are embedded. PNG/PDF/SVG exports also remain available as separate files. Eleven figure families implement the protocol's seven reporting groups. Row-normalised confusion colours and explicit counts address the imbalance concern. Length labels display offer counts and class support. Sparse or absent strata remain explicit.

Scientific interpretation is unchanged: a negative IPAT comparison or worse held-out calibration score can accompany PASS. Positive temperature scaling must not change the fixed-threshold decisions; probability-rounding anomalies stop execution for investigation. AUROC/AP changes caused by probability saturation are recorded rather than misrepresented as improved discrimination.

Future changes require a documented new version before affected evaluations. Existing outputs are never overwritten. If an IO error occurs, preserve the attempt and report the failure; do not silently redirect outputs outside the project or edit frozen files to bypass hashes.
