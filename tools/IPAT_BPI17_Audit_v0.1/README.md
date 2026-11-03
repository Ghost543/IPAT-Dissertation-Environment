# IPAT BPI17 source audit v0.1

Date: 10 September 2026. Status: **source integrity passed; modelling readiness requires review**.

This kit audits the user-supplied archive without altering it, fitting preprocessing, selecting a model or constructing a training split. It does not contain the raw dataset. EXP-04 and the frozen MATH specifications are unchanged.

## Reproduce in Windows PowerShell

Extract the accompanying ZIP into a new folder. From the dissertation project root, set the archive path to its actual location:

```powershell
$auditRoot = Join-Path (Get-Location) 'tools\IPAT_BPI17_Audit_v0.1'
$rawArchive = Join-Path $env:USERPROFILE 'Downloads\BPI Challenge 2017.xes.gz'
$auditOutput = Join-Path (Get-Location) ('results\data03\bpi17_audit_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))

uv run python -m unittest discover -s "$auditRoot\tests" -v
uv run python "$auditRoot\scripts\audit_bpi17.py" --input "$rawArchive" --output-dir "$auditOutput" --plots
Invoke-Item "$auditOutput\bpi17_audit.html"
```

The parser and tests require Python 3.11+ and the standard library. Plots use matplotlib if already installed in the project environment; otherwise the JSON and HTML are produced and `plots_generated` is false. No environment or dependency changes are made automatically. An existing non-empty output directory is rejected to protect evidence.

Eight fixture tests cover prefix cutoffs, offer-specific outcome linkage, disagreement visibility, timestamp ordering, timezone handling, boundary overlap, bad embedded counts and truncated archives. Passing these tests validates the audit behaviour, not model performance. The command can exit successfully with `readiness_status=REVIEW_REQUIRED`; inspect the JSON status and findings.

## Provenance

- Uploaded filename: BPI Challenge 2017.xes.gz
- Compressed size: 29,658,747 bytes.
- SHA-256: `183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249`.
- User-supplied attribution: Boudewijn van Dongen; publisher Eindhoven University of Technology; first published 6 February 2017.
- User-supplied description: loan applications filed through an online system in 2016 and subsequent events until 1 February 2017, 15:11; multiple offers per application.
- User-supplied terms label: [4TU General Terms of Use](https://doi.org/10.4121/resource:terms_of_use).
- [Dataset landing page linked by EBES](https://data.4tu.nl/articles/_/12696884/1).
- Independent retrieval of the landing page returned HTTP 403; the full terms were not retrieved. No Creative Commons, MIT or other dataset licence is inferred. The EBES software licence is not a dataset licence.
- The DOI `10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f` in the supplied description references the earlier related dataset; it is not used as the identifier of this uploaded 2017 archive.
- The file hash identifies the bytes audited here. It is not a comparison against an independently downloaded publisher checksum. Exact repository version and full terms remain to be verified.

## Method and preservation boundary

The reader processes the gzip stream to the end and counts XES trace and event elements. Original within-trace event order is retained. An application is one trace; an offer prefix consists of all events through an `O_Create Offer` event, inclusive. The creation event's `EventID` identifies the offer. Its equality to the next `O_Created` event's `OfferID` is checked across all offers.

The audit reads later outcomes only to investigate target semantics. Future events are never included in the candidate prefix length or prefix missingness calculation. No model tensors are produced. Trace-level `RequestedAmount`, `LoanGoal` and `ApplicationType` are treated as repeated attributes for missingness accounting, not as proof of historical availability at each event.

Missingness rates weight each candidate prefix event occurrence equally. Shared histories therefore appear in multiple offer prefixes. These are dataset descriptions, not independent observations for statistical inference. Categorical cardinalities describe the complete source archive; they are not fitted training vocabularies.

MATH-02 preservation applies to the observed scalars admitted into the chosen representation. It cannot establish that a retrospectively stored field was available at prediction time, recover absent values or repair a contaminated train/test split. No mathematical specification was changed for this audit.

## Measured findings

| Quantity | Observed value |
|---|---:|
| Applications | 31,509 |
| Raw events | 1,202,267 |
| Offer prefixes | 42,995 |
| Median / mean / maximum prefix length | 10 / 13.1135 / 139 |
| Recorded Accepted=true | 30,136 (70.0919%) |
| Recorded Accepted=false | 12,859 (29.9081%) |
| Applications crossing the reference boundary | 151 |
| All offers belonging to those crossing applications | 386 |
| Offers before / on or after 20 October 2016 UTC | 34,197 / 8,798 |
| Offers with no later O_Accepted, O_Refused or O_Cancelled | 174 |
| CreditScore equal to zero at offer creation | 27,735 (64.5075%) |

The raw events span `2016-01-01T09:51:15.304+00:00` to `2017-02-01T14:11:03.499+00:00`. The one-hour difference from the supplied descriptive end time is compatible with local CET versus UTC, but the description's timezone was not independently verified.

Eight integrity gates pass. Application and offer IDs are present and unique; timestamps parse with timezones and are strictly increasing within each trace; all creation-to-next-offer links agree; labels are Boolean; event totals match embedded metadata. No exact duplicate event records were found within an application. Cross-application duplicate-history equivalence is not tested by that check.

An independent SAX counter confirms application, event, creation-offer, label and O_Accepted totals. The report contains no predictive performance estimates.

## Target semantics: material distinction

| Recorded Accepted | Later linked O_Accepted present | Later linked O_Accepted absent |
|---|---:|---:|
| true | 14,015 | 16,121 |
| false | 3,213 | 9,646 |

The two definitions disagree for 19,334 offers (44.9680%). The observed discrepancy establishes that the fields are not interchangeable. It does not establish which represents the desired business outcome, nor prove corrupted labels.

The [EBES reference script](https://github.com/On-Point-RND/EBES/blob/main/preprocess/bpi_17.py) takes its target from the recorded `Accepted` attribute at creation. A faithful target comparison would name that field explicitly. Defining the target instead from later `O_Accepted` events would create a different prediction task and require a censoring/label-availability policy. Neither choice has been frozen by this audit.

`Accepted` and `Selected` must not be input features in the proposed future-outcome task. Application ID, OfferID and EventID serve audit/join roles only. Other fields still require a prediction-time availability review; a field's attachment to a historical event is not itself evidence that it was not retrospectively updated.

## Schema and structural missingness

Six candidate numerical fields are present: `RequestedAmount`, `FirstWithdrawalAmount`, `NumberOfTerms`, `MonthlyCost`, `CreditScore`, `OfferedAmount`. All are populated at every offer creation. The five offer-specific fields are absent from 89.5563% of event occurrences across candidate prefixes because they are stored only at offer creation. This absence must remain distinguishable from an observed numerical zero.

Seven candidate categorical fields are available: `Action`, `org:resource`, `concept:name`, `EventOrigin`, `lifecycle:transition`, `LoanGoal`, `ApplicationType`. All are present under the trace-repetition convention described above. The current IPAT synthetic assembler has only two categorical slots, so a schema-aware extension and compatibility tests are required.

CreditScore zeros are measured, not automatically classified as missing. The same principle applies to zeros in other fields. Any sentinel interpretation must be documented before preprocessing is frozen. No global clipping, binning, vocabulary fitting or replacement has been applied.

## Split implications and next contract

The 20 October 2016 boundary is taken from the EBES source only as a diagnostic. Splitting each offer independently at that date puts 151 parent applications on both sides. Retain the application ID through all processing and enforce disjoint parent applications across train, validation, calibration and test.

A prospective option is to quarantine boundary-crossing applications, but the resulting exclusions and split counts must be measured and approved as the chosen protocol; this audit does not silently make that change. Train/calibration label availability at the evaluation cutoff and possible right-censoring also need explicit rules. A chronological input split alone does not establish that all training outcomes were knowable at that time.

Before EXP-05 can run: resolve and name the target; confirm field availability and sentinel policy; define grouped temporal partitions and label availability; pin reference source; freeze training-only preprocessing; validate the schema adapter and reproduce the processed-data manifest. Acquiring the archive and passing source integrity checks do not complete DATA-03.

## Contents and verification scope

- `scripts/audit_bpi17.py`: reproducible audit, JSON, HTML and optional PNG/SVG/PDF plots.
- `tests/test_audit_bpi17.py`: eight controlled-fixture checks.
- `evidence/`: audit generated from the supplied raw archive, independent counts and validation receipt.
- `PROVENANCE.json`: supplied versus independently measured provenance.
- `MANIFEST.json`: SHA-256 of all packaged files except the manifest itself.

The plots were visually inspected. The HTML contains local SVG figures and an expandable JSON evidence panel, with no external network dependencies; browser rendering was not independently tested. The source data and frozen model implementation are not included or modified.
