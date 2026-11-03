# Measured contract preflight

Source: uploaded BPI17 archive with pinned SHA-256. No model results.

| Partition | Applications | Offers | False | True | True % | Median length | Max length |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 13,865 | 19,071 | 5,837 | 13,234 | 69.39 | 11 | 139 |
| validation | 6,006 | 7,871 | 2,237 | 5,634 | 71.58 | 10 | 93 |
| calibration | 4,770 | 6,117 | 1,715 | 4,402 | 71.96 | 10 | 111 |
| test | 6,356 | 8,613 | 2,555 | 6,058 | 70.34 | 10 | 133 |
| excluded | 512 | 1,323 | 515 | 808 | 61.07 | 18 | 138 |

10 fixture tests, 9 preflight checks and 5 additional cross-checks passed. Full candidate length histogram and class counts match the prior independent audit.

512 applications/1,323 offers excluded by the boundary rule. Exclusion is associated with longer prefixes and higher false-label prevalence. All exclusions are retained in the manifest.

These are development measurements. Adapter validation, user-local replication and source review remain pending.
