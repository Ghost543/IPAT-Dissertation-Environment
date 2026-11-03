# EXP-05 source-review evidence, v0.1

The two supplied PDFs establish the BPI Challenge 2017 identity and its listed
4TU General Terms of Use. Repository capture page 1 gives DOI
10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b, creator Boudewijn van Dongen,
publisher Eindhoven University of Technology and first publication 2017-02-06.
Page 2 lists BPI Challenge 2017.xes.gz, 29,658,747 bytes and MD5
10b37a2f78e870d78406198403ff13d2. No numbered dataset version is visible;
we identify the release by date and exact file identifiers instead.

The supplied August 2016 terms allow non-commercial reuse/transformation,
require attribution and dataset citation, retain attribution on redistribution,
and require bibliographic details of resulting publications to be provided to
4TU. We have not sent any notification. Retrospective feature availability and
CreditScore-zero interpretation remain the previously documented limitations.

The temporary archive in the assistant workspace was found truncated at
21,474,304 bytes with a gzip EOF error on 2026-09-13. This does not establish a
problem with the user's Windows file. The historical frozen source audit
records 29,658,747 bytes and SHA-256
183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249.
Do not replace frozen evidence with the incomplete temporary copy.

Before creating the receipt, verify_source.py requires the local raw archive's
byte size and MD5 to match the repository, SHA-256 to match the accepted pilot
configuration, both PDFs to match their recorded SHA-256, and gzip to read
successfully to EOF. This binds the reviewed documents to the actual archive.
The MD5 is a repository comparison identifier; SHA-256 also pins the source.
The script reads raw data only for integrity checking, without computing
predictive metrics or changing the processed cache, protocol or model code.
No new preflight is required for this documentation-only package.

Download the ZIP into your project root, then run in PowerShell:

```powershell
Expand-Archive -LiteralPath '.\IPAT_EXP05_Source_Review_v0.1.zip' -DestinationPath '.\IPAT_EXP05_Source_Review_v0.1'
uv run python .\IPAT_EXP05_Source_Review_v0.1\verify_source.py --source '.\BPI Challenge 2017.xes.gz'
if ($LASTEXITCODE -ne 0) { throw 'Source verification failed; retain the output and do not start the pilot.' }
```

If your raw archive is elsewhere, change only --source. A successful check
creates source_review.json beside verify_source.py; the PDF paths inside the
receipt are relative to the receipt, as required by the existing runner.
An existing receipt is never overwritten. Keep the complete evidence folder.

After PASS, run the accepted formal pilot from the project root:

```powershell
$Exp05Pilot = '.\results\exp05_pilot_v0.1'
uv run python scripts/run_exp05_pilot.py pilot `
  --cache results/bpi17_adapter_v0.1_local/cache `
  --output $Exp05Pilot `
  --device cuda --microbatch 64 `
  --source-review IPAT_EXP05_Source_Review_v0.1/source_review.json `
  --preflight results/exp05_preflight_local_iofix_v0.1.1/exp05_preflight.json
if ($LASTEXITCODE -ne 0) { throw 'Pilot interrupted or failed; retain all output.' }
Start-Process (Join-Path $Exp05Pilot 'plots\report.html')
```

Use the actual accepted preflight location if yours differs. Do not change
its contents. If this pilot directory already holds the same interrupted run,
add --resume to the pilot command; do not delete evidence. A new run requires
a new output directory. Do not change the runtime or code between preflight
and pilot; the runner checks their exact match.

The pilot uses the accepted train/validation protocol, all three seeds,
20 epochs per neural arm, three GBDT fits, the dummy reference, and existing
plots/offline HTML. Calibration/test predictive evaluation remains disabled.
Execution PASS is distinct from hypothesis support.

Return source_review.json, exp05_acceptance.json and exp05_pilot_summary.json
from the completed run. Preserve per-arm predictions, checkpoints and plots.

Development verification: synthetic integrity fixture and compatibility with the
original provenance_review function passed; existing receipt preservation and
truncated-fixture rejection passed. These checks are engineering evidence only.
The canonical BPI17 archive must still pass on the user machine.
