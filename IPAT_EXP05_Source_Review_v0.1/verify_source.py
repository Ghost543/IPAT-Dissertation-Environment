"""Complete the reviewed BPI17 source receipt only after local archive verification."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_SHA = '183c5e5189282779c811c78c33ff936351b3dd201165d612211fc220936f8249'
EXPECTED_MD5 = '10b37a2f78e870d78406198403ff13d2'
EXPECTED_BYTES = 29658747
EVIDENCE = {
    'repository_evidence': ('evidence/BPI17_repository_record.pdf', '0dee6b1273789ba74ae84fa6e28292b75a913c1b61a9fca1bd3ea3a9ff750b36'),
    'terms_evidence': ('evidence/General_terms_of_use.pdf', 'e4dc9caf93f5d19cc5b7d1d48a15f363e5f59eb8a54f1aabafed2e79f4a26f04'),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/exp05_pilot_v0.1.json'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if config.get('source_sha256') != EXPECTED_SHA:
        raise ValueError('Configuration source differs from the frozen dataset; no receipt written.')
    for relative, expected in EVIDENCE.values():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Evidence PDF mismatch: ' + relative)
    size = args.source.stat().st_size
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with args.source.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            sha.update(chunk)
            md5.update(chunk)
    observed = dict(bytes=size, sha256=sha.hexdigest(), md5=md5.hexdigest())
    print(json.dumps(observed, indent=2))
    if (size, sha.hexdigest(), md5.hexdigest()) != (EXPECTED_BYTES, EXPECTED_SHA, EXPECTED_MD5):
        raise ValueError('Archive differs from repository/frozen identifiers; no receipt written. Retain the file and report these values.')
    with gzip.open(args.source, 'rb') as handle:
        while handle.read(1024 * 1024):
            pass  # Require successful end-of-stream and gzip integrity validation.
    receipt = {
        'status': 'VERIFIED',
        'source_sha256': EXPECTED_SHA,
        'official_url': 'https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b',
        'repository_version': 'Release first published 2017-02-06; exact archive pinned by repository MD5, byte size and frozen SHA-256. No numbered version is displayed in the supplied record.',
        'terms_url': 'https://doi.org/10.4121/resource:terms_of_use',
        'reviewer': 'Codex: visual review of user-supplied repository capture and full terms; archive verification by this local script',
        'review_date': '2026-09-13',
        'permitted_use_basis': 'The repository record explicitly lists 4TU General Terms of Use. The supplied August 2016 terms permit non-commercial reuse and transformation. This MSc dissertation use retains creator attribution, dataset citation and redistribution acknowledgement; bibliographic details of resulting publications are to be provided to 4TU (clauses 1-5). Clause 7 applies where personal data are present. This is not a Creative Commons licence assertion.',
        'repository_evidence': dict(path=EVIDENCE['repository_evidence'][0], sha256=EVIDENCE['repository_evidence'][1]),
        'terms_evidence': dict(path=EVIDENCE['terms_evidence'][0], sha256=EVIDENCE['terms_evidence'][1]),
        'archive_verification': dict(**observed, gzip_read_to_eof=True, verified_at_utc=datetime.now(timezone.utc).isoformat(), source_path=str(args.source.resolve())),
        'scope': 'Documented repository release and exact archive identity; no claim that an undisplayed version number or later terms revision was independently verified.',
    }
    destination = ROOT / 'source_review.json'
    # Exclusive creation preserves any existing receipt and its timestamp.
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('PASS: ' + str(destination))

if __name__ == '__main__':
    main()
