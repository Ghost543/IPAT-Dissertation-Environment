"""Run and record the EXP-04 acceptance suite; any skip is an incomplete gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import importlib.metadata
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/exp04/exp04_local_acceptance_v0.1.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "tests/test_exp04.py"
    spec = importlib.util.spec_from_file_location("exp04_acceptance_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    payload = {
        "milestone": "EXP-04", "scope": "Implementation acceptance; not a formal corruption-evaluation result",
        "status": "PASS" if result.wasSuccessful() and not result.skipped else "INCOMPLETE_OR_FAILED",
        "tests_run": result.testsRun,
        "passed": result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
        "skipped": [{"test": str(test), "reason": why} for test, why in result.skipped],
        "failures": [{"test": str(test), "traceback": why} for test, why in result.failures],
        "errors": [{"test": str(test), "traceback": why} for test, why in result.errors],
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "scikit-learn", "h5py", "matplotlib")},
        "python": platform.python_version(), "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((root / "src/ipat").rglob("*.py")) + [path, Path(__file__)]},
    }
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Evidence already exists; choose a new --output path to preserve it")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{payload['status']}: {payload['passed']}/{payload['tests_run']} passed; "
          f"{len(result.skipped)} skipped. Wrote {output.resolve()}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
