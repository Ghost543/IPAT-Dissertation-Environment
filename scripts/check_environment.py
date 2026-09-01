from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError


def safe_version(pkg: str):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def command(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"error": repr(exc)}


repo = Path(__file__).resolve().parents[1]
lock = repo / "uv.lock"
manifest = {
    "environment_id": "IPAT-DISS-ENV-v1",
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "packages": {name: safe_version(name) for name in [
        "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyarrow", "h5py",
        "pyyaml", "pydantic", "optuna", "tqdm", "torch"
    ]},
    "git": command(["git", "rev-parse", "HEAD"]),
    "nvidia_smi": command(["nvidia-smi"]),
    "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
}

try:
    import torch
    manifest["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
except Exception as exc:
    manifest["torch"] = {"available": False, "error": repr(exc)}

out_dir = repo / "env" / "manifests"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "environment_manifest.json"
out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest, indent=2))
print(f"\nWrote: {out_file}")
