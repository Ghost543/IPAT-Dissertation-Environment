# IPAT Dissertation Environment v1

This scaffold defines the canonical compute environment for the IPAT dissertation.

## Frozen environment policy

- **Primary runtime:** Python.
- **Secondary/interoperable runtime:** MATLAB.
- **Canonical Python:** CPython 3.11.x.
- **Python environment/lock manager:** `uv`.
- **Deep learning:** PyTorch, installed only after the host GPU/driver has been checked with the official PyTorch installation selector. The resolved build is then locked in `uv.lock`.
- **MATLAB:** record the installed MATLAB release and toolboxes. MATLAB Engine for Python must match the installed MATLAB release.
- **Canonical interchange formats:** HDF5 for arrays/tensors, JSON/YAML for configuration and metadata, CSV/Parquet for tabular outputs.
- **No notebook is the source of truth:** notebooks may analyse results, but reusable logic lives in `src/ipat/` and is covered by tests.

## Phase 1 — Inspect the host before installing PyTorch/MATLAB Engine

### Windows PowerShell

```powershell
python --version
nvidia-smi
matlab -batch "disp(version('-release')); disp(version)"
```

### Linux/macOS

```bash
python3 --version
nvidia-smi || true
matlab -batch "disp(version('-release')); disp(version)"
```

## Phase 2 — Install uv and Python 3.11

Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv python install 3.11
uv sync --dev
```

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv sync --dev
```

This creates `.venv`. Commit `.python-version`, `pyproject.toml`, and, after full resolution, `uv.lock`.

## Phase 3 — Install PyTorch

Use the official selector at https://pytorch.org/get-started/locally/ after checking the GPU/driver. Then add the selected PyTorch build to this project and regenerate `uv.lock`.

Do **not** guess a CUDA build and do **not** change PyTorch during an experiment series without creating a new environment version.

## Phase 4 — Connect MATLAB

Python 3.11 is compatible with MATLAB R2023b through R2026a. If the installed MATLAB release is older than R2023b, stop and adjust the canonical Python version before experiments.

From Python, install or expose the MATLAB Engine package appropriate to the installed release.

For MATLAB -> Python, run:

```matlab
run("matlab/configure_python.m")
```

## Phase 5 — Validate and capture

Python:

```powershell
uv run python scripts/check_environment.py
uv run pytest
```

MATLAB:

```powershell
matlab -batch "run('matlab/check_environment.m')"
```

Every experiment should save a run manifest containing Python, package lock hash, PyTorch/CUDA, GPU, MATLAB release/toolboxes, OS/CPU, Git commit, experiment seeds, dataset ID, and config ID.
