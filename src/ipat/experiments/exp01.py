from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

# Required by PyTorch for deterministic CUDA GEMM when deterministic algorithms are enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ipat.data.synthetic import SplitData, load_hdf5
from ipat.data.torch_dataset import SyntheticTorchDataset
from ipat.model.baseline import BaselineConfig, StandardTransformerBaseline
from ipat.model.retention import RetentionAwareTransformer, RetentionConfig
from ipat.training.preprocessing import (
    BaselinePreprocessingStats,
    fit_baseline_preprocessing_stats,
)


@dataclass(frozen=True)
class Exp01Config:
    experiment_id: str = "EXP-01"
    subset_size: int = 32
    subset_seed: int = 20260903
    model_seed: int = 20260903
    steps: int = 300
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    dropout_override: float = 0.0
    gamma: float = 1.0
    target_loss: float = 0.05
    target_accuracy: float = 1.0
    log_every: int = 25
    deterministic_algorithms: bool = True
    initial_parity_atol: float = 1e-6
    require_canonical_train_split: bool = True

    def validate(self) -> None:
        if self.subset_size <= 0:
            raise ValueError("subset_size must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not (0.0 <= self.dropout_override < 1.0):
            raise ValueError("dropout_override must lie in [0, 1)")
        if self.gamma < 0:
            raise ValueError("MATH-05 requires gamma >= 0")
        if self.target_loss <= 0:
            raise ValueError("target_loss must be positive")
        if not (0.0 < self.target_accuracy <= 1.0):
            raise ValueError("target_accuracy must lie in (0, 1]")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.initial_parity_atol < 0:
            raise ValueError("initial_parity_atol must be non-negative")


def seed_everything(seed: int, deterministic_algorithms: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)


def deterministic_subset_indices(n: int, subset_size: int, seed: int) -> list[int]:
    if n <= 0:
        raise ValueError("n must be positive")
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")
    subset_size = min(subset_size, n)
    rng = np.random.default_rng(seed)
    # Sort the sampled IDs so both models consume the exact same full batch in a stable order.
    return sorted(rng.choice(n, size=subset_size, replace=False).astype(int).tolist())


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _shared_state_differences(
    baseline: StandardTransformerBaseline,
    retention: RetentionAwareTransformer,
) -> dict[str, float]:
    b_state = baseline.state_dict()
    r_state = retention.state_dict()
    diffs: dict[str, float] = {}
    for name, value in b_state.items():
        if name not in r_state or r_state[name].shape != value.shape:
            continue
        rv = r_state[name]
        if value.dtype.is_floating_point:
            diffs[name] = float((value - rv).abs().max().item())
        else:
            diffs[name] = float((value != rv).any().item())
    return diffs


def build_matched_models(
    stats: BaselinePreprocessingStats,
    exp_cfg: Exp01Config,
    baseline_cfg: BaselineConfig | None = None,
) -> tuple[StandardTransformerBaseline, RetentionAwareTransformer, dict[str, Any]]:
    """Build IMP-01 and IMP-02 with identical shared initial state.

    The retention scorer is the only additional trainable component.  It is zero initialised, so
    positive gamma has zero forward effect at step 0.  Shared IMP-01 state is explicitly copied into
    IMP-02 rather than relying only on construction-order RNG equivalence.
    """
    exp_cfg.validate()
    base = baseline_cfg or BaselineConfig()
    base = replace(base, dropout=exp_cfg.dropout_override)
    ret_cfg = RetentionConfig(**asdict(base), gamma=exp_cfg.gamma)

    seed_everything(exp_cfg.model_seed, exp_cfg.deterministic_algorithms)
    baseline = StandardTransformerBaseline(base, stats)

    # Construction of the retention model may consume RNG differently in future revisions;
    # explicit shared-state loading makes the matched contract robust to that implementation detail.
    seed_everything(exp_cfg.model_seed, exp_cfg.deterministic_algorithms)
    retention = RetentionAwareTransformer(ret_cfg, stats)
    retention.load_imp01_shared_state(baseline.state_dict())

    diffs = _shared_state_differences(baseline, retention)
    max_shared_diff = max(diffs.values(), default=0.0)
    if max_shared_diff != 0.0:
        raise RuntimeError(f"Matched initialisation failed; max shared-state diff={max_shared_diff}")

    retention_names = {name for name, _ in retention.named_parameters()}
    baseline_names = {name for name, _ in baseline.named_parameters()}
    extra_trainable = sorted(retention_names - baseline_names)
    if extra_trainable != ["retention_scorer.linear.bias", "retention_scorer.linear.weight"]:
        raise RuntimeError(f"Unexpected IMP-02-only trainable parameters: {extra_trainable}")

    metadata = {
        "baseline_state_sha256": _state_sha256(baseline.state_dict()),
        "max_initial_shared_state_abs_diff": max_shared_diff,
        "retention_only_trainable_parameters": extra_trainable,
        "baseline_parameter_count": sum(p.numel() for p in baseline.parameters()),
        "retention_parameter_count": sum(p.numel() for p in retention.parameters()),
    }
    metadata["retention_extra_parameter_count"] = (
        metadata["retention_parameter_count"] - metadata["baseline_parameter_count"]
    )
    return baseline, retention, metadata


def _batch_from_split(
    split: SplitData,
    indices: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dataset = SyntheticTorchDataset(split, indices=indices)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    return {k: v.to(device) for k, v in batch.items()}


def _initial_parity(
    baseline: StandardTransformerBaseline,
    retention: RetentionAwareTransformer,
    batch: dict[str, torch.Tensor],
    atol: float,
) -> dict[str, float | bool]:
    baseline.eval()
    retention.eval()
    with torch.no_grad():
        b = baseline(batch, return_diagnostics=True)
        r = retention(batch, return_diagnostics=True)
    logit_diff = float((b["logits"] - r["logits"]).abs().max().item())
    hidden_diff = float((b["hidden"] - r["hidden"]).abs().max().item())
    z_diff = float((b["z_pre_norm"] - r["z_pre_norm"]).abs().max().item())
    attention_diff = max(
        (float((ba - ra).abs().max().item()) for ba, ra in zip(b["attention"], r["attention"])),
        default=0.0,
    )
    contrast = float(r["retention"]["contrast"].abs().max().item())
    within_tolerance = (
        logit_diff <= atol
        and hidden_diff <= atol
        and z_diff <= atol
        and attention_diff <= atol
        and contrast == 0.0
    )
    return {
        "within_tolerance": within_tolerance,
        "atol": atol,
        "max_logit_abs_diff": logit_diff,
        "max_hidden_abs_diff": hidden_diff,
        "max_z_pre_norm_abs_diff": z_diff,
        "max_attention_abs_diff": attention_diff,
        "max_zero_init_retention_contrast": contrast,
    }


def _train_one(
    name: str,
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    exp_cfg: Exp01Config,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp_cfg.learning_rate,
        weight_decay=exp_cfg.weight_decay,
    )
    y = batch["y"].float()
    trajectory: list[dict[str, float | int]] = []
    first_loss: float | None = None
    reached_step: int | None = None
    final_loss = float("nan")
    final_accuracy = float("nan")
    final_contrast: float | None = None

    for step in range(1, exp_cfg.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if name == "retention":
            out = model(batch, return_diagnostics=True)
            contrast = float(out["retention"]["contrast"].mean().detach().item())
        else:
            out = model(batch)
            contrast = None

        loss = F.binary_cross_entropy_with_logits(out["logits"], y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = (out["logits"] >= 0).long()
            accuracy = float((pred == batch["y"]).float().mean().item())

        final_loss = float(loss.item())
        final_accuracy = accuracy
        final_contrast = contrast
        if first_loss is None:
            first_loss = final_loss

        if step == 1 or step % exp_cfg.log_every == 0:
            record: dict[str, float | int] = {
                "step": step,
                "loss": final_loss,
                "accuracy": final_accuracy,
            }
            if contrast is not None:
                record["retention_contrast"] = contrast
            trajectory.append(record)

        if final_loss <= exp_cfg.target_loss and final_accuracy >= exp_cfg.target_accuracy:
            reached_step = step
            # Always save the exact acceptance step even when it is not a scheduled logging step.
            if not trajectory or trajectory[-1]["step"] != step:
                record = {"step": step, "loss": final_loss, "accuracy": final_accuracy}
                if contrast is not None:
                    record["retention_contrast"] = contrast
                trajectory.append(record)
            break

    return {
        "passed": reached_step is not None,
        "reached_step": reached_step,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "final_retention_contrast": final_contrast,
        "trajectory": trajectory,
    }


def run_exp01(
    train: SplitData | str | Path,
    exp_cfg: Exp01Config,
    *,
    device: str | torch.device | None = None,
    baseline_cfg: BaselineConfig | None = None,
) -> dict[str, Any]:
    """Run the matched tiny-sample sanity experiment.

    This is an implementation sanity experiment.  Step-to-target and loss trajectories are logged,
    but they must not be interpreted as generalisation performance or architectural superiority.
    """
    exp_cfg.validate()
    split = load_hdf5(train) if isinstance(train, (str, Path)) else train
    n = int(split["y"].shape[0])
    if exp_cfg.require_canonical_train_split:
        expected = int(split.metadata.get("config", {}).get("n_train", -1))
        split_name = split.metadata.get("split")
        if split_name != "train" or expected <= 0 or n != expected:
            raise ValueError(
                "EXP-01 requires the canonical DATA-01 training split; "
                f"got split={split_name!r}, n={n}, configured n_train={expected}. "
                "Regenerate with `uv run python scripts/generate_ipat_syn.py` (without --debug)."
            )
    indices = deterministic_subset_indices(n, exp_cfg.subset_size, exp_cfg.subset_seed)
    stats = fit_baseline_preprocessing_stats(split)
    device_obj = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    batch = _batch_from_split(split, indices, device_obj)

    baseline, retention, match = build_matched_models(stats, exp_cfg, baseline_cfg)
    baseline = baseline.to(device_obj)
    retention = retention.to(device_obj)

    parity = _initial_parity(baseline, retention, batch, exp_cfg.initial_parity_atol)
    if not parity["within_tolerance"]:
        raise RuntimeError(f"EXP-01 initial zero-effect parity failed: {parity}")

    # No stochastic layers are active because dropout_override=0.  Each model receives the same
    # full batch and optimiser hyperparameters.  The model-specific parameters then evolve freely.
    baseline_result = _train_one("baseline", baseline, batch, exp_cfg)
    retention_result = _train_one("retention", retention, batch, exp_cfg)
    passed = bool(baseline_result["passed"] and retention_result["passed"])

    labels = batch["y"].detach().cpu().numpy()
    result: dict[str, Any] = {
        "experiment_id": exp_cfg.experiment_id,
        "status": "PASS" if passed else "FAIL",
        "interpretation_scope": (
            "Implementation/trainability sanity only; no claim of generalisation or model superiority."
        ),
        "config": asdict(exp_cfg),
        "device": str(device_obj),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "subset_indices": indices,
        "subset_size": len(indices),
        "subset_positive_count": int(labels.sum()),
        "subset_prevalence": float(labels.mean()),
        "preprocessing": stats.to_dict(),
        "matching": match,
        "initial_zero_effect_parity": parity,
        "baseline": baseline_result,
        "retention": retention_result,
    }
    return result


def save_result(result: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return path
