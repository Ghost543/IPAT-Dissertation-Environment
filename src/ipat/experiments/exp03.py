"""Matched EXP-03 runner. Sweep evaluation occurs only after checkpoint selection.

Outputs use VALIDATION_PENDING, never 'hypothesis passed'. Scientific completion
requires the investigator's review of the artefacts and local test results.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ipat.data.synthetic import load_hdf5
from ipat.data.torch_dataset import SyntheticTorchDataset
from ipat.experiments.exp01 import seed_everything
from ipat.experiments.exp03_contract import (
    MODELS, Exp03Config, aggregate_runs, array_hash, file_hash, json_hash, require,
    summarise_predictions, validate_inputs,
)
from ipat.model.adaptive import FullIPATConfig, FullIPATTransformer
from ipat.model.baseline import BaselineConfig, StandardTransformerBaseline
from ipat.model.retention import RetentionAwareTransformer, RetentionConfig
from ipat.training.preprocessing import fit_baseline_preprocessing_stats

# Ground-truth diagnostics must never be passed as model inputs or optimiser targets.
INPUT_FIELDS = ("x_obs", "missing_mask", "event_type", "nuisance_cat", "delta",
                "timestamp", "position", "valid_mask")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_hash(state) -> str:
    return json_hash({k: array_hash(v.detach().cpu().numpy()) for k, v in sorted(state.items())})


def shared_check(source, target) -> dict:
    src, dst = source.state_dict(), target.state_dict()
    require(set(src).issubset(dst), "Missing shared state entries")
    for name, value in src.items():
        require(value.shape == dst[name].shape and torch.equal(value, dst[name]),
                f"Shared initial state differs: {name}")
    return {"shared_entries": len(src), "max_difference": 0.0,
            "shared_sha256": state_hash(src), "target_only_entries": sorted(set(dst)-set(src))}


def build_models(stats, cfg: Exp03Config, base: BaselineConfig, seed: int):
    seed_everything(seed, True)
    baseline = StandardTransformerBaseline(base, stats)
    seed_everything(seed, True)
    retention = RetentionAwareTransformer(RetentionConfig(**asdict(base), gamma=cfg.gamma), stats)
    retention.load_imp01_shared_state(baseline.state_dict())
    seed_everything(seed, True)
    full = FullIPATTransformer(FullIPATConfig(
        **asdict(base), gamma=cfg.gamma, identity_floor=cfg.identity_floor,
        asinh_scales=cfg.asinh_scales), stats)
    full.load_imp02_shared_state(retention.state_dict())
    models = dict(zip(MODELS, (baseline, retention, full)))
    evidence = {"baseline_to_retention": shared_check(baseline, retention),
                "retention_to_full": shared_check(retention, full)}
    for name, model in models.items():
        evidence[name] = {"parameter_count": sum(p.numel() for p in model.parameters()),
                          "initial_state_sha256": state_hash(model.state_dict())}
    return models, evidence


def initial_parity(models, val, cfg, device):
    batch = next(iter(loader(val, range(min(8, len(val["y"]))), cfg.batch_size)))
    inputs = model_inputs(batch, device)
    for model in models.values():
        model.to(device).eval()
    with torch.no_grad():
        a, b = [models[name](inputs, return_diagnostics=True) for name in ("baseline", "retention")]
    errors = {"logits": float((a["logits"]-b["logits"]).abs().max()),
              "attention": max(float((x-y).abs().max()) for x, y in zip(a["attention"], b["attention"]))}
    require(all(np.isfinite(e) and e <= 1e-6 for e in errors.values()), "Initial zero-effect parity failed")
    for model in models.values():
        model.cpu()
    return {"status": "PASS", "atol": 1e-6, "maximum_differences": errors}


def loader(data, indices, batch_size):
    return DataLoader(SyntheticTorchDataset(data, indices=list(indices)), batch_size=batch_size,
                      shuffle=False, num_workers=0, drop_last=False)


def model_inputs(batch, device):
    return {key: batch[key].to(device) for key in INPUT_FIELDS}


def epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    # Local RNG independent of model construction, dropout or validation iteration.
    return np.random.default_rng(np.random.SeedSequence([seed, 303, epoch])).permutation(n)


def validation_nll(model, val, cfg, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader(val, range(len(val["y"])), cfg.batch_size):
            logits = model(model_inputs(batch, device))["logits"]
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"].to(device).float(), reduction="sum")
            require(bool(torch.isfinite(loss)), "Non-finite validation loss")
            total += float(loss)
    return total / len(val["y"])


def train_model(model, train, val, cfg, seed, device, name="model"):
    """No sweep argument: evaluation data cannot determine stopping/checkpoint choice."""
    seed_everything(seed, True)
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    trajectory, order_hashes = [], []
    started = time.monotonic()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = epoch_order(len(train["y"]), seed, epoch)
        order_hashes.append(array_hash(order))
        total = 0.0
        for batch in loader(train, order, cfg.batch_size):
            optimiser.zero_grad(set_to_none=True)
            logits = model(model_inputs(batch, device))["logits"]
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"].to(device).float())
            require(bool(torch.isfinite(loss)), "Non-finite training loss")
            loss.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                    "Non-finite gradient")
            optimiser.step()
            total += float(loss.detach()) * len(batch["y"])
        score = validation_nll(model, val, cfg, device)
        trajectory.append({"epoch": epoch, "train_nll": total / len(train["y"]), "val_nll": score})
        if score < best_loss - cfg.min_delta:
            best_loss, best_epoch, best_state, stale = score, epoch, cpu_state(model), 0
        else:
            stale += 1
        print(f"{name} seed={seed} epoch={epoch:03d} train_nll={trajectory[-1]['train_nll']:.6f} "
              f"val_nll={score:.6f} best_epoch={best_epoch}", flush=True)
        if stale >= cfg.patience:
            break
    require(best_state is not None, "No valid checkpoint")
    model.load_state_dict(best_state)
    require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),
            "Non-finite model state")
    return {"best_epoch": best_epoch, "best_val_nll": best_loss,
            "epochs_run": len(trajectory), "trajectory": trajectory,
            "epoch_order_sha256": order_hashes, "selected_state_sha256": state_hash(best_state),
            "training_seconds": time.monotonic()-started}


def evaluate(model, sweep, cfg, device):
    model.eval()
    probabilities, logits_parts, pieces = [], [], {}
    offset = 0
    with torch.no_grad():
        for batch in loader(sweep, range(len(sweep["y"])), cfg.batch_size):
            inputs = model_inputs(batch, device)
            valid = inputs["valid_mask"]
            signal = batch["signal_mask"].to(device) & valid
            nuisance = valid & ~signal
            count = valid.sum(1)
            require(bool((signal.sum(1) > 0).all() & (nuisance.sum(1) > 0).all()), "Undefined signal diagnostic")
            out = model(inputs, return_diagnostics=True)
            logits_parts.append(out["logits"].cpu().numpy())
            probabilities.append(out["prob"].cpu().numpy())
            targets = torch.as_tensor(sweep["probe_target_position"][offset:offset+len(count)],
                                      dtype=torch.long, device=device)
            rows = torch.arange(len(count), device=device)
            attention = torch.stack(out["attention"]).mean(dim=(0, 2))  # B x query x key
            require(bool(torch.isfinite(attention).all()), "Non-finite attention")
            require(bool((attention[~valid] == 0).all()), "Padded queries have attention")
            require(bool((attention.masked_select(~valid[:, None, :]) == 0).all()), "Padded keys have attention")
            key_mass = (attention * valid[:, :, None]).sum(1) / count[:, None]
            values = {
                "signal_attention_mass": (key_mass * signal).sum(1),
                "probe_attention_mass": key_mass[rows, targets],
                "uniform_signal_mass": signal.sum(1) / count,
                "uniform_probe_mass": 1.0 / count,
            }
            if "retention" in out:
                r = out["retention"]["r"]
                require(bool(torch.isfinite(r).all() & (r >= 0).all() & (r <= 1).all()), "Invalid retention")
                require(bool((r[~valid] == 0).all()), "Padded retention nonzero")
                centred = out["retention"]["centered_log_r"]
                require(bool(torch.isfinite(centred).all()), "Invalid log-retention")
                values.update({
                    "signal_retention_mean": (r * signal).sum(1) / signal.sum(1),
                    "nuisance_retention_mean": (r * nuisance).sum(1) / nuisance.sum(1),
                    "probe_retention": r[rows, targets],
                    "log_retention_signal_margin": (centred * signal).sum(1)/signal.sum(1)
                        - (centred * nuisance).sum(1)/nuisance.sum(1),
                })
                # Per-sequence signal-vs-nuisance rank AUROC, equal sequence weighting.
                aucs = []
                for row in range(len(count)):
                    scores = r[row, valid[row]].cpu().numpy()
                    truth = signal[row, valid[row]].cpu().numpy()
                    from sklearn.metrics import roc_auc_score
                    aucs.append(roc_auc_score(truth, scores))
                values["retention_signal_auroc"] = torch.tensor(aucs, device=device)
            for name, value in values.items():
                pieces.setdefault(name, []).append(value.cpu().numpy())
            offset += len(count)
    logits = np.concatenate(logits_parts)
    diagnostics = {k: np.concatenate(v) for k, v in pieces.items()}
    metrics = summarise_predictions(sweep, logits, diagnostics)
    from scipy.special import expit
    reported_prob = np.concatenate(probabilities)
    require(np.allclose(reported_prob, expit(logits.astype(np.float64)), rtol=0, atol=2e-7),
            "Model probability/logit mismatch")
    arrays = {"logits": logits, "probability": expit(logits.astype(np.float64)),
              "sequence_id": sweep["sequence_id"], "triplet_id": sweep["triplet_id"],
              "variant": sweep["variant"], "target_position": sweep["probe_target_position"],
              "label": sweep["y"], "p_true": sweep["p_true"], **diagnostics}
    return metrics, arrays


def source_inventory(root):
    paths = sorted((root / "src" / "ipat").rglob("*.py")) + [root / "scripts" / "run_exp03.py"]
    return {str(p.relative_to(root)): file_hash(p) for p in paths}


def environment(device, cfg):
    versions = {}
    for package in ("torch", "numpy", "scipy", "scikit-learn", "h5py", "pyyaml"):
        versions[package] = importlib.metadata.version(package)
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "packages": versions, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cpu_threads": cfg.cpu_threads, "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "torch_build": torch.__config__.show()}


def load_completed(folder: Path, fingerprint: str):
    report = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    require(report.get("content_sha256") == json_hash({k: v for k, v in report.items()
                                                     if k != "content_sha256"}),
            "Stored run summary was altered")
    require(report["fingerprint"] == fingerprint and report["status"] == "RUN_COMPLETE",
            "Stored run does not match this experiment")
    for name, digest in report["artefact_sha256"].items():
        require(file_hash(folder / name) == digest, f"Modified/corrupt resume artefact: {name}")
    return report


def run_one(folder, model, train, val, sweep, cfg, seed, device, name, fingerprint):
    folder.mkdir(parents=True, exist_ok=True)
    training = train_model(model, train, val, cfg, seed, device, name)
    metrics, arrays = evaluate(model, sweep, cfg, device)
    checkpoint_tmp = folder / "checkpoint.pt.tmp"
    torch.save(cpu_state(model), checkpoint_tmp)
    checkpoint_tmp.replace(folder / "checkpoint.pt")
    with (folder / "predictions.npz.tmp").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    (folder / "predictions.npz.tmp").replace(folder / "predictions.npz")
    report = {"status": "RUN_COMPLETE", "model": name, "seed": seed,
              "fingerprint": fingerprint, "training": training, "metrics": metrics,
              "artefact_sha256": {p: file_hash(folder / p) for p in ("checkpoint.pt", "predictions.npz")}}
    report["content_sha256"] = json_hash(report)
    atomic_json(folder / "run.json", report)  # Commit marker written last.
    model.cpu()
    return report


def compare_replay(primary, replay, first_folder, second_folder, atol):
    a, b = primary["training"], replay["training"]
    require(a["best_epoch"] == b["best_epoch"] and a["epoch_order_sha256"] == b["epoch_order_sha256"],
            "Replay checkpoint/order mismatch")
    require(a["trajectory"] == b["trajectory"], "Replay training trajectory mismatch")
    require(a["selected_state_sha256"] == b["selected_state_sha256"], "Replay selected state mismatch")
    with np.load(first_folder / "predictions.npz", allow_pickle=False) as left, \
            np.load(second_folder / "predictions.npz", allow_pickle=False) as right:
        require(set(left.files) == set(right.files), "Replay output fields differ")
        max_error = 0.0
        for key in left.files:
            if np.issubdtype(left[key].dtype, np.floating):
                require(np.allclose(left[key], right[key], rtol=0, atol=atol), f"Replay differs: {key}")
                max_error = max(max_error, float(np.max(np.abs(left[key]-right[key]))))
            else:
                require(np.array_equal(left[key], right[key]), f"Replay identity differs: {key}")
    return {"status": "PASS", "max_output_difference": max_error, "atol": atol,
            "exact_selected_state": True, "exact_training_trajectory": True}


def run_exp03(train_path, val_path, sweep_path, out_dir, cfg: Exp03Config,
              base: BaselineConfig | None = None, device="cpu", resume=False,
              preflight_only=False, development=False):
    cfg.validate(formal=not development)
    base = base or BaselineConfig()
    base.validate()
    torch.set_num_threads(cfg.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    root = Path(__file__).resolve().parents[3]
    paths = {"train": Path(train_path), "val": Path(val_path), "sweep": Path(sweep_path)}
    data = {key: load_hdf5(path) for key, path in paths.items()}
    gate = validate_inputs(data["train"], data["val"], data["sweep"], formal=not development)
    require(base.n_num == data["train"]["x_obs"].shape[-1]
            and base.event_type_cardinality == 4
            and base.nuisance_cardinality == data["train"].metadata["config"]["n_nuisance_categories"],
            "Model/data feature contract mismatch")
    stats = fit_baseline_preprocessing_stats(data["train"])
    env = environment(device, cfg)
    manifest = {"experiment": "EXP-03", "version": "v0.1", "development": development,
                "protocol": asdict(cfg), "model": asdict(base), "preprocessing": stats.to_dict(),
                "inputs_sha256": {k: file_hash(p) for k, p in paths.items()},
                "source_sha256": source_inventory(root), "environment": env}
    fingerprint = json_hash(manifest)
    folder = Path(out_dir)
    existing = folder / "manifest.json"
    if folder.exists() and any(folder.iterdir()):
        require(resume and existing.exists(), "Output exists: use --resume or a new empty directory")
        previous = json.loads(existing.read_text(encoding="utf-8"))
        require(previous["fingerprint"] == fingerprint,
                "Resume refused: configuration/data/source/environment changed; use a new output directory")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(existing, {"fingerprint": fingerprint, **manifest})
    atomic_json(folder / "preflight.json", {"fingerprint": fingerprint, "data_gates": gate,
                "status": "PREFLIGHT_PASS", "formal_runs_performed": False,
                "scope": "Data validation only; no predictive result"})
    if preflight_only:
        return {"status": "PREFLIGHT_PASS", "output": str(folder)}
    runs, matching = [], {}
    for seed in cfg.seeds:
        models, evidence = build_models(stats, cfg, base, seed)
        evidence["initial_zero_effect_parity"] = initial_parity(models, data["val"], cfg, device)
        matching[str(seed)] = evidence
        for name, model in models.items():
            run_folder = folder / f"seed_{seed}" / name
            if (run_folder / "run.json").exists():
                report = load_completed(run_folder, fingerprint)
                require(report["model"] == name and report["seed"] == seed, "Resume identity mismatch")
                print(f"Reusing verified {name} seed={seed}", flush=True)
            else:
                report = run_one(run_folder, model, data["train"], data["val"], data["sweep"],
                                 cfg, seed, device, name, fingerprint)
            runs.append(report)
            atomic_json(folder / "progress.json", {"status": "IN_PROGRESS", "fingerprint": fingerprint,
                "completed_primary_runs": len(runs), "expected_primary_runs": 3*len(cfg.seeds)})
        del models
    # Unequal stopping epochs are allowed; consumed order prefixes must be identical.
    for seed in cfg.seeds:
        orders = [r["training"]["epoch_order_sha256"] for r in runs if r["seed"] == seed]
        n = min(map(len, orders))
        require(all(order[:n] == orders[0][:n] for order in orders), "Unmatched batch order")
    replay_evidence = {}
    seed = cfg.seeds[0]
    models, _ = build_models(stats, cfg, base, seed)
    for name, model in models.items():
        replay_folder = folder / "replay" / f"seed_{seed}" / name
        if (replay_folder / "run.json").exists():
            replay = load_completed(replay_folder, fingerprint)
        else:
            replay = run_one(replay_folder, model, data["train"], data["val"], data["sweep"],
                             cfg, seed, device, name, fingerprint)
        original = next(r for r in runs if r["seed"] == seed and r["model"] == name)
        replay_evidence[name] = compare_replay(original, replay, folder / f"seed_{seed}" / name,
                                              replay_folder, cfg.replay_atol)
    prediction_count = sum(r["metrics"]["prediction_count"] for r in runs)
    require(prediction_count == len(cfg.seeds)*3*len(data["sweep"]["y"]), "Prediction count mismatch")
    summary = {"milestone": "EXP-03", "version": "v0.1", "fingerprint": fingerprint,
               "status": "DEVELOPMENT_ONLY" if development else "EXECUTION_COMPLETE_REVIEW_REQUIRED",
               "created_utc": datetime.now(timezone.utc).isoformat(), "data_gates": gate,
               "seeds": list(cfg.seeds), "matching": matching, "runs": runs,
               "replay": replay_evidence, "primary_prediction_count": prediction_count,
               "primary_run_count": len(runs), "replay_runs_excluded_from_inference": 3,
               "aggregate": aggregate_runs(runs, cfg.seeds),
               "interpretation": "Execution validity is not a favourable hypothesis result. Review task quality alongside sensitivity. Retention/attention are non-causal diagnostics. No end-to-end preservation claim.",
               "completion_pending": ["local acceptance-suite evidence", "investigator interpretation", "research-record freeze"]}
    summary_path = folder / "exp03_position_sweep_v0.1.json"
    atomic_json(summary_path, summary)
    atomic_json(folder / "progress.json", {"status": summary["status"], "fingerprint": fingerprint,
        "completed_primary_runs": len(runs), "replay_passed": True, "summary_sha256": file_hash(summary_path)})
    return summary
