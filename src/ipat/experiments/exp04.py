"""Frozen-checkpoint EXP-04 evaluation with automatic static and HTML reporting."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import importlib.metadata
import json
import numpy as np
import torch
from scipy.special import expit
from ipat.data.synthetic import load_hdf5
from ipat.experiments.exp03 import (
    atomic_json, build_models, loader, model_inputs, state_hash, environment,
)
from ipat.experiments.exp03_contract import Exp03Config, file_hash, json_hash, require
from ipat.experiments.exp04_contract import (
    Exp04Config, MODELS, SEEDS, CONDITIONS, PARENT_FINGERPRINT, PARENT_SUMMARY_SHA256,
    validate_data, metrics, fit_temperature, aggregate,
)
from ipat.model.baseline import BaselineConfig


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_signed(path):
    result = read_json(path)
    require(result.get("content_sha256") == json_hash({k: v for k, v in result.items() if k != "content_sha256"}),
            f"Altered JSON: {path}")
    return result


def signed_json(path, payload):
    atomic_json(path, {**payload, "content_sha256": json_hash(payload)})


def safe_relative(root, relative):
    root = Path(root).resolve()
    target = (root/relative.replace("\\", "/")).resolve()
    require(target.is_relative_to(root), "Artefact path escapes its root")
    return target


def parent_contract(parent_dir, root, formal=True):
    parent_dir, root = Path(parent_dir), Path(root)
    parent = read_json(parent_dir/"manifest.json")
    require(parent.get("fingerprint") == json_hash({k: v for k, v in parent.items() if k != "fingerprint"}),
            "EXP-03 manifest fingerprint mismatch")
    require(parent.get("experiment") == "EXP-03", "Wrong parent experiment")
    summary_path = parent_dir/"exp03_position_sweep_v0.1.json"
    summary = read_json(summary_path)
    require(summary["fingerprint"] == parent["fingerprint"], "Parent summary/manifest mismatch")
    seeds = tuple(parent["protocol"]["seeds"])
    require(tuple(summary["seeds"]) == seeds, "Parent summary seeds differ")
    if formal:
        require(not parent["development"] and seeds == SEEDS, "Formal EXP-04 requires ten formal parent seeds")
        require(parent["fingerprint"] == PARENT_FINGERPRINT, "Wrong accepted EXP-03 parent fingerprint")
        require(file_hash(summary_path) == PARENT_SUMMARY_SHA256, "Wrong accepted EXP-03 summary bytes")
        require(summary["status"] == "EXECUTION_COMPLETE_REVIEW_REQUIRED", "Parent execution incomplete")
    # Additional EXP-04 files are allowed; every recorded parent source must remain identical.
    for relative, digest in parent["source_sha256"].items():
        require(file_hash(safe_relative(root, relative)) == digest, f"Frozen parent source changed: {relative}")
    require(set(summary["replay"]) == set(MODELS)
            and all(r["status"] == "PASS" for r in summary["replay"].values()), "Parent replay incomplete")
    summary_runs = {(r["model"], r["seed"]): r for r in summary["runs"]}
    require(len(summary_runs) == len(summary["runs"]) == 3*len(seeds), "Incomplete parent run set")
    reports, inventory = {}, {}
    for seed in seeds:
        for name in MODELS:
            relative = f"seed_{seed}/{name}"
            folder = parent_dir/relative
            report = read_signed(folder/"run.json")
            require(report["status"] == "RUN_COMPLETE" and report["model"] == name and report["seed"] == seed,
                    "Parent run identity mismatch")
            require(report["fingerprint"] == parent["fingerprint"], "Parent run fingerprint mismatch")
            require(report == summary_runs.get((name, seed)), "Parent run/summary disagreement")
            for artifact, digest in report["artefact_sha256"].items():
                require(file_hash(safe_relative(folder, artifact)) == digest, f"Parent artefact changed: {relative}/{artifact}")
            require("checkpoint.pt" in report["artefact_sha256"], "Parent checkpoint missing")
            reports[name, seed] = report
            inventory[relative] = {"run_json": file_hash(folder/"run.json"),
                                  "checkpoint": file_hash(folder/"checkpoint.pt")}
    return parent, seeds, reports, {"manifest": file_hash(parent_dir/"manifest.json"),
        "summary": file_hash(summary_path), "runs": inventory}


def rank_auc(signal, scores):
    """Per-sequence rank statistic. NaN denotes undefined, never a zero score."""
    pos, neg = scores[signal], scores[~signal]
    if not len(pos) or not len(neg):
        return np.nan
    return float(((pos[:, None] > neg).astype(float)+.5*(pos[:, None] == neg)).mean())


def predict(model, data, cfg, device, diagnostics=True):
    model.eval()
    pieces = {"logits": []}
    with torch.no_grad():
        for batch in loader(data, range(len(data["y"])), cfg.batch_size):
            inputs = model_inputs(batch, device)
            out = model(inputs, return_diagnostics=diagnostics)
            logit = out["logits"].detach().cpu().numpy()
            require(np.isfinite(logit).all(), "Non-finite logits")
            require(np.allclose(out["prob"].cpu().numpy(), expit(logit.astype(float)), rtol=0, atol=2e-7),
                    "Model probability/logit mismatch")
            pieces["logits"].append(logit)
            if not diagnostics:
                continue
            valid = inputs["valid_mask"]
            signal = batch["signal_mask"].to(device) & valid
            count = valid.sum(1)
            require(bool((count > 0).all()), "Empty sequence")
            attention = torch.stack(out["attention"]).mean(dim=(0, 2))
            require(bool(torch.isfinite(attention).all()), "Non-finite attention")
            require(bool((attention[~valid] == 0).all()
                         & (attention.masked_select(~valid[:, None, :]) == 0).all()), "Padded attention nonzero")
            mass = (attention*valid[:, :, None]).sum(1)/count[:, None]
            values = {"signal_attention_mass": (mass*signal).sum(1).cpu().numpy(),
                "uniform_signal_mass": (signal.sum(1)/count).cpu().numpy(),
                "signal_event_count": signal.sum(1).cpu().numpy(),
                "nuisance_event_count": (count-signal.sum(1)).cpu().numpy()}
            if "retention" in out:
                r = out["retention"]["r"]
                require(bool(torch.isfinite(r).all() & (r >= 0).all() & (r <= 1).all()), "Invalid retention values")
                require(bool((r[~valid] == 0).all()), "Padded retention nonzero")
                rr, vv, ss = (x.cpu().numpy() for x in (r, valid, signal))
                values["retention_signal_auroc"] = np.array([rank_auc(s[v], a[v]) for s, a, v in zip(ss, rr, vv)])
                values["signal_retention_mean"] = np.array([float(a[s].mean()) if s.any() else np.nan for a, s in zip(rr, ss)])
                values["nuisance_retention_mean"] = np.array([float(a[v & ~s].mean()) if (v & ~s).any() else np.nan
                                                            for a, v, s in zip(rr, vv, ss)])
            for key, value in values.items():
                pieces.setdefault(key, []).append(value)
    arrays = {k: np.concatenate(v) for k, v in pieces.items()}
    arrays.update(sequence_id=data["sequence_id"], label=data["y"])
    if diagnostics:
        arrays["retention_ranking_eligible"] = (arrays["signal_event_count"] > 0) & (arrays["nuisance_event_count"] > 0)
    return arrays


def diagnostic_summary(arrays):
    result = {}
    for key, values in arrays.items():
        if key not in ("logits", "sequence_id", "label"):
            finite = np.isfinite(values)
            result[key] = {"mean": float(values[finite].mean()) if finite.any() else None,
                "eligible_count": int(finite.sum()), "undefined_count": int((~finite).sum())}
    result["interpretation"] = "Descriptive, non-causal diagnostics; NaN in NPZ and null in JSON mean undefined, never zero."
    return result


def save_npz(path, arrays):
    path = Path(path)
    with path.with_suffix(path.suffix+".tmp").open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    path.with_suffix(path.suffix+".tmp").replace(path)


def compare_arrays(left, right, atol):
    require(set(left) == set(right), "Replay output keys differ")
    error = 0.
    for key in left:
        a, b = left[key], right[key]
        require(a.shape == b.shape and a.dtype == b.dtype, f"Replay shape/dtype mismatch: {key}")
        if np.issubdtype(a.dtype, np.floating):
            require(np.array_equal(np.isnan(a), np.isnan(b)), f"Replay eligibility mismatch: {key}")
            require(np.allclose(a, b, rtol=0, atol=atol, equal_nan=True), f"Replay mismatch: {key}")
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                error = max(error, float(np.max(np.abs(a[finite]-b[finite]))))
        else:
            require(np.array_equal(a, b), f"Replay identities differ: {key}")
    return error


def source_inventory(root):
    paths = sorted((root/"src/ipat").rglob("*.py"))
    paths += [root/"scripts/run_exp04.py"]
    return {p.relative_to(root).as_posix(): file_hash(p) for p in paths}


def load_run(folder, fingerprint):
    report = read_signed(folder/"run.json")
    require(report["fingerprint"] == fingerprint and report["status"] == "RUN_COMPLETE", "Resume run mismatch")
    for relative, digest in report["artefact_sha256"].items():
        require(file_hash(safe_relative(folder, relative)) == digest, f"Resume artefact changed: {relative}")
    return report


def run_exp04(parent_dir, data_dir, out_dir, cfg=None, device="cpu", resume=False,
              preflight_only=False, development=False):
    cfg = cfg or Exp04Config()
    cfg.validate()
    root = Path(__file__).resolve().parents[3]
    parent_dir, data_dir, folder = map(Path, (parent_dir, data_dir, out_dir))
    require(not folder.resolve().is_relative_to(parent_dir.resolve()), "EXP-04 output must be outside EXP-03 parent")
    require(not folder.resolve().is_relative_to(data_dir.resolve()), "Output must be outside frozen data directory")
    torch.set_num_threads(cfg.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    print("Verifying frozen EXP-03 parent artefacts and source files...", flush=True)
    parent, seeds, parents, parent_inventory = parent_contract(parent_dir, root, formal=not development)
    print("Verifying canonical splits and nine corruption files (read-only)...", flush=True)
    cal, gates, data_hashes, stats = validate_data(data_dir, parent, formal=not development)
    env = environment(device, cfg)
    env["packages"]["matplotlib"] = importlib.metadata.version("matplotlib")
    if not development:
        for key, version in parent["environment"]["packages"].items():
            require(env["packages"].get(key) == version, f"Parent environment version differs: {key}; restore the EXP-03 environment")
        require(env["python"] == parent["environment"]["python"], "Parent Python version differs")
    test_rows = gates["test"]["rows"]
    manifest = {"experiment": "EXP-04", "version": "v0.1", "development": development,
        "protocol": asdict(cfg), "seeds": list(seeds), "parent_fingerprint": parent["fingerprint"],
        "parent_artefacts": parent_inventory, "inputs_sha256": data_hashes,
        "preprocessing": parent["preprocessing"], "environment": env, "source_sha256": source_inventory(root),
        "primary": "raw NLL clean-minus-corrupted; equal severity weights within equal family weights",
        "calibration": "clean-calibration temperature fixed for all conditions; 15-bin top-label ECE",
        "accepted_date": "2026-09-09"}
    fingerprint = json_hash(manifest)
    if folder.exists() and any(folder.iterdir()):
        require(resume and (folder/"manifest.json").exists(), "Output exists: use --resume or a new directory")
        previous = read_json(folder/"manifest.json")
        require(previous == {"fingerprint": fingerprint, **manifest}, "Resume refused: inputs, parent, source, config or environment changed")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder/"manifest.json", {"fingerprint": fingerprint, **manifest})
    atomic_json(folder/"preflight.json", {"status": "PREFLIGHT_PASS", "fingerprint": fingerprint,
        "data_gates": gates, "formal_runs_performed": False, "scope": "Data and provenance verification only"})
    if preflight_only:
        return {"status": "PREFLIGHT_PASS", "fingerprint": fingerprint}
    runs = []
    parent_cfg = Exp03Config(**parent["protocol"])
    base_cfg = BaselineConfig(**parent["model"])
    for seed in seeds:
        models, _ = build_models(stats, parent_cfg, base_cfg, seed)
        for name, model in models.items():
            run_folder = folder/f"seed_{seed}"/name
            if (run_folder/"run.json").exists():
                report = load_run(run_folder, fingerprint)
                require(report["model"] == name and report["seed"] == seed, "Resume run identity mismatch")
                runs.append(report)
                print(f"Reusing verified {name} seed={seed}", flush=True)
                continue
            run_folder.mkdir(parents=True, exist_ok=True)
            checkpoint = parent_dir/f"seed_{seed}"/name/"checkpoint.pt"
            require(file_hash(checkpoint) == parents[name, seed]["artefact_sha256"]["checkpoint.pt"], "Checkpoint changed during evaluation")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            require(state_hash(state) == parents[name, seed]["training"]["selected_state_sha256"], "Selected state hash mismatch")
            model.load_state_dict(state, strict=True)
            require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),
                    "Non-finite selected checkpoint state")
            model.to(device).eval()
            model.requires_grad_(False)
            before = state_hash(model.state_dict())
            cal_arrays = predict(model, cal, cfg, device, diagnostics=False)
            temperature = fit_temperature(cal_arrays["label"], cal_arrays["logits"], cfg)
            save_npz(run_folder/"calibration_predictions.npz", cal_arrays)
            replay = {}
            if seed == seeds[0]:
                repeated = predict(model, cal, cfg, device, diagnostics=False)
                replay["calibration"] = compare_arrays(cal_arrays, repeated, cfg.replay_atol)
                require(fit_temperature(repeated["label"], repeated["logits"], cfg) == temperature,
                        "Temperature fitting replay differs")
            conditions = {}
            for condition in CONDITIONS:
                filename = "test.h5" if condition == "clean" else f"test_{condition}.h5"
                path = data_dir/filename
                require(file_hash(path) == data_hashes[condition], "Test data changed during evaluation")
                data = load_hdf5(path)
                arrays = predict(model, data, cfg, device)
                score = {p: metrics(arrays["label"], arrays["logits"], t) for p, t in
                         (("raw", 1.), ("temperature_scaled", temperature["temperature"]))}
                require(score["raw"]["accuracy"] == score["temperature_scaled"]["accuracy"], "Temperature changed accuracy")
                require(score["raw"]["auroc"] == score["temperature_scaled"]["auroc"], "Temperature changed logit ranking")
                score["diagnostics"] = diagnostic_summary(arrays)
                conditions[condition] = score
                save_npz(run_folder/f"{condition}_predictions.npz", arrays)
                if seed == seeds[0]:
                    replay[condition] = compare_arrays(arrays, predict(model, data, cfg, device), cfg.replay_atol)
                print(f"{name} seed={seed} {condition}: NLL={score['raw']['nll']:.6f} "
                      f"AUROC={score['raw']['auroc']:.6f} T={temperature['temperature']:.5f}", flush=True)
            require(before == state_hash(model.state_dict()), "Model state changed during evaluation")
            artifacts = {p.name: file_hash(p) for p in sorted(run_folder.glob("*.npz"))}
            report = {"status": "RUN_COMPLETE", "fingerprint": fingerprint, "model": name, "seed": seed,
                "selected_state_sha256": before, "temperature": temperature, "conditions": conditions,
                "prediction_count": len(cal["y"])+test_rows*len(CONDITIONS),
                "test_prediction_count": test_rows*len(CONDITIONS), "model_state_unchanged": True,
                "replay": {"performed": seed == seeds[0], "atol": cfg.replay_atol,
                           "maximum_differences": replay, "status": "PASS" if replay else "NOT_SCHEDULED"},
                "artefact_sha256": artifacts}
            signed_json(run_folder/"run.json", report)
            runs.append(read_signed(run_folder/"run.json"))
            model.cpu()
            atomic_json(folder/"progress.json", {"status": "IN_PROGRESS", "completed_runs": len(runs),
                "expected_runs": 3*len(seeds), "fingerprint": fingerprint})
        del models
    for name in MODELS:
        require(next(r for r in runs if r["model"] == name and r["seed"] == seeds[0])["replay"]["status"] == "PASS",
                "Scheduled replay missing")
    summary = {"milestone": "EXP-04", "version": "v0.1", "fingerprint": fingerprint,
        "status": "DEVELOPMENT_ONLY" if development else "EXECUTION_COMPLETE_REVIEW_REQUIRED",
        "created_utc": datetime.now(timezone.utc).isoformat(), "seeds": list(seeds), "data_gates": gates,
        "runs": runs, "aggregate": aggregate(runs, seeds), "primary_run_count": len(runs),
        "primary_test_prediction_count": sum(r["test_prediction_count"] for r in runs),
        "calibration_prediction_count": len(cal["y"])*len(runs),
        "review_flags": [f"Temperature boundary: {r['model']} seed={r['seed']}" for r in runs if r["temperature"]["boundary_solution"]],
        "interpretation": "Execution validity is not superiority. Oriented performance change: higher is better; ECE change: lower is better. Absolute performance must accompany degradation. Retention diagnostics are non-causal; missing/deleted information is not recovered by injectivity.",
        "completion_pending": ["Local acceptance-suite evidence", "Investigator interpretation", "Research-record freeze"]}
    signed_json(folder/"exp04_corruption_retention_v0.1.json", summary)
    summary = read_signed(folder/"exp04_corruption_retention_v0.1.json")
    from ipat.experiments.exp04_report import write_report
    write_report(summary, folder)
    atomic_json(folder/"progress.json", {"status": summary["status"], "fingerprint": fingerprint,
        "completed_runs": len(runs), "report": "exp04_report.html"})
    return summary
