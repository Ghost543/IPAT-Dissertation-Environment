"""EXP-04 v0.1: accepted 9 September 2026. Frozen MATH-07/08 metrics.

No model training. Numerical optimiser bounds are engineering defaults; boundary
solutions are reported for review. Inference is conditional on fixed test data.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from ipat.experiments.exp03_contract import (
    MODELS, SEEDS, require, describe, signed_rank_exact, json_hash,
)

FAMILIES = {
    "missing": (.10, .20, .40), "delete": (.10, .25),
    "noise": (.10, .25), "timejitter": (.10, .25),
}
CONDITIONS = ("clean",) + tuple(f"{c}_{round(s*100):02d}" for c, levels in FAMILIES.items() for s in levels)
PANELS = ("raw", "temperature_scaled")
METRICS = ("nll", "auroc", "accuracy", "brier", "ece", "ece_equal_mass")
PARENT_FINGERPRINT = "fd4c3146f6f8dd893f2129d3fe5663426eb566ab9d11f8d3ecd50cf987d6286f"
PARENT_SUMMARY_SHA256 = "fc0d66ec89269c741f2be16ffbe31bab2837511621a33d42d55c3290fa11a623"


@dataclass(frozen=True)
class Exp04Config:
    batch_size: int = 64
    cpu_threads: int = 1
    temperature_min: float = .01
    temperature_max: float = 100.
    temperature_xatol: float = 1e-8
    temperature_nll_atol: float = 1e-10
    replay_atol: float = 0.
    ece_bins: int = 15

    def validate(self):
        for name in ("batch_size", "cpu_threads", "ece_bins"):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0, f"Invalid {name}")
        require(self.ece_bins == 15, "Frozen MATH-08 requires 15 ECE bins")
        for name in ("temperature_min", "temperature_max", "temperature_xatol", "temperature_nll_atol", "replay_atol"):
            require(np.isfinite(getattr(self, name)), f"Non-finite {name}")
        require(0 < self.temperature_min < 1 < self.temperature_max, "Temperature bounds must bracket 1")
        require(self.temperature_xatol > 0 and self.temperature_nll_atol >= 0 and self.replay_atol >= 0,
                "Invalid numerical tolerance")


def checked_inputs(y, logits):
    y, logits = np.asarray(y), np.asarray(logits, dtype=np.float64)
    require(y.ndim == logits.ndim == 1 and y.shape == logits.shape and len(y) > 0, "Metric shape mismatch")
    require(np.isfinite(logits).all() and np.isin(y, (0, 1)).all(), "Invalid metric inputs")
    return y.astype(np.float64), logits


def nll(y, logits):
    y, logits = checked_inputs(y, logits)
    return float(np.logaddexp(0., (1.-2.*y)*logits).mean())


def calibration_bins(y, logits, bins=15, equal_mass=False):
    y, logits = checked_inputs(y, logits)
    require(type(bins) is int and bins > 0, "Invalid bin count")
    confidence = expit(np.abs(logits))
    correct = ((logits >= 0) == y).astype(np.float64)
    if equal_mass:
        # Deterministic tie handling: stable input/sequence order; ties may span bins.
        groups = np.array_split(np.argsort(confidence, kind="stable"), bins)
    else:
        edges = np.linspace(0., 1., bins+1)
        index = np.minimum(np.searchsorted(edges, confidence, side="right")-1, bins-1)
        groups = [np.flatnonzero(index == i) for i in range(bins)]
    records = []
    for i, rows in enumerate(groups):
        records.append({"bin": i, "count": len(rows),
            "confidence": float(confidence[rows].mean()) if len(rows) else None,
            "accuracy": float(correct[rows].mean()) if len(rows) else None,
            "lower": None if equal_mass else i/bins,
            "upper": None if equal_mass else (i+1)/bins})
    ece = sum(r["count"]*abs(r["accuracy"]-r["confidence"]) for r in records if r["count"])/len(y)
    return {"ece": float(ece), "bins": records, "scheme": "equal_mass_stable_ties" if equal_mass else "equal_width"}


def metrics(y, logits, temperature=1.):
    y, logits = checked_inputs(y, logits)
    require(set(np.unique(y)) == {0., 1.}, "Both classes required for AUROC")
    require(np.isfinite(temperature) and temperature > 0, "Invalid temperature")
    scaled = logits/temperature
    require(np.isfinite(scaled).all(), "Non-finite scaled logits")
    p = expit(scaled)
    width = calibration_bins(y, scaled)
    mass = calibration_bins(y, scaled, equal_mass=True)
    return {"nll": nll(y, scaled), "auroc": float(roc_auc_score(y, scaled)),
            "accuracy": float(np.mean((scaled >= 0) == y)), "brier": float(np.mean((p-y)**2)),
            "ece": width["ece"], "ece_equal_mass": mass["ece"],
            "reliability": width["bins"], "reliability_equal_mass": mass["bins"]}


def fit_temperature(y, logits, cfg):
    cfg.validate()
    y, logits = checked_inputs(y, logits)
    before = nll(y, logits)
    low, high = np.log([cfg.temperature_min, cfg.temperature_max])
    result = minimize_scalar(lambda log_t: nll(y, logits/np.exp(log_t)),
        bounds=(low, high), method="bounded", options={"xatol": cfg.temperature_xatol, "maxiter": 1000})
    require(result.success and np.isfinite(result.fun), "Temperature optimiser failed")
    candidates = [(before, 1.), (float(result.fun), float(np.exp(result.x))),
                  (nll(y, logits/cfg.temperature_min), cfg.temperature_min),
                  (nll(y, logits/cfg.temperature_max), cfg.temperature_max)]
    after, temperature = min(candidates, key=lambda p: p[0])
    require(after <= before + cfg.temperature_nll_atol, "Temperature calibration worsened fitting NLL")
    require(np.array_equal(logits >= 0, logits/temperature >= 0), "Temperature changed decisions")
    boundary = temperature in (cfg.temperature_min, cfg.temperature_max)
    return {"temperature": temperature, "clean_cal_nll_raw": before, "clean_cal_nll_scaled": after,
            "boundary_solution": boundary, "optimiser_success": True, "calibration_rows": len(y),
            "fit_scope": "clean calibration only; fixed for all test conditions",
            "bounds": [cfg.temperature_min, cfg.temperature_max]}


def macro_change(conditions, panel, metric):
    require(set(conditions) == set(CONDITIONS), "Incomplete/extra corruption conditions")
    clean = conditions["clean"][panel][metric]
    # Calibration change has its own frozen sign: positive is deterioration.
    direction = 1 if metric in ("auroc", "accuracy", "ece", "ece_equal_mass") else -1
    per_family = {c: float(np.mean([direction*(conditions[f"{c}_{round(s*100):02d}"][panel][metric]-clean)
                                  for s in levels])) for c, levels in FAMILIES.items()}
    return {"mean": float(np.mean(list(per_family.values()))), "per_family": per_family,
            "positive_means": "deterioration" if metric.startswith("ece") else "improvement"}


def aggregate(runs, seeds):
    seeds = tuple(seeds)
    lookup = {(r["model"], r["seed"]): r for r in runs}
    require(len(lookup) == len(runs) == 3*len(seeds), "Missing/duplicate runs")
    require(set(lookup) == {(m, s) for m in MODELS for s in seeds}, "Unexpected model/seed pair")
    output = {"models": {}, "paired_primary_contrasts": {}}
    for model in MODELS:
        items = [lookup[model, s] for s in seeds]
        output["models"][model] = {
            "absolute": {c: {p: {metric: describe([r["conditions"][c][p][metric] for r in items])
                                  for metric in METRICS} for p in PANELS} for c in CONDITIONS},
            "macro_change": {p: {metric: describe([macro_change(r["conditions"], p, metric)["mean"]
                                                   for r in items]) for metric in METRICS} for p in PANELS}}
    for a, b in (("retention", "baseline"), ("full_ipat", "retention"), ("full_ipat", "baseline")):
        values = [macro_change(lookup[a, s]["conditions"], "raw", "nll")["mean"]
                  - macro_change(lookup[b, s]["conditions"], "raw", "nll")["mean"] for s in seeds]
        output["paired_primary_contrasts"][f"{a}_minus_{b}"] = {
            **describe(values), "positive_favours": a, "secondary_signed_rank": signed_rank_exact(values)}
    ordered = sorted(output["paired_primary_contrasts"].values(), key=lambda v: v["secondary_signed_rank"]["pvalue"])
    adjusted = 0.
    for i, item in enumerate(ordered):
        adjusted = max(adjusted, min(1., (3-i)*item["secondary_signed_rank"]["pvalue"]))
        item["secondary_signed_rank"]["holm_pvalue"] = adjusted
    return output


def exact_data(actual, expected, context):
    require(json_hash(actual.metadata) == json_hash(expected.metadata), f"{context}: metadata mismatch")
    require(set(actual.arrays) == set(expected.arrays), f"{context}: array fields mismatch")
    for key in expected.arrays:
        a, b = actual[key], expected[key]
        require(a.dtype == b.dtype and np.array_equal(a, b), f"{context}: frozen generator mismatch in {key}")


def make_corruption(clean, condition, train_scale):
    from ipat.data.synthetic import (apply_missingness_corruption, apply_event_deletion,
                                    apply_numerical_noise, apply_temporal_jitter)
    family, percent = condition.rsplit("_", 1)
    severity = int(percent)/100
    require(condition in CONDITIONS[1:], "Unknown frozen corruption")
    if family == "missing":
        return apply_missingness_corruption(clean, severity, 10100+int(percent))
    if family == "delete":
        return apply_event_deletion(clean, severity, 10200+int(percent))
    if family == "noise":
        return apply_numerical_noise(clean, severity, train_scale, 10300+int(percent))
    return apply_temporal_jitter(clean, severity, 10400+int(percent))


def validate_data(data_dir, parent, formal=True):
    from pathlib import Path
    from ipat.data.synthetic import IPATSyntheticConfig, generate_split, load_hdf5, compute_train_numeric_scale, summarize
    from ipat.training.preprocessing import fit_baseline_preprocessing_stats
    from ipat.experiments.exp03_contract import file_hash, sequence_fingerprints
    root = Path(data_dir)
    clean_splits, hashes, gate = {}, {}, {}
    for name, count in (("train", 8000), ("val", 2000), ("cal", 2000), ("test", 4000)):
        path = root/f"{name}.h5"
        data = load_hdf5(path)
        cfg = IPATSyntheticConfig(**data.metadata["config"])
        if formal:
            require(asdict(cfg) == asdict(IPATSyntheticConfig()) and len(data["y"]) == count,
                    f"Expected canonical {name} configuration/count")
        exact_data(data, generate_split(cfg, name), name)
        require(set(np.unique(data["y"])) == {0, 1}, f"{name} requires both classes")
        hashes[name] = file_hash(path)
        if name in ("train", "val"):
            require(hashes[name] == parent["inputs_sha256"][name], f"Parent {name} bytes changed")
        clean_splits[name] = data
        gate[name] = {"rows": len(data["y"]), "exact_generator_reproduction": True}
    config_hashes = {json_hash(d.metadata["config"]) for d in clean_splits.values()}
    require(len(config_hashes) == 1, "Split configurations differ")
    fingerprints = [sequence_fingerprints(d) for d in clean_splits.values()]
    for i in range(len(fingerprints)):
        for j in range(i):
            require(not fingerprints[i].intersection(fingerprints[j]), "Partition observation overlap")
    stats = fit_baseline_preprocessing_stats(clean_splits["train"])
    require(json_hash(stats.to_dict()) == json_hash(parent["preprocessing"]), "Parent preprocessing differs")
    scale = compute_train_numeric_scale(clean_splits["train"])
    clean = clean_splits["test"]
    gate["conditions"] = {"clean": summarize(clean)}
    hashes["clean"] = hashes.pop("test")
    for condition in CONDITIONS[1:]:
        path = root/f"test_{condition}.h5"
        data = load_hdf5(path)
        exact_data(data, make_corruption(clean, condition, scale), condition)
        gate["conditions"][condition] = {**summarize(data), "corruption": data.metadata["corruption"],
            "sequences_without_signal": int((data["signal_mask"].sum(1) == 0).sum()),
            "deleted_events": int(clean["length"].sum()-data["length"].sum())}
        hashes[condition] = file_hash(path)
    gate.update(status="PASS", no_exact_partition_overlap=True,
                corruption_verification="Exact in-memory reproduction; existing files never rewritten",
                deletion_semantics="Surviving source timestamp/delta fields preserved; positions compacted")
    return clean_splits["cal"], gate, hashes, stats
