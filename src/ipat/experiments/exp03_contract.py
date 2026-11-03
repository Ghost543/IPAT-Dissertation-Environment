"""EXP-03 data and metric contract. No model fitting or PyTorch dependency.

The primary endpoint and ten paired training seeds were approved on 8 September 2026.
This module validates the frozen DATA-01 artefacts; it never regenerates or repairs them.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata, t
from sklearn.metrics import roc_auc_score

VARIANTS = ("early", "middle", "late")
MODELS = ("baseline", "retention", "full_ipat")
SEEDS = tuple(range(20260908, 20260918))
EVENT_FIELDS = (
    "x_full", "x_obs", "missing_mask", "latent_u", "event_type", "nuisance_cat",
    "delta", "signal_mask", "contribution",
)


def require(condition: bool, message: str) -> None:
    # Unlike assert, acceptance gates must survive python -O.
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Exp03Config:
    seeds: tuple[int, ...] = SEEDS
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    patience: int = 5
    min_delta: float = 0.0
    gamma: float = 1.0
    identity_floor: float = 0.25
    asinh_scales: tuple[float, ...] = (0.25, 1.0, 4.0)
    cpu_threads: int = 1
    replay_atol: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "asinh_scales", tuple(self.asinh_scales))

    def validate(self, formal: bool = True) -> None:
        require(len(self.seeds) > 0 and len(set(self.seeds)) == len(self.seeds),
                "Training seeds must be non-empty and unique")
        require(all(isinstance(s, int) and 0 <= s < 2**32 for s in self.seeds), "Invalid seed")
        if formal:
            require(self.seeds == SEEDS, "Formal v0.1 requires the fixed ten-seed schedule")
        for name in ("epochs", "batch_size", "patience", "cpu_threads"):
            require(isinstance(getattr(self, name), int) and getattr(self, name) > 0,
                    f"{name} must be a positive integer")
        for name in ("learning_rate", "weight_decay", "min_delta", "gamma", "replay_atol"):
            value = getattr(self, name)
            require(np.isfinite(value) and value >= 0, f"Invalid {name}")
        require(self.learning_rate > 0, "learning_rate must be positive")
        require(0 < self.identity_floor < 1, "Invalid identity floor")
        require(len(self.asinh_scales) > 0 and len(set(self.asinh_scales)) == len(self.asinh_scales)
                and all(np.isfinite(a) and a > 0 for a in self.asinh_scales), "Invalid scales")


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def basic_data_gate(data) -> None:
    from ipat.data.synthetic import recompute_truth
    a = data.arrays
    n = len(a["y"])
    require(n > 0 and all(len(v) == n for v in a.values()), "Inconsistent row counts")
    require(len(np.unique(a["sequence_id"])) == n, "Duplicate sequence IDs")
    require(set(np.unique(a["y"]).tolist()) == {0, 1}, "Both binary classes are required")
    valid = a["valid_mask"]
    require(np.all(valid == (np.arange(valid.shape[1])[None, :] < a["length"][:, None])),
            "Padding must be a contiguous suffix matching length")
    require(np.all(a["length"] > 0), "Empty sequence")
    require(not np.any(a["signal_mask"] & ~valid), "Padded signal")
    require(np.all(a["signal_mask"].sum(1) == data.metadata["config"]["n_signal"]),
            "Signal count mismatch")
    require(np.all(a["signal_mask"] == (valid & np.isin(a["event_type"], [0, 1]))),
            "Signal mask/event type mismatch")
    for key, value in a.items():
        if np.issubdtype(value.dtype, np.number):
            require(np.isfinite(value).all(), f"Non-finite data: {key}")
    require(np.all(a["delta"][valid] > 0), "Non-positive event gap")
    require(np.all((a["p_true"] >= 0) & (a["p_true"] <= 1)), "Invalid true probability")
    observed = valid[..., None] & ~a["missing_mask"]
    require(np.array_equal(a["x_obs"][observed], a["x_full"][observed]), "Observed data altered")
    require(np.all(a["x_obs"][~observed] == 0), "Missing/padded observation not zero")
    for key in ("x_full", "latent_u", "delta", "timestamp", "contribution", "missing_mask"):
        require(np.all(a[key][~valid] == 0), f"Invalid padded {key}")
    for key in ("event_type", "nuisance_cat", "position"):
        require(np.all(a[key][~valid] == -1), f"Invalid padded {key}")
    for row, length in enumerate(a["length"]):
        length = int(length)
        require(np.array_equal(a["position"][row, :length], np.arange(length)), "Bad positions")
        require(np.allclose(a["timestamp"][row, :length],
                            np.cumsum(a["delta"][row, :length]), rtol=1e-12, atol=1e-12),
                "Timestamps not reconstructed from gaps")
    contribution, logits, probability = recompute_truth(data)
    for name, calculated in (("contribution", contribution), ("sequence_logit", logits),
                             ("p_true", probability)):
        require(np.allclose(a[name], calculated, rtol=1e-12, atol=1e-12), f"Truth mismatch: {name}")


def triplet_rows(sweep) -> np.ndarray:
    """Return K x 3 row indices; input order need not be grouped by triplet."""
    ids = sweep["triplet_id"]
    labels = sweep["variant"].astype("U")
    grouped = []
    for key in np.unique(ids):
        rows = np.flatnonzero(ids == key)
        require(len(rows) == 3 and set(labels[rows]) == set(VARIANTS), "Incomplete triplet")
        grouped.append([int(rows[np.flatnonzero(labels[rows] == v)[0]]) for v in VARIANTS])
    return np.asarray(grouped, dtype=np.int64)


def validate_sweep(sweep, expected_triplets: int | None = 1000) -> dict:
    from ipat.data.synthetic import IPATSyntheticConfig, generate_position_sweep
    require(sweep.metadata.get("profile") == "position_sweep", "Wrong sweep profile")
    cfg = IPATSyntheticConfig(**sweep.metadata["config"])
    require(cfg.temporal_signal_weight == cfg.position_sweep_temporal_weight == 0,
            "Label-invariant sweep must have zero temporal label weight")
    basic_data_gate(sweep)
    groups = triplet_rows(sweep)
    if expected_triplets is not None:
        require(len(groups) == expected_triplets, "Wrong number of triplets (debug data?)")
    for rows in groups:
        for name in ("y", "p_true", "sequence_logit", "length"):
            require(np.all(sweep[name][rows] == sweep[name][rows[0]]), f"Unpaired {name}")
        length = int(sweep["length"][rows[0]])
        targets = np.array([round(f * (length - 1)) for f in (0.1, 0.5, 0.9)])
        require(np.array_equal(sweep["probe_target_position"][rows], targets), "Bad probe targets")
        for key in EVENT_FIELDS:
            first = sweep[key][rows[0], :length]
            probe = first[targets[0]]
            remainder = np.delete(first, targets[0], axis=0)
            for row, target in zip(rows, targets):
                current = sweep[key][row, :length]
                require(np.array_equal(current[target], probe), f"Probe identity changed: {key}")
                require(np.array_equal(np.delete(current, target, axis=0), remainder),
                        f"Non-probe order/content changed: {key}")
        require(sweep["signal_mask"][rows[0], targets[0]], "Probe is not informative")
    # Exact regeneration checks selection of the original first signal as well as reproducibility.
    regenerated = generate_position_sweep(cfg, len(groups), seed=sweep.metadata["seed"])
    require(set(sweep.arrays) == set(regenerated.arrays), "Unexpected sweep fields")
    source_order = np.argsort(sweep["sequence_id"])
    reference_order = np.argsort(regenerated["sequence_id"])
    for name in sweep.arrays:
        require(np.array_equal(sweep[name][source_order], regenerated[name][reference_order]),
                f"Sweep differs from frozen generator: {name}")
    return {"status": "PASS", "triplets": len(groups), "sequences": len(groups) * 3,
            "exact_generator_reproduction": True, "temporal_label_weight": 0.0}


def sequence_fingerprints(data) -> set[str]:
    """Observable ordered event fingerprints; exclude labels and record IDs."""
    fields = ("x_obs", "missing_mask", "event_type", "nuisance_cat", "delta", "timestamp")
    output = set()
    for row, length in enumerate(data["length"]):
        h = hashlib.sha256()
        for key in fields:
            h.update(array_hash(data[key][row, :int(length)]).encode())
        output.add(h.hexdigest())
    return output


def validate_inputs(train, val, sweep, formal=True) -> dict:
    from ipat.data.synthetic import IPATSyntheticConfig
    for name, data, size in (("train", train, 8000), ("val", val, 2000)):
        require(data.metadata.get("split") == name and data.metadata.get("profile") == "canonical",
                f"Expected canonical {name}")
        require("corruption" not in data.metadata, "Corrupted input is not permitted")
        basic_data_gate(data)
        if formal:
            require(len(data["y"]) == size, f"Canonical {name} must contain {size} rows")
            require(data.metadata["config"] == asdict(IPATSyntheticConfig()),
                    f"Non-canonical {name} configuration")
            require(data.metadata.get("label_seed_offset", 0) == 0, "Altered label seed")
    expected = asdict(IPATSyntheticConfig(**train.metadata["config"]))
    expected["temporal_signal_weight"] = 0.0
    require(sweep.metadata["config"] == expected, "Sweep configuration differs beyond temporal weight")
    require(val.metadata["config"] == train.metadata["config"], "Train/validation config mismatch")
    require(sweep.metadata["seed"] == train.metadata["config"]["master_seed"] + 77,
            "Unexpected DATA-01 sweep seed")
    report = validate_sweep(sweep, 1000 if formal else None)
    fingerprints = [sequence_fingerprints(d) for d in (train, val, sweep)]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        require(not (fingerprints[i] & fingerprints[j]), "Exact input overlap between partitions")
    report["no_exact_partition_overlap"] = True
    report["train_temporal_label_weight"] = train.metadata["config"]["temporal_signal_weight"]
    report["limitation"] = "Canonical temporal-weight 0.4 training; weight 0.0 sweep; not a pure in-distribution positional ablation."
    return report


def binary_metrics(y, logits) -> dict:
    y, logits = np.asarray(y), np.asarray(logits, dtype=np.float64)
    require(y.ndim == logits.ndim == 1 and y.shape == logits.shape, "Metric shape mismatch")
    require(np.isfinite(logits).all() and set(np.unique(y)) == {0, 1}, "Invalid metric inputs")
    probability = expit(logits)
    return {"auroc": float(roc_auc_score(y, probability)),
            "accuracy": float(np.mean((logits >= 0) == y)),
            "nll": float(np.mean(np.logaddexp(0, logits) - y * logits)),
            "brier": float(np.mean((probability - y)**2))}


def summarise_predictions(sweep, logits, diagnostics: dict | None = None) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    require(logits.shape == sweep["y"].shape and np.isfinite(logits).all(), "Invalid predictions")
    groups = triplet_rows(sweep)
    per_position = {v: binary_metrics(sweep["y"][groups[:, i]], logits[groups[:, i]])
                    for i, v in enumerate(VARIANTS)}
    names = tuple(per_position["early"])
    report = {
        "prediction_count": len(logits),
        "primary_probability_range": float(np.ptp(expit(logits)[groups], axis=1).mean()),
        "mean_logit_range": float(np.ptp(logits[groups], axis=1).mean()),
        "per_position": per_position,
        "macro": {k: float(np.mean([per_position[v][k] for v in VARIANTS])) for k in names},
        "worst_position": {k: float((min if k in ("auroc", "accuracy") else max)(
            per_position[v][k] for v in VARIANTS)) for k in names},
    }
    if diagnostics:
        require(all(np.asarray(a).shape == logits.shape and np.isfinite(a).all()
                    for a in diagnostics.values()), "Invalid diagnostic arrays")
        report["diagnostics_by_position"] = {
            v: {k: float(np.mean(a[groups[:, i]])) for k, a in diagnostics.items()}
            for i, v in enumerate(VARIANTS)
        }
        if "probe_retention" in diagnostics:
            report["probe_retention_range"] = float(np.ptp(
                diagnostics["probe_retention"][groups], axis=1).mean())
    return report


def describe(values) -> dict:
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and len(a) > 0 and np.isfinite(a).all(), "Invalid seed values")
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else None
    half = float(t.ppf(0.975, len(a)-1) * sd / np.sqrt(len(a))) if sd is not None else None
    return {"values": a.tolist(), "mean": mean, "median": float(np.median(a)), "sd": sd,
            "mean_95ci_t": [mean-half, mean+half] if half is not None else None,
            "ci_scope": "Approximate t interval over training seeds, conditional on fixed datasets; no dataset-sampling uncertainty."}


def signed_rank_exact(differences) -> dict:
    """Exhaustive sign-flip null on midranks, valid with ties; discard exact zeros.

    Secondary inference assumes independent, symmetric seed differences under the null.
    Ten seeds -> at most 1024 assignments. Do not mislabel scipy 'exact' with ties.
    """
    d = np.asarray(differences, dtype=np.float64)
    require(np.isfinite(d).all() and len(d) <= 20, "Invalid sign-rank inputs")
    d = d[d != 0]
    if len(d) == 0:
        return {"pvalue": 1.0, "nonzero_pairs": 0, "assignments": 1}
    ranks = rankdata(np.abs(d), method="average")
    observed = abs(float(np.dot(np.sign(d), ranks)))
    extreme = sum(abs(np.dot(sign, ranks)) >= observed
                  for sign in itertools.product((-1, 1), repeat=len(d)))
    return {"pvalue": extreme / 2**len(d), "nonzero_pairs": len(d), "assignments": 2**len(d)}


def aggregate_runs(runs: list[dict], seeds) -> dict:
    lookup = {(r["model"], r["seed"]): r["metrics"] for r in runs}
    require(len(lookup) == len(runs) == 3 * len(seeds), "Missing or duplicate model/seed runs")
    require(set(lookup) == {(m, s) for m in MODELS for s in seeds}, "Unexpected model/seed set")
    result = {"models": {}, "paired_primary_contrasts": {}}
    for model in MODELS:
        get = lambda key: [lookup[model, s][key] for s in seeds]
        result["models"][model] = {
            "primary_probability_range": describe(get("primary_probability_range")),
            "mean_logit_range": describe(get("mean_logit_range")),
            "macro": {k: describe([lookup[model, s]["macro"][k] for s in seeds])
                      for k in ("auroc", "accuracy", "nll", "brier")},
            "worst_position": {k: describe([lookup[model, s]["worst_position"][k] for s in seeds])
                               for k in ("auroc", "accuracy", "nll", "brier")},
            "per_position": {v: {k: describe([lookup[model, s]["per_position"][v][k]
                                                 for s in seeds])
                                 for k in ("auroc", "accuracy", "nll", "brier")}
                             for v in VARIANTS},
        }
    for a, b in (("retention", "baseline"), ("full_ipat", "retention"), ("full_ipat", "baseline")):
        differences = [lookup[a, s]["primary_probability_range"] -
                       lookup[b, s]["primary_probability_range"] for s in seeds]
        result["paired_primary_contrasts"][f"{a}_minus_{b}"] = {
            **describe(differences), "negative_favours": a,
            "secondary_signed_rank": signed_rank_exact(differences)}
    # Holm correction across the three secondary contrasts.
    ordered = sorted(result["paired_primary_contrasts"].values(),
                     key=lambda x: x["secondary_signed_rank"]["pvalue"])
    adjusted = 0.0
    for i, item in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, (3-i) * item["secondary_signed_rank"]["pvalue"]))
        item["secondary_signed_rank"]["holm_pvalue"] = adjusted
    return result
