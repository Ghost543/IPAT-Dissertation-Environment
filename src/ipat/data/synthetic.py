from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

EVENT_TYPE_TO_ID = {"A": 0, "B": 1, "D": 2, "N": 3}
SPLIT_CODES = {"train": 11, "val": 23, "cal": 37, "test": 53}


@dataclass(frozen=True)
class IPATSyntheticConfig:
    dataset_id: str = "IPAT-SYN-v0.1"
    master_seed: int = 20260831
    n_train: int = 8000
    n_val: int = 2000
    n_cal: int = 2000
    n_test: int = 4000
    t_min: int = 24
    t_max: int = 64
    n_signal: int = 3
    n_num: int = 6
    n_nuisance_categories: int = 8
    p_missing_clean: float = 0.05
    p_zero_x6: float = 0.60
    gap_log_mean: float = 0.0
    gap_log_sd: float = 0.35
    temporal_signal_weight: float = 0.4
    position_sweep_temporal_weight: float = 0.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IPATSyntheticConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**payload.get("synthetic", payload))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def n_for_split(self, split: str) -> int:
        values = {"train": self.n_train, "val": self.n_val, "cal": self.n_cal, "test": self.n_test}
        if split not in values:
            raise ValueError(f"Unknown split {split!r}")
        return values[split]


@dataclass
class SplitData:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def copy(self) -> "SplitData":
        return SplitData(
            arrays={k: np.array(v, copy=True) for k, v in self.arrays.items()},
            metadata=json.loads(json.dumps(self.metadata)),
        )


def _rng(seed: int, *codes: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *codes]))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _phi_asinh(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.arcsinh(alpha * x) / alpha


def _build_single_sequence(cfg: IPATSyntheticConfig, rng: np.random.Generator, temporal_weight: float) -> dict[str, Any]:
    t = int(rng.integers(cfg.t_min, cfg.t_max + 1))
    signal_idx = np.sort(rng.choice(t, size=cfg.n_signal, replace=False))

    u = rng.uniform(-2.0, 2.0, size=(t, 3)).astype(np.float64)
    x = np.empty((t, cfg.n_num), dtype=np.float64)
    x[:, 0] = np.sinh(0.5 * u[:, 0]) / 0.5
    x[:, 1] = u[:, 1]
    x[:, 2] = np.sinh(2.0 * u[:, 2]) / 2.0
    x[:, 3] = rng.normal(0.0, 1.0, size=t)
    x[:, 4] = rng.standard_t(df=5, size=t) / np.sqrt(5.0 / 3.0)
    zero_gate = rng.random(t) < cfg.p_zero_x6
    x[:, 5] = np.where(zero_gate, 0.0, rng.normal(0.0, 1.0, size=t))

    event_type = rng.choice([EVENT_TYPE_TO_ID["D"], EVENT_TYPE_TO_ID["N"]], size=t).astype(np.int16)
    event_type[signal_idx] = rng.choice([EVENT_TYPE_TO_ID["A"], EVENT_TYPE_TO_ID["B"]], size=cfg.n_signal).astype(np.int16)
    nuisance_cat = rng.integers(0, cfg.n_nuisance_categories, size=t, dtype=np.int16)

    delta = rng.lognormal(cfg.gap_log_mean, cfg.gap_log_sd, size=t).astype(np.float64)
    timestamp = np.cumsum(delta)
    position = np.arange(t, dtype=np.int16)

    signal_mask = np.zeros(t, dtype=bool)
    signal_mask[signal_idx] = True
    sign = np.where(event_type == EVENT_TYPE_TO_ID["A"], 1.0, -1.0)
    raw_contribution = (
        0.9 * u[:, 0]
        - 0.7 * u[:, 1]
        + 0.6 * u[:, 2]
        + 0.5 * sign
        + temporal_weight * np.tanh(np.log(delta))
    )
    contribution = signal_mask.astype(np.float64) * raw_contribution
    sequence_logit = float(contribution.sum() / np.sqrt(cfg.n_signal))
    p_true = float(_sigmoid(np.array([sequence_logit]))[0])

    missing = rng.random((t, cfg.n_num)) < cfg.p_missing_clean
    x_obs = x.copy()
    x_obs[missing] = 0.0

    return {
        "length": t,
        "x_full": x,
        "x_obs": x_obs,
        "missing_mask": missing,
        "latent_u": u,
        "event_type": event_type,
        "nuisance_cat": nuisance_cat,
        "delta": delta,
        "timestamp": timestamp,
        "position": position,
        "signal_mask": signal_mask,
        "contribution": contribution,
        "sequence_logit": sequence_logit,
        "p_true": p_true,
    }


def _allocate(n: int, cfg: IPATSyntheticConfig) -> dict[str, np.ndarray]:
    bt = (n, cfg.t_max)
    return {
        "sequence_id": np.empty(n, dtype="S40"),
        "length": np.zeros(n, dtype=np.int16),
        "x_full": np.zeros((n, cfg.t_max, cfg.n_num), dtype=np.float64),
        "x_obs": np.zeros((n, cfg.t_max, cfg.n_num), dtype=np.float64),
        "missing_mask": np.zeros((n, cfg.t_max, cfg.n_num), dtype=bool),
        "latent_u": np.zeros((n, cfg.t_max, 3), dtype=np.float64),
        "event_type": np.full(bt, -1, dtype=np.int16),
        "nuisance_cat": np.full(bt, -1, dtype=np.int16),
        "delta": np.zeros(bt, dtype=np.float64),
        "timestamp": np.zeros(bt, dtype=np.float64),
        "position": np.full(bt, -1, dtype=np.int16),
        "valid_mask": np.zeros(bt, dtype=bool),
        "signal_mask": np.zeros(bt, dtype=bool),
        "contribution": np.zeros(bt, dtype=np.float64),
        "sequence_logit": np.zeros(n, dtype=np.float64),
        "p_true": np.zeros(n, dtype=np.float64),
        "y": np.zeros(n, dtype=np.int8),
    }


def _write(arrays: dict[str, np.ndarray], row: int, record: dict[str, Any], sequence_id: str) -> None:
    t = int(record["length"])
    arrays["sequence_id"][row] = sequence_id.encode("utf-8")
    arrays["length"][row] = t
    arrays["valid_mask"][row, :t] = True
    for name in (
        "x_full", "x_obs", "missing_mask", "latent_u", "event_type", "nuisance_cat",
        "delta", "timestamp", "position", "signal_mask", "contribution",
    ):
        arrays[name][row, :t] = record[name]
    arrays["sequence_logit"][row] = record["sequence_logit"]
    arrays["p_true"][row] = record["p_true"]


def generate_split(cfg: IPATSyntheticConfig, split: str, n_sequences: int | None = None, *, label_seed_offset: int = 0) -> SplitData:
    if split not in SPLIT_CODES:
        raise ValueError(f"Unknown split {split!r}")
    n = cfg.n_for_split(split) if n_sequences is None else int(n_sequences)
    if n <= 0:
        raise ValueError("n_sequences must be positive")
    if cfg.n_signal > cfg.t_min:
        raise ValueError("n_signal cannot exceed t_min")

    feature_rng = _rng(cfg.master_seed, SPLIT_CODES[split], 1001)
    label_rng = _rng(cfg.master_seed, SPLIT_CODES[split], 2003, label_seed_offset)
    arrays = _allocate(n, cfg)

    for i in range(n):
        rec = _build_single_sequence(cfg, feature_rng, cfg.temporal_signal_weight)
        _write(arrays, i, rec, f"{cfg.dataset_id}:{split}:{i:07d}")

    arrays["y"] = (label_rng.random(n) < arrays["p_true"]).astype(np.int8)
    metadata = {
        "dataset_id": cfg.dataset_id,
        "profile": "canonical",
        "split": split,
        "master_seed": cfg.master_seed,
        "label_seed_offset": label_seed_offset,
        "config": cfg.to_dict(),
        "event_type_mapping": EVENT_TYPE_TO_ID,
    }
    return SplitData(arrays, metadata)


def recompute_truth(data: SplitData, cfg: IPATSyntheticConfig | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = cfg or IPATSyntheticConfig(**data.metadata["config"])
    valid = data["valid_mask"]
    signal = data["signal_mask"]
    u = data["latent_u"]
    event_type = data["event_type"]
    delta = np.where(valid, data["delta"], 1.0)
    sign = np.where(event_type == EVENT_TYPE_TO_ID["A"], 1.0, -1.0)
    contribution = signal.astype(np.float64) * (
        0.9 * u[..., 0]
        - 0.7 * u[..., 1]
        + 0.6 * u[..., 2]
        + 0.5 * sign
        + cfg.temporal_signal_weight * np.tanh(np.log(delta))
    )
    contribution[~valid] = 0.0
    logits = contribution.sum(axis=1) / np.sqrt(cfg.n_signal)
    return contribution, logits, _sigmoid(logits)


def verify_inverse_feature_family(data: SplitData) -> dict[str, float]:
    valid = data["valid_mask"]
    x, u = data["x_full"], data["latent_u"]
    return {
        "x1_alpha_0.5": float(np.max(np.abs(_phi_asinh(x[..., 0][valid], 0.5) - u[..., 0][valid]))),
        "x2_identity": float(np.max(np.abs(x[..., 1][valid] - u[..., 1][valid]))),
        "x3_alpha_2": float(np.max(np.abs(_phi_asinh(x[..., 2][valid], 2.0) - u[..., 2][valid]))),
    }


def compute_train_numeric_scale(train: SplitData) -> np.ndarray:
    valid = train["valid_mask"][..., None]
    observed = valid & (~train["missing_mask"])
    scales = np.empty(train["x_obs"].shape[-1], dtype=np.float64)
    for j in range(scales.size):
        vals = train["x_obs"][..., j][observed[..., j]]
        if vals.size < 2:
            raise ValueError(f"Too few observed values for feature {j}")
        scales[j] = np.std(vals, ddof=0)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Training scales must be finite and positive")
    return scales


def apply_missingness_corruption(clean: SplitData, target_rate: float, seed: int) -> SplitData:
    base_rate = float(clean.metadata["config"]["p_missing_clean"])
    if not (base_rate <= target_rate < 1.0):
        raise ValueError(f"target_rate must be in [{base_rate},1)")
    p_add = (target_rate - base_rate) / (1.0 - base_rate)
    out = clean.copy()
    rng = _rng(seed, 3101)
    valid = out["valid_mask"][..., None]
    original = out["missing_mask"]
    eligible = valid & (~original)
    add = (rng.random(original.shape) < p_add) & eligible
    combined = original | add
    out.arrays["missing_mask"] = combined
    out.arrays["x_obs"] = out["x_full"].copy()
    out.arrays["x_obs"][combined] = 0.0
    out.arrays["x_obs"][~np.broadcast_to(valid, out["x_obs"].shape)] = 0.0
    out.arrays["corruption_missingness_add_mask"] = add
    achieved = float((combined & valid).sum() / np.broadcast_to(valid, combined.shape).sum())
    out.metadata["corruption"] = {"type": "missingness", "target_rate": target_rate, "p_add": p_add, "seed": seed, "achieved_rate": achieved}
    return out


def _compact_row(out: SplitData, row: int, keep: np.ndarray, deleted_original: np.ndarray) -> None:
    old_t = int(out["length"][row])
    new_t = int(keep.size)
    three_d = ["x_full", "x_obs", "missing_mask", "latent_u"]
    two_d = ["event_type", "nuisance_cat", "delta", "timestamp", "position", "valid_mask", "signal_mask", "contribution"]

    for name in three_d:
        vals = np.array(out.arrays[name][row, keep, :], copy=True)
        out.arrays[name][row, :, :] = False if name == "missing_mask" else 0
        out.arrays[name][row, :new_t, :] = vals
    for name in two_d:
        vals = np.array(out.arrays[name][row, keep], copy=True)
        fill = -1 if name in {"event_type", "nuisance_cat", "position"} else 0
        out.arrays[name][row, :] = fill
        out.arrays[name][row, :new_t] = vals

    out.arrays["valid_mask"][row, :] = False
    out.arrays["valid_mask"][row, :new_t] = True
    out.arrays["position"][row, :] = -1
    out.arrays["position"][row, :new_t] = np.arange(new_t, dtype=np.int16)
    out.arrays["length"][row] = new_t
    out.arrays["deleted_event_mask_original"][row, :old_t] = deleted_original


def apply_event_deletion(clean: SplitData, fraction: float, seed: int) -> SplitData:
    if not (0.0 < fraction < 1.0):
        raise ValueError("fraction must lie in (0,1)")
    out = clean.copy()
    rng = _rng(seed, 3203)
    n, tmax = out["valid_mask"].shape
    out.arrays["deleted_event_mask_original"] = np.zeros((n, tmax), dtype=bool)
    counts = []
    for i in range(n):
        t = int(out["length"][i])
        n_del = min(max(1, int(round(t * fraction))), t - 1)
        idx = np.sort(rng.choice(t, n_del, replace=False))
        deleted = np.zeros(t, dtype=bool)
        deleted[idx] = True
        keep = np.flatnonzero(~deleted)
        _compact_row(out, i, keep, deleted)
        counts.append(n_del)
    out.metadata["corruption"] = {"type": "event_deletion", "fraction": fraction, "seed": seed, "mean_deleted_events": float(np.mean(counts))}
    return out


def apply_numerical_noise(clean: SplitData, q: float, train_scale: np.ndarray, seed: int) -> SplitData:
    if q <= 0:
        raise ValueError("q must be positive")
    train_scale = np.asarray(train_scale, dtype=np.float64)
    if train_scale.shape != (clean["x_obs"].shape[-1],):
        raise ValueError("train_scale has wrong shape")
    out = clean.copy()
    rng = _rng(seed, 3307)
    noise = rng.normal(0.0, q * train_scale.reshape(1, 1, -1), size=out["x_obs"].shape)
    mask = np.broadcast_to(out["valid_mask"][..., None], out["x_obs"].shape) & (~out["missing_mask"])
    out.arrays["x_obs"][mask] += noise[mask]
    out.arrays["corruption_numeric_noise"] = np.where(mask, noise, 0.0)
    out.metadata["corruption"] = {"type": "numerical_noise", "q": q, "seed": seed, "train_scale": train_scale.tolist()}
    return out


def apply_temporal_jitter(clean: SplitData, q: float, seed: int) -> SplitData:
    if q <= 0:
        raise ValueError("q must be positive")
    out = clean.copy()
    rng = _rng(seed, 3407)
    eps_all = np.zeros_like(out["delta"])
    new_delta = np.zeros_like(out["delta"])
    new_timestamp = np.zeros_like(out["timestamp"])
    for i, t_ in enumerate(out["length"]):
        t = int(t_)
        eps = rng.normal(0.0, q, size=t)
        d = out["delta"][i, :t] * np.exp(eps)
        eps_all[i, :t] = eps
        new_delta[i, :t] = d
        new_timestamp[i, :t] = np.cumsum(d)
    out.arrays["delta"] = new_delta
    out.arrays["timestamp"] = new_timestamp
    out.arrays["corruption_temporal_log_jitter"] = eps_all
    out.metadata["corruption"] = {"type": "temporal_jitter", "q": q, "seed": seed}
    return out


def _move_order(t: int, probe_idx: int, target_idx: int) -> np.ndarray:
    order = list(range(t))
    item = order.pop(probe_idx)
    order.insert(target_idx, item)
    return np.asarray(order, dtype=np.int64)


def _permute_record(record: dict[str, Any], order: np.ndarray) -> dict[str, Any]:
    moved = dict(record)
    for name in ("x_full", "x_obs", "missing_mask", "latent_u", "event_type", "nuisance_cat", "delta", "signal_mask", "contribution"):
        moved[name] = np.asarray(record[name])[order].copy()
    moved["timestamp"] = np.cumsum(moved["delta"])
    moved["position"] = np.arange(int(record["length"]), dtype=np.int16)
    return moved


def generate_position_sweep(cfg: IPATSyntheticConfig, n_triplets: int, seed: int | None = None) -> SplitData:
    if n_triplets <= 0:
        raise ValueError("n_triplets must be positive")
    seed = cfg.master_seed if seed is None else int(seed)
    feature_rng = _rng(seed, 4101)
    label_rng = _rng(seed, 4201)
    arrays = _allocate(n_triplets * 3, cfg)
    arrays["variant"] = np.empty(n_triplets * 3, dtype="S8")
    arrays["triplet_id"] = np.empty(n_triplets * 3, dtype=np.int32)
    arrays["probe_target_position"] = np.empty(n_triplets * 3, dtype=np.int16)

    for k in range(n_triplets):
        rec = _build_single_sequence(cfg, feature_rng, cfg.position_sweep_temporal_weight)
        probe = int(np.flatnonzero(rec["signal_mask"])[0])
        t = int(rec["length"])
        targets = {
            "early": int(round(0.10 * (t - 1))),
            "middle": int(round(0.50 * (t - 1))),
            "late": int(round(0.90 * (t - 1))),
        }
        y = np.int8(label_rng.random() < rec["p_true"])
        for j, (variant, target) in enumerate(targets.items()):
            row = 3 * k + j
            moved = _permute_record(rec, _move_order(t, probe, target))
            _write(arrays, row, moved, f"{cfg.dataset_id}:position:{k:06d}:{variant}")
            arrays["y"][row] = y
            arrays["variant"][row] = variant.encode("utf-8")
            arrays["triplet_id"][row] = k
            arrays["probe_target_position"][row] = target

    metadata = {
        "dataset_id": cfg.dataset_id,
        "profile": "position_sweep",
        "seed": seed,
        "config": replace(cfg, temporal_signal_weight=cfg.position_sweep_temporal_weight).to_dict(),
        "variants": ["early", "middle", "late"],
    }
    return SplitData(arrays, metadata)


def save_hdf5(data: SplitData, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["metadata_json"] = json.dumps(data.metadata, sort_keys=True)
        for name, arr in data.arrays.items():
            h5.create_dataset(name, data=arr, compression="gzip", shuffle=True)
    return path


def load_hdf5(path: str | Path) -> SplitData:
    with h5py.File(path, "r") as h5:
        metadata = json.loads(h5.attrs["metadata_json"])
        arrays = {name: h5[name][()] for name in h5.keys()}
    return SplitData(arrays, metadata)


def summarize(data: SplitData) -> dict[str, Any]:
    valid3 = np.broadcast_to(data["valid_mask"][..., None], data["missing_mask"].shape)
    realized_missing = float((data["missing_mask"] & valid3).sum() / valid3.sum())
    return {
        "n_sequences": int(data["length"].size),
        "length_min": int(data["length"].min()),
        "length_max": int(data["length"].max()),
        "length_mean": float(data["length"].mean()),
        "class_prevalence": float(data["y"].mean()),
        "mean_p_true": float(data["p_true"].mean()),
        "signal_events_per_sequence": np.unique(data["signal_mask"].sum(axis=1)).tolist(),
        "realized_missing_rate": realized_missing,
    }
