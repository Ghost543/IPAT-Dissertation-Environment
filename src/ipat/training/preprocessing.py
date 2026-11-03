from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BaselinePreprocessingStats:
    numeric_mean: tuple[float, ...]
    numeric_scale: tuple[float, ...]
    timestamp_mean: float
    timestamp_scale: float
    delta_mean: float
    delta_scale: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_scale(values: np.ndarray, floor: float = 1e-8) -> float:
    scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale):
        raise ValueError("Encountered a non-finite standard deviation")
    return max(scale, floor)


def fit_baseline_preprocessing_stats(train: Any) -> BaselinePreprocessingStats:
    """Fit train-only standardisation statistics from a DATA-01 SplitData-like object."""
    x = np.asarray(train["x_obs"], dtype=np.float64)
    missing = np.asarray(train["missing_mask"], dtype=bool)
    valid = np.asarray(train["valid_mask"], dtype=bool)

    means: list[float] = []
    scales: list[float] = []
    for j in range(x.shape[-1]):
        observed = valid & (~missing[..., j])
        vals = x[..., j][observed]
        if vals.size < 2:
            raise ValueError(f"Not enough observed values to fit numerical feature {j}")
        means.append(float(np.mean(vals)))
        scales.append(_safe_scale(vals))

    timestamps = np.asarray(train["timestamp"], dtype=np.float64)[valid]
    deltas = np.asarray(train["delta"], dtype=np.float64)[valid]
    if timestamps.size < 2 or deltas.size < 2:
        raise ValueError("Not enough valid temporal observations")

    return BaselinePreprocessingStats(
        numeric_mean=tuple(means),
        numeric_scale=tuple(scales),
        timestamp_mean=float(np.mean(timestamps)),
        timestamp_scale=_safe_scale(timestamps),
        delta_mean=float(np.mean(deltas)),
        delta_scale=_safe_scale(deltas),
    )
