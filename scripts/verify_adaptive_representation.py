from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from ipat.model.adaptive import (
    AdaptiveNumericalEncoder,
    AdaptiveStandardTransformer,
    FullIPATConfig,
    FullIPATTransformer,
)
from ipat.model.baseline import BaselineConfig, StandardTransformerBaseline
from ipat.model.retention import RetentionAwareTransformer, RetentionConfig
from ipat.training.preprocessing import BaselinePreprocessingStats


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stats() -> BaselinePreprocessingStats:
    return BaselinePreprocessingStats(
        numeric_mean=(0.0,) * 6,
        numeric_scale=(1.0,) * 6,
        timestamp_mean=10.0,
        timestamp_scale=5.0,
        delta_mean=1.0,
        delta_scale=0.5,
    )


def batch() -> dict[str, torch.Tensor]:
    seed_everything(20260904)
    b, t = 3, 8
    valid = torch.tensor(
        [[1] * 8, [1] * 6 + [0] * 2, [1] * 4 + [0] * 4], dtype=torch.bool
    )
    missing = (torch.rand(b, t, 6) < 0.1) & valid.unsqueeze(-1)
    x = torch.randn(b, t, 6)
    x[missing] = 0.0
    x[~valid.unsqueeze(-1).expand_as(x)] = 0.0
    event_type = torch.randint(0, 4, (b, t))
    nuisance = torch.randint(0, 8, (b, t))
    event_type[~valid] = -1
    nuisance[~valid] = -1
    delta = torch.rand(b, t) + 0.5
    delta[~valid] = 0.0
    timestamp = torch.cumsum(delta, dim=1)
    timestamp[~valid] = 0.0
    position = torch.arange(t).expand(b, -1).clone()
    position[~valid] = -1
    return {
        "x_obs": x,
        "missing_mask": missing,
        "event_type": event_type,
        "nuisance_cat": nuisance,
        "delta": delta,
        "timestamp": timestamp,
        "position": position,
        "valid_mask": valid,
        "y": torch.tensor([0, 1, 0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen IMP-03 invariants")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = FullIPATConfig(dropout=0.0)
    encoder = AdaptiveNumericalEncoder(cfg, stats())
    with torch.no_grad():
        encoder.mixture_logits.normal_()
    weights = encoder.mixture_weights()
    # The invariant is exact over the reals, but the implementation is evaluated in
    # float32.  Allow a small, dtype-scaled accumulation tolerance rather than a
    # decimal threshold below float32 epsilon at one.
    weight_sum_tolerance = 8.0 * torch.finfo(weights.dtype).eps
    grid = torch.linspace(-20.0, 20.0, 2001).view(1, -1, 1).expand(-1, -1, 6)
    g, _, _ = encoder.transform_standardised(grid)
    min_increment = float(torch.diff(g, dim=1).min().item())
    min_derivative = float(encoder.derivative(grid).min().item())

    adaptive = AdaptiveStandardTransformer(cfg, stats()).eval()
    b = batch()
    u, _ = adaptive.assembler(b)
    z = adaptive.projector(u, b["valid_mask"])
    w = adaptive.projector.orthogonal_weight()
    recovered = torch.matmul(z - adaptive.projector.bias, w)
    max_reconstruction_error = float((recovered - u).abs().max().item())
    max_gram_error = float((w.T @ w - torch.eye(cfg.d_in)).abs().max().item())

    seed = 20260904
    base_cfg = BaselineConfig(dropout=0.0)
    identity_cfg = FullIPATConfig(dropout=0.0, asinh_scales=(), gamma=0.0)
    torch.manual_seed(seed)
    baseline = StandardTransformerBaseline(base_cfg, stats()).eval()
    torch.manual_seed(seed)
    identity_full = FullIPATTransformer(identity_cfg, stats()).eval()
    base_out = baseline(b, return_diagnostics=True)
    identity_out = identity_full(b, return_diagnostics=True)
    identity_logit_diff = float(
        (base_out["logits"] - identity_out["logits"]).abs().max().item()
    )
    identity_attention_diff = max(
        float((a - f).abs().max().item())
        for a, f in zip(base_out["attention"], identity_out["attention"], strict=True)
    )

    torch.manual_seed(seed)
    imp02 = RetentionAwareTransformer(RetentionConfig(dropout=0.0), stats())
    torch.manual_seed(seed)
    full = FullIPATTransformer(FullIPATConfig(dropout=0.0), stats())
    imp02_params = dict(imp02.named_parameters())
    full_params = dict(full.named_parameters())
    shared_state_max_diff = max(
        float((value - full_params[name]).abs().max().item())
        for name, value in imp02_params.items()
    )

    result = {
        "milestone": "IMP-03",
        "status": "PASS",
        "scope": (
            "Development acceptance only; local dissertation-environment "
            "replication required."
        ),
        "identity_floor": cfg.identity_floor,
        "mixture_weight_dtype": str(weights.dtype).removeprefix("torch."),
        "minimum_observed_identity_weight": float(weights[:, 0].min().item()),
        "maximum_weight_sum_error": float((weights.sum(dim=-1) - 1.0).abs().max().item()),
        "weight_sum_absolute_tolerance": weight_sum_tolerance,
        "minimum_grid_increment": min_increment,
        "minimum_grid_derivative": min_derivative,
        "maximum_projection_gram_error": max_gram_error,
        "maximum_projection_reconstruction_error": max_reconstruction_error,
        "identity_gamma_zero_max_logit_difference": identity_logit_diff,
        "identity_gamma_zero_max_attention_difference": identity_attention_diff,
        "maximum_initial_imp02_shared_parameter_difference": shared_state_max_diff,
    }
    assert result["minimum_observed_identity_weight"] >= cfg.identity_floor
    assert result["maximum_weight_sum_error"] <= weight_sum_tolerance
    assert min_increment > 0.0
    assert min_derivative >= cfg.identity_floor - 1e-7
    assert max_gram_error <= 1e-5
    assert max_reconstruction_error <= 1e-4
    assert identity_logit_diff == 0.0
    assert identity_attention_diff == 0.0
    assert shared_state_max_diff == 0.0

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
