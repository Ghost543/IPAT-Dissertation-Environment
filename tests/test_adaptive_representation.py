from __future__ import annotations

import pytest
import torch

from ipat.model.adaptive import (
    AdaptiveNumericalEncoder,
    AdaptiveRepresentationConfig,
    AdaptiveStandardTransformer,
    FullIPATConfig,
    FullIPATTransformer,
)
from ipat.model.baseline import (
    BaselineConfig,
    FixedIdentityNumericalEncoder,
    StandardTransformerBaseline,
)
from ipat.model.retention import RetentionAwareTransformer, RetentionConfig
from ipat.training.preprocessing import BaselinePreprocessingStats


def stats() -> BaselinePreprocessingStats:
    return BaselinePreprocessingStats(
        numeric_mean=(0.0,) * 6,
        numeric_scale=(1.0,) * 6,
        timestamp_mean=10.0,
        timestamp_scale=5.0,
        delta_mean=1.0,
        delta_scale=0.5,
    )


def batch(b: int = 3, t: int = 8) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260904)
    valid = torch.zeros(b, t, dtype=torch.bool)
    for i, length in enumerate([t, t - 2, max(1, t - 4)]):
        valid[i, :length] = True
    missing = (torch.rand(b, t, 6) < 0.15) & valid.unsqueeze(-1)
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


@pytest.mark.parametrize("rho", [0.0, 1.0])
def test_config_rejects_invalid_identity_floor(rho: float):
    with pytest.raises(ValueError):
        AdaptiveRepresentationConfig(identity_floor=rho).validate()


@pytest.mark.parametrize("scales", [(0.0,), (-1.0,), (1.0, 1.0)])
def test_config_rejects_invalid_asinh_scales(scales: tuple[float, ...]):
    with pytest.raises(ValueError):
        AdaptiveRepresentationConfig(asinh_scales=scales).validate()


def test_mixture_weights_are_convex_and_respect_identity_floor():
    cfg = AdaptiveRepresentationConfig(identity_floor=0.30)
    enc = AdaptiveNumericalEncoder(cfg, stats())
    with torch.no_grad():
        enc.mixture_logits.normal_()
    weights = enc.mixture_weights()
    weight_sum_tolerance = 8.0 * torch.finfo(weights.dtype).eps
    assert torch.all(weights >= 0.0)
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(cfg.n_num),
        atol=weight_sum_tolerance,
        rtol=0,
    )
    assert torch.all(weights[:, 0] >= cfg.identity_floor)


def test_scaled_asinh_candidates_follow_frozen_formula():
    cfg = AdaptiveRepresentationConfig(asinh_scales=(2.0,))
    enc = AdaptiveNumericalEncoder(cfg, stats())
    x = torch.tensor([[[1.5] * 6]])
    candidates = enc.candidate_values(x)
    assert torch.equal(candidates[..., 0], x)
    assert torch.allclose(candidates[..., 1], torch.asinh(2.0 * x) / 2.0)


def test_mixture_is_strictly_increasing_and_derivative_is_bounded_by_rho():
    cfg = AdaptiveRepresentationConfig(identity_floor=0.20)
    enc = AdaptiveNumericalEncoder(cfg, stats())
    with torch.no_grad():
        enc.mixture_logits.normal_()
    grid = torch.linspace(-20.0, 20.0, 2001).view(1, -1, 1).expand(-1, -1, 6)
    g, _, _ = enc.transform_standardised(grid)
    assert torch.all(torch.diff(g, dim=1) > 0.0)
    assert torch.all(enc.derivative(grid) >= cfg.identity_floor - 1e-7)


def test_missing_and_padding_are_neutral_but_missingness_remains_explicit():
    cfg = AdaptiveRepresentationConfig()
    enc = AdaptiveNumericalEncoder(cfg, stats())
    x = torch.tensor([[[2.0] * 6, [3.0] * 6]])
    missing = torch.zeros_like(x, dtype=torch.bool)
    missing[0, 0, 2] = True
    valid = torch.tensor([[True, False]])
    encoded = enc.encode(x, missing, valid)
    blocks = encoded["block"].view(1, 2, 6, 3)
    assert torch.all(blocks[0, 0, 2] == 0.0)
    assert torch.all(blocks[0, 1] == 0.0)
    assert torch.all(encoded["mixture_derivative"][0, 0, [0, 1, 3, 4, 5]] > 0.0)
    assert encoded["mixture_derivative"][0, 0, 2].item() == 0.0


def test_identity_only_encoder_matches_imp01_exactly():
    base_cfg = BaselineConfig(dropout=0.0)
    adaptive_cfg = AdaptiveRepresentationConfig(dropout=0.0, asinh_scales=())
    torch.manual_seed(111)
    fixed = FixedIdentityNumericalEncoder(base_cfg, stats())
    torch.manual_seed(111)
    adaptive = AdaptiveNumericalEncoder(adaptive_cfg, stats())
    b = batch()
    fixed_out, fixed_std = fixed(b["x_obs"], b["missing_mask"], b["valid_mask"])
    adaptive_out, adaptive_std = adaptive(
        b["x_obs"], b["missing_mask"], b["valid_mask"]
    )
    assert torch.equal(fixed_std, adaptive_std)
    assert torch.equal(fixed_out, adaptive_out)


def test_adaptive_models_keep_imp01_imp02_dimensions_and_output_contract():
    adaptive_cfg = AdaptiveRepresentationConfig(dropout=0.0)
    full_cfg = FullIPATConfig(dropout=0.0, gamma=1.0)
    b = batch()
    adaptive = AdaptiveStandardTransformer(adaptive_cfg, stats()).eval()
    full = FullIPATTransformer(full_cfg, stats()).eval()
    out_a = adaptive(b, return_diagnostics=True)
    out_f = full(b, return_diagnostics=True)
    for out in (out_a, out_f):
        assert out["logits"].shape == (3,)
        assert out["u"].shape[-1] == adaptive_cfg.d_in == 39
        assert out["z_pre_norm"].shape[-1] == adaptive_cfg.d_model == 40
        assert out["components"]["num"].shape[-1] == adaptive_cfg.d_num == 18
        assert torch.isfinite(out["logits"]).all()
    assert "retention" not in out_a
    assert "retention" in out_f


def test_full_identity_gamma_zero_reduces_exactly_to_imp01():
    seed = 222
    base_cfg = BaselineConfig(dropout=0.0)
    full_cfg = FullIPATConfig(dropout=0.0, asinh_scales=(), gamma=0.0)
    torch.manual_seed(seed)
    baseline = StandardTransformerBaseline(base_cfg, stats()).eval()
    torch.manual_seed(seed)
    full = FullIPATTransformer(full_cfg, stats()).eval()
    b = batch()
    out_b = baseline(b, return_diagnostics=True)
    out_f = full(b, return_diagnostics=True)
    assert torch.equal(out_b["logits"], out_f["logits"])
    assert torch.equal(out_b["u"], out_f["u"])
    assert torch.equal(out_b["hidden"], out_f["hidden"])
    for base_attention, full_attention in zip(
        out_b["attention"], out_f["attention"], strict=True
    ):
        assert torch.equal(base_attention, full_attention)


def test_full_ipat_shared_parameters_match_imp02_under_same_seed():
    seed = 333
    torch.manual_seed(seed)
    imp02 = RetentionAwareTransformer(RetentionConfig(dropout=0.0), stats())
    torch.manual_seed(seed)
    full = FullIPATTransformer(FullIPATConfig(dropout=0.0), stats())
    imp02_params = dict(imp02.named_parameters())
    full_params = dict(full.named_parameters())
    for name, value in imp02_params.items():
        assert name in full_params, name
        assert torch.equal(value, full_params[name]), name
    assert set(full_params) - set(imp02_params) == {"assembler.numeric.mixture_logits"}


def test_mixture_logits_receive_finite_nonzero_gradient():
    enc = AdaptiveNumericalEncoder(AdaptiveRepresentationConfig(), stats())
    x = torch.full((2, 3, 6), 2.0)
    g, _, _ = enc.transform_standardised(x)
    g.sum().backward()
    grad = enc.mixture_logits.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() > 0


def test_mixed_representation_preserves_math03_projection_reconstruction():
    cfg = AdaptiveRepresentationConfig(dropout=0.0)
    model = AdaptiveStandardTransformer(cfg, stats()).eval()
    b = batch()
    u, _ = model.assembler(b)
    z = model.projector(u, b["valid_mask"])
    w = model.projector.orthogonal_weight()
    recovered = torch.matmul(z - model.projector.bias, w)
    assert torch.allclose(recovered, u, atol=1e-5, rtol=1e-5)
    assert torch.allclose(w.T @ w, torch.eye(cfg.d_in), atol=1e-6, rtol=1e-6)
