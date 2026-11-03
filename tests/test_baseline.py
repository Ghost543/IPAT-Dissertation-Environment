from __future__ import annotations

import numpy as np
import torch

from ipat.model.baseline import (
    BaselineConfig,
    FixedIdentityNumericalEncoder,
    SemiOrthogonalProjector,
    StandardSelfAttention,
    StandardTransformerBaseline,
)
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


def cfg(dropout: float = 0.0) -> BaselineConfig:
    return BaselineConfig(dropout=dropout)


def batch(b: int = 3, t: int = 8) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    valid = torch.zeros(b, t, dtype=torch.bool)
    lengths = [t, t - 2, max(1, t - 4)]
    for i, length in enumerate(lengths):
        valid[i, :length] = True
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


def test_derived_dimensions_follow_math03_rule():
    c = cfg()
    assert c.d_num == 18
    assert c.d_cat == 8
    assert c.d_time == 7
    assert c.d_mask == 6
    assert c.d_in == 39
    assert c.d_model == 40
    assert c.d_model >= c.d_in
    assert c.d_model % c.n_heads == 0


def test_fixed_numeric_encoder_keeps_explicit_identity_and_neutral_missing():
    c = cfg()
    enc = FixedIdentityNumericalEncoder(c, stats())
    x = torch.tensor([[[2.0, -1.0, 0.5, 3.0, 1.0, 0.0]]])
    missing = torch.tensor([[[False, True, False, False, False, False]]])
    valid = torch.tensor([[True]])
    out, x_std = enc(x, missing, valid)
    assert x_std[0, 0, 0].item() == 2.0
    assert x_std[0, 0, 1].item() == 0.0
    blocks = out.view(1, 1, 6, 3)
    assert blocks[0, 0, 0, 0].item() == 2.0
    assert torch.all(blocks[0, 0, 1] == 0)


def test_semiorthogonal_projector_is_exact_up_to_float_tolerance():
    c = cfg()
    proj = SemiOrthogonalProjector(c.d_in, c.d_model)
    w = proj.orthogonal_weight()
    gram = w.T @ w
    assert torch.allclose(gram, torch.eye(c.d_in), atol=1e-6, rtol=1e-6)

    u1 = torch.randn(5, c.d_in)
    u2 = torch.randn(5, c.d_in)
    z1 = torch.nn.functional.linear(u1, w)
    z2 = torch.nn.functional.linear(u2, w)
    assert torch.allclose(
        torch.linalg.vector_norm(z1 - z2, dim=-1),
        torch.linalg.vector_norm(u1 - u2, dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_standard_attention_normalises_valid_keys_and_zeros_padding():
    c = cfg()
    attn = StandardSelfAttention(c.d_model, c.n_heads, dropout=0.0)
    x = torch.randn(2, 6, c.d_model)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    out, weights = attn(x, valid, return_attention=True)
    assert weights is not None
    for bi in range(2):
        for qi in range(6):
            if valid[bi, qi]:
                sums = weights[bi, :, qi, :].sum(dim=-1)
                assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
            else:
                assert torch.all(weights[bi, :, qi, :] == 0)
        assert torch.all(weights[bi, :, :, ~valid[bi]] == 0)
        assert torch.all(out[bi, ~valid[bi], :] == 0)


def test_model_forward_shapes_padding_and_finite_values():
    c = cfg()
    model = StandardTransformerBaseline(c, stats()).eval()
    b = batch()
    out = model(b, return_diagnostics=True)
    assert out["logits"].shape == (3,)
    assert out["prob"].shape == (3,)
    assert out["u"].shape[-1] == c.d_in
    assert out["z_pre_norm"].shape[-1] == c.d_model
    assert len(out["attention"]) == c.n_layers
    assert torch.isfinite(out["logits"]).all()
    pad = ~b["valid_mask"]
    assert torch.all(out["u"][pad] == 0)
    assert torch.all(out["z_pre_norm"][pad] == 0)
    assert torch.all(out["hidden"][pad] == 0)


def test_model_backward_has_finite_nonzero_gradients():
    c = cfg()
    model = StandardTransformerBaseline(c, stats())
    b = batch()
    logits = model(b)["logits"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, b["y"].float()
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g).item() > 0 for g in grads)


def test_eval_is_deterministic():
    c = cfg(dropout=0.1)
    model = StandardTransformerBaseline(c, stats()).eval()
    b = batch()
    a = model(b)["logits"]
    d = model(b)["logits"]
    assert torch.equal(a, d)


def test_no_retention_or_gamma_parameters_in_imp01():
    model = StandardTransformerBaseline(cfg(), stats())
    names = [name.lower() for name, _ in model.named_parameters()]
    assert not any("retention" in name for name in names)
    assert not any("gamma" in name for name in names)
