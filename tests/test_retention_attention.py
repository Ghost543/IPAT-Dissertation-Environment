from __future__ import annotations

import torch
import torch.nn.functional as F

from ipat.model.baseline import BaselineConfig, StandardSelfAttention, StandardTransformerBaseline
from ipat.model.retention import (
    RetentionAwareSelfAttention,
    RetentionAwareTransformer,
    RetentionConfig,
    RetentionScorer,
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


def base_cfg(dropout: float = 0.0) -> BaselineConfig:
    return BaselineConfig(dropout=dropout)


def ret_cfg(gamma: float = 1.0, dropout: float = 0.0) -> RetentionConfig:
    return RetentionConfig(gamma=gamma, dropout=dropout)


def batch(b: int = 3, t: int = 8) -> dict[str, torch.Tensor]:
    torch.manual_seed(77)
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


def test_retention_config_rejects_negative_gamma():
    try:
        ret_cfg(gamma=-0.1).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("negative gamma should be rejected")


def test_retention_scorer_zero_init_is_centered_zero_and_stable():
    scorer = RetentionScorer(d_model=5)
    z = torch.tensor(
        [[[1000.0] * 5, [-1000.0] * 5, [1.0] * 5, [0.0] * 5]],
        dtype=torch.float32,
    )
    valid = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    out = scorer(z, valid)
    assert torch.all(out["r"][valid] == 0.5)
    assert torch.all(out["centered_log_r"][valid] == 0.0)
    assert torch.all(out["centered_log_r"][~valid] == 0.0)
    assert torch.isfinite(out["log_r"]).all()
    assert torch.isfinite(out["contrast"]).all()


def test_centered_log_retention_sums_to_zero_over_valid_events():
    scorer = RetentionScorer(d_model=4)
    with torch.no_grad():
        scorer.linear.weight.copy_(torch.tensor([[0.4, -0.2, 0.1, 0.3]]))
        scorer.linear.bias.fill_(0.2)
    z = torch.randn(2, 6, 4)
    valid = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool
    )
    out = scorer(z, valid)
    sums = (out["centered_log_r"] * valid).sum(dim=1)
    assert torch.allclose(sums, torch.zeros_like(sums), atol=1e-6, rtol=0)
    assert torch.all(out["r"][~valid] == 0)


def test_gamma_zero_attention_matches_standard_exactly():
    torch.manual_seed(9)
    standard = StandardSelfAttention(40, 4, dropout=0.0)
    torch.manual_seed(99)
    retention = RetentionAwareSelfAttention(40, 4, dropout=0.0, gamma=0.0)
    retention.qkv.load_state_dict(standard.qkv.state_dict())
    retention.out_proj.load_state_dict(standard.out_proj.state_dict())

    x = torch.randn(2, 7, 40)
    valid = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool
    )
    centered = torch.randn(2, 7) * valid
    out_a, w_a = standard(x, valid, return_attention=True)
    out_b, w_b = retention(x, valid, centered, return_attention=True)
    assert torch.equal(out_a, out_b)
    assert torch.equal(w_a, w_b)


def test_gamma_zero_qkv_and_output_projection_gradients_match():
    torch.manual_seed(10)
    standard = StandardSelfAttention(40, 4, dropout=0.0)
    retention = RetentionAwareSelfAttention(40, 4, dropout=0.0, gamma=0.0)
    retention.qkv.load_state_dict(standard.qkv.state_dict())
    retention.out_proj.load_state_dict(standard.out_proj.state_dict())

    x_a = torch.randn(2, 5, 40, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)
    valid = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    centered = torch.randn(2, 5, requires_grad=True)

    standard(x_a, valid)[0].square().sum().backward()
    retention(x_b, valid, centered)[0].square().sum().backward()

    assert torch.equal(standard.qkv.weight.grad, retention.qkv.weight.grad)
    assert torch.equal(standard.qkv.bias.grad, retention.qkv.bias.grad)
    assert torch.equal(standard.out_proj.weight.grad, retention.out_proj.weight.grad)
    assert torch.equal(standard.out_proj.bias.grad, retention.out_proj.bias.grad)
    assert torch.equal(x_a.grad, x_b.grad)
    assert centered.grad is not None
    assert torch.count_nonzero(centered.grad).item() == 0


def test_equal_retention_is_standard_attention_for_nonzero_gamma():
    standard = StandardSelfAttention(40, 4, dropout=0.0)
    retention = RetentionAwareSelfAttention(40, 4, dropout=0.0, gamma=3.0)
    retention.qkv.load_state_dict(standard.qkv.state_dict())
    retention.out_proj.load_state_dict(standard.out_proj.state_dict())
    x = torch.randn(2, 6, 40)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    centered = torch.zeros(2, 6)
    a, wa = standard(x, valid, return_attention=True)
    b, wb = retention(x, valid, centered, return_attention=True)
    assert torch.equal(a, b)
    assert torch.equal(wa, wb)


def test_padded_keys_queries_and_retention_diagnostics_are_zeroed():
    attn = RetentionAwareSelfAttention(40, 4, dropout=0.0, gamma=1.0)
    x = torch.randn(2, 6, 40)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    centered = torch.randn(2, 6) * valid
    out, weights = attn(x, valid, centered, return_attention=True)
    assert weights is not None
    for bi in range(2):
        assert torch.all(weights[bi, :, :, ~valid[bi]] == 0)
        assert torch.all(weights[bi, :, ~valid[bi], :] == 0)
        assert torch.all(out[bi, ~valid[bi], :] == 0)


def test_gamma_derivative_matches_math05_formula():
    torch.manual_seed(14)
    attn = RetentionAwareSelfAttention(8, 2, dropout=0.0, gamma=1.0)
    x = torch.randn(1, 4, 8)
    valid = torch.ones(1, 4, dtype=torch.bool)
    lam = torch.tensor([[0.4, -0.3, 0.2, -0.3]], dtype=torch.float32)
    gamma = torch.tensor(0.7, requires_grad=True)
    _, weights = attn(
        x, valid, lam, return_attention=True, gamma_override=gamma
    )
    p = weights[0, 0, 0]
    s = 1
    grad = torch.autograd.grad(p[s], gamma, retain_graph=True)[0]
    expected = p[s] * (lam[0, s] - torch.sum(p * lam[0]))
    assert torch.allclose(grad, expected, atol=1e-6, rtol=1e-5)


def test_eta_derivative_matches_math05_formula():
    torch.manual_seed(15)
    attn = RetentionAwareSelfAttention(8, 2, dropout=0.0, gamma=1.3)
    x = torch.randn(1, 4, 8)
    valid = torch.ones(1, 4, dtype=torch.bool)
    eta = torch.tensor([[0.5, -0.2, 0.9, -1.0]], requires_grad=True)
    log_r = F.logsigmoid(eta)
    lam = log_r - log_r.mean(dim=1, keepdim=True)
    _, weights = attn(x, valid, lam, return_attention=True)
    p = weights[0, 0, 0]
    s, u = 2, 1
    grad_eta = torch.autograd.grad(p[s], eta, retain_graph=True)[0][0, u]
    r_u = torch.sigmoid(eta[0, u])
    expected = 1.3 * (1.0 - r_u) * p[s] * (-p[u])
    assert torch.allclose(grad_eta, expected, atol=1e-6, rtol=1e-5)


def test_shared_initialisation_matches_imp01_with_same_seed():
    seed = 4242
    torch.manual_seed(seed)
    baseline = StandardTransformerBaseline(base_cfg(), stats())
    torch.manual_seed(seed)
    retention = RetentionAwareTransformer(ret_cfg(gamma=0.0), stats())

    base_params = dict(baseline.named_parameters())
    ret_params = dict(retention.named_parameters())
    for name, p in base_params.items():
        assert name in ret_params, name
        assert torch.equal(p, ret_params[name]), name


def test_full_model_gamma_zero_forward_and_shared_gradients_match():
    seed = 5050
    torch.manual_seed(seed)
    baseline = StandardTransformerBaseline(base_cfg(), stats())
    torch.manual_seed(seed)
    retention = RetentionAwareTransformer(ret_cfg(gamma=0.0), stats())
    b = batch()

    out_a = baseline(b, return_diagnostics=True)
    out_b = retention(b, return_diagnostics=True)
    assert torch.equal(out_a["logits"], out_b["logits"])
    assert torch.equal(out_a["hidden"], out_b["hidden"])
    for wa, wb in zip(out_a["attention"], out_b["attention"], strict=True):
        assert torch.equal(wa, wb)

    y = b["y"].float()
    F.binary_cross_entropy_with_logits(out_a["logits"], y).backward()
    F.binary_cross_entropy_with_logits(out_b["logits"], y).backward()
    base_params = dict(baseline.named_parameters())
    ret_params = dict(retention.named_parameters())
    for name, p in base_params.items():
        assert p.grad is not None, name
        assert ret_params[name].grad is not None, name
        assert torch.equal(p.grad, ret_params[name].grad), name
    scorer_grads = [p.grad for p in retention.retention_scorer.parameters()]
    assert all(g is not None for g in scorer_grads)
    assert all(torch.count_nonzero(g).item() == 0 for g in scorer_grads)


def test_zero_effect_initialisation_matches_imp01_even_gamma_nonzero():
    seed = 6060
    torch.manual_seed(seed)
    baseline = StandardTransformerBaseline(base_cfg(), stats()).eval()
    torch.manual_seed(seed)
    retention = RetentionAwareTransformer(ret_cfg(gamma=2.0), stats()).eval()
    b = batch()
    a = baseline(b)["logits"]
    out = retention(b, return_diagnostics=True)
    assert torch.equal(a, out["logits"])
    assert torch.all(out["retention"]["centered_log_r"][b["valid_mask"]] == 0)


def test_nonuniform_retention_changes_attention_when_gamma_positive():
    seed = 7070
    torch.manual_seed(seed)
    baseline = StandardTransformerBaseline(base_cfg(), stats()).eval()
    torch.manual_seed(seed)
    retention = RetentionAwareTransformer(ret_cfg(gamma=1.0), stats()).eval()
    with torch.no_grad():
        retention.retention_scorer.linear.weight.fill_(0.05)
        retention.retention_scorer.linear.bias.zero_()
    b = batch()
    out_a = baseline(b, return_diagnostics=True)
    out_b = retention(b, return_diagnostics=True)
    assert torch.any(out_b["retention"]["contrast"] > 0)
    assert not torch.equal(out_a["attention"][0], out_b["attention"][0])
    assert not torch.equal(out_a["logits"], out_b["logits"])


def test_gamma_is_buffer_not_learnable_parameter():
    model = RetentionAwareTransformer(ret_cfg(gamma=1.0), stats())
    names = [name for name, _ in model.named_parameters()]
    assert not any(name.endswith("gamma") for name in names)
    buffer_names = [name for name, _ in model.named_buffers()]
    assert any(name.endswith("attention.gamma") for name in buffer_names)
