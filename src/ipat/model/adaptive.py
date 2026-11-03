from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ipat.model.baseline import (
    BaselineConfig,
    CategoricalEncoder,
    PreNormTransformerBlock,
    SemiOrthogonalProjector,
    TemporalEncoder,
)
from ipat.model.retention import (
    RetentionAwarePreNormTransformerBlock,
    RetentionScorer,
)
from ipat.training.preprocessing import BaselinePreprocessingStats


@dataclass(frozen=True)
class AdaptiveRepresentationConfig(BaselineConfig):
    """MATH-02 adaptive numerical representation configuration.

    ``asinh_scales`` contains the positive alpha values for
    ``asinh(alpha * x) / alpha``.  The identity candidate is always present.
    The defaults are engineering starting values, not theoretical constants.
    """

    identity_floor: float = 0.25
    asinh_scales: tuple[float, ...] = (0.25, 1.0, 4.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "asinh_scales", tuple(float(x) for x in self.asinh_scales))

    @property
    def n_candidates(self) -> int:
        return 1 + len(self.asinh_scales)

    def validate(self) -> None:
        super().validate()
        if not (0.0 < self.identity_floor < 1.0):
            raise ValueError("MATH-02 requires identity_floor to lie in (0, 1)")
        scales = torch.tensor(self.asinh_scales, dtype=torch.float64)
        if scales.numel() and (not torch.isfinite(scales).all() or torch.any(scales <= 0.0)):
            raise ValueError("Every scaled-asinh alpha must be finite and positive")
        if len(set(self.asinh_scales)) != len(self.asinh_scales):
            raise ValueError("scaled-asinh alpha values must be unique")


@dataclass(frozen=True)
class FullIPATConfig(AdaptiveRepresentationConfig):
    gamma: float = 1.0

    def validate(self) -> None:
        super().validate()
        if self.gamma < 0.0:
            raise ValueError("MATH-05 requires gamma >= 0")


class AdaptiveNumericalEncoder(nn.Module):
    """Identity-floor mixture of strictly increasing scalar candidates.

    For feature ``j``, the implementation follows frozen MATH-02:

    ``p_j = softmax(a_j)``
    ``pi_j0 = rho + (1-rho) p_j0``
    ``pi_jk = (1-rho) p_jk`` for ``k > 0``
    ``g_j(x) = sum_k pi_jk psi_jk(x)``

    where ``psi_j0(x)=x`` and every other candidate is
    ``asinh(alpha_k*x)/alpha_k``.  Consequently ``dg_j/dx >= pi_j0 >= rho``.
    The emitted block remains ``[g_j || h_j(g_j)]`` with exactly the same width
    and learned expansion parameter names as IMP-01/IMP-02.
    """

    def __init__(
        self,
        cfg: AdaptiveRepresentationConfig,
        stats: BaselinePreprocessingStats,
    ) -> None:
        super().__init__()
        cfg.validate()
        if len(stats.numeric_mean) != cfg.n_num or len(stats.numeric_scale) != cfg.n_num:
            raise ValueError("Numerical preprocessing statistics do not match n_num")
        self.cfg = cfg
        self.register_buffer("mean", torch.tensor(stats.numeric_mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(stats.numeric_scale, dtype=torch.float32))

        # Keep these names, shapes and creation order matched to IMP-01/IMP-02.
        self.weight = nn.Parameter(torch.empty(cfg.n_num, cfg.numeric_embedding_dim))
        self.bias = nn.Parameter(torch.zeros(cfg.n_num, cfg.numeric_embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Zero logits are deterministic and consume no RNG, preserving all later shared
        # parameter initialisation under a matched model seed.
        self.mixture_logits = nn.Parameter(torch.zeros(cfg.n_num, cfg.n_candidates))
        self.register_buffer(
            "asinh_scales",
            torch.tensor(cfg.asinh_scales, dtype=torch.float32),
        )

    def mixture_weights(self) -> torch.Tensor:
        p = torch.softmax(self.mixture_logits, dim=-1)
        if self.cfg.n_candidates == 1:
            return torch.ones_like(p)
        weights = (1.0 - self.cfg.identity_floor) * p
        identity = weights[:, :1] + self.cfg.identity_floor
        return torch.cat([identity, weights[:, 1:]], dim=-1)

    def candidate_values(self, x_std: torch.Tensor) -> torch.Tensor:
        identity = x_std.unsqueeze(-1)
        if self.asinh_scales.numel() == 0:
            return identity
        scales = self.asinh_scales.to(dtype=x_std.dtype, device=x_std.device)
        nonlinear = torch.asinh(x_std.unsqueeze(-1) * scales) / scales
        return torch.cat([identity, nonlinear], dim=-1)

    def transform_standardised(
        self,
        x_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = self.candidate_values(x_std)
        weights = self.mixture_weights().to(dtype=x_std.dtype, device=x_std.device)
        g = torch.sum(candidates * weights, dim=-1)
        return g, candidates, weights

    def derivative(self, x_std: torch.Tensor) -> torch.Tensor:
        identity = torch.ones_like(x_std).unsqueeze(-1)
        if self.asinh_scales.numel() == 0:
            candidate_derivatives = identity
        else:
            scales = self.asinh_scales.to(dtype=x_std.dtype, device=x_std.device)
            nonlinear = torch.rsqrt(1.0 + (x_std.unsqueeze(-1) * scales).square())
            candidate_derivatives = torch.cat([identity, nonlinear], dim=-1)
        weights = self.mixture_weights().to(dtype=x_std.dtype, device=x_std.device)
        return torch.sum(candidate_derivatives * weights, dim=-1)

    def encode(
        self,
        x_obs: torch.Tensor,
        missing_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        observed = valid_mask.unsqueeze(-1) & (~missing_mask)
        x_std_raw = (x_obs - self.mean) / self.scale
        x_std = torch.where(observed, x_std_raw, torch.zeros_like(x_std_raw))
        g_raw, candidates_raw, weights = self.transform_standardised(x_std)
        g = torch.where(observed, g_raw, torch.zeros_like(g_raw))
        candidates = torch.where(
            observed.unsqueeze(-1),
            candidates_raw,
            torch.zeros_like(candidates_raw),
        )

        expanded = g.unsqueeze(-1) * self.weight + self.bias
        expanded = torch.where(
            observed.unsqueeze(-1),
            expanded,
            torch.zeros_like(expanded),
        )
        block = torch.cat([g.unsqueeze(-1), expanded], dim=-1)
        derivative = torch.where(
            observed,
            self.derivative(x_std),
            torch.zeros_like(g),
        )
        return {
            "block": block.flatten(start_dim=-2),
            "x_std": x_std,
            "g": g,
            "candidates": candidates,
            "mixture_weights": weights,
            "mixture_derivative": derivative,
        }

    def forward(
        self,
        x_obs: torch.Tensor,
        missing_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x_obs, missing_mask, valid_mask)
        return encoded["block"], encoded["x_std"]


class AdaptiveEventAssembler(nn.Module):
    """MATH-03 event assembler with only the numerical block replaced."""

    def __init__(
        self,
        cfg: AdaptiveRepresentationConfig,
        stats: BaselinePreprocessingStats,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.numeric = AdaptiveNumericalEncoder(cfg, stats)
        self.categorical = CategoricalEncoder(cfg)
        self.temporal = TemporalEncoder(cfg, stats)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = batch["valid_mask"]
        numeric = self.numeric.encode(batch["x_obs"], batch["missing_mask"], valid)
        num = numeric["block"]
        cat = self.categorical(batch["event_type"], batch["nuisance_cat"], valid)
        time = self.temporal(batch["timestamp"], batch["delta"], batch["position"], valid)
        mask = batch["missing_mask"].to(num.dtype) * valid.unsqueeze(-1)
        u = torch.cat([num, cat, time, mask], dim=-1)
        u = u * valid.unsqueeze(-1)
        components = {
            "x_std": numeric["x_std"],
            "g": numeric["g"],
            "num": num,
            "numeric_candidates": numeric["candidates"],
            "mixture_weights": numeric["mixture_weights"],
            "mixture_derivative": numeric["mixture_derivative"],
            "cat": cat,
            "time": time,
            "mask": mask,
        }
        return u, components


class AdaptiveStandardTransformer(nn.Module):
    """IMP-01 attention stack with the IMP-03 representation only."""

    def __init__(
        self,
        cfg: AdaptiveRepresentationConfig,
        stats: BaselinePreprocessingStats,
    ) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.stats = stats
        self.assembler = AdaptiveEventAssembler(cfg, stats)
        self.projector = SemiOrthogonalProjector(cfg.d_in, cfg.d_model)
        self.blocks = nn.ModuleList([PreNormTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.head_norm = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, 1)

    @staticmethod
    def masked_mean(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.unsqueeze(-1).to(x.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, Any]:
        valid = batch["valid_mask"]
        if not torch.all(valid.any(dim=1)):
            raise ValueError("Every sequence must contain at least one valid event")
        u, components = self.assembler(batch)
        z = self.projector(u, valid)
        hidden = z
        attention: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, weights = block(hidden, valid, return_attention=return_diagnostics)
            if return_diagnostics and weights is not None:
                attention.append(weights)
        pooled = self.masked_mean(hidden, valid)
        logits = self.classifier(self.head_norm(pooled)).squeeze(-1)
        out: dict[str, Any] = {"logits": logits, "prob": torch.sigmoid(logits)}
        if return_diagnostics:
            out.update(
                {
                    "u": u,
                    "z_pre_norm": z,
                    "hidden": hidden,
                    "pooled": pooled,
                    "attention": attention,
                    "components": components,
                }
            )
        return out


class FullIPATTransformer(nn.Module):
    """IMP-03 full IPAT: adaptive representation plus frozen IMP-02 attention."""

    def __init__(
        self,
        cfg: FullIPATConfig,
        stats: BaselinePreprocessingStats,
    ) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.stats = stats

        # Construction order remains matched to IMP-02.  The extra mixture logits are
        # zero-initialised and consume no RNG, so later shared parameters remain identical.
        self.assembler = AdaptiveEventAssembler(cfg, stats)
        self.projector = SemiOrthogonalProjector(cfg.d_in, cfg.d_model)
        self.blocks = nn.ModuleList(
            [RetentionAwarePreNormTransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.head_norm = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, 1)
        self.retention_scorer = RetentionScorer(cfg.d_model)

    @staticmethod
    def masked_mean(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        weights = valid_mask.unsqueeze(-1).to(x.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    def set_gamma(self, gamma: float) -> None:
        if gamma < 0.0:
            raise ValueError("MATH-05 requires gamma >= 0")
        for block in self.blocks:
            block.attention.set_gamma(gamma)

    def load_imp02_shared_state(self, imp02_state: dict[str, torch.Tensor]) -> None:
        ours = self.state_dict()
        copied = 0
        for name, value in imp02_state.items():
            if name in ours and ours[name].shape == value.shape:
                ours[name] = value.detach().clone()
                copied += 1
        if copied == 0:
            raise ValueError("No IMP-02-compatible state entries were found")
        self.load_state_dict(ours)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_diagnostics: bool = False,
        gamma_override: torch.Tensor | float | None = None,
    ) -> dict[str, Any]:
        valid = batch["valid_mask"]
        if not torch.all(valid.any(dim=1)):
            raise ValueError("Every sequence must contain at least one valid event")
        u, components = self.assembler(batch)
        z = self.projector(u, valid)
        retention = self.retention_scorer(z, valid)
        centered = retention["centered_log_r"]

        hidden = z
        attention: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, weights = block(
                hidden,
                valid,
                centered,
                return_attention=return_diagnostics,
                gamma_override=gamma_override,
            )
            if return_diagnostics and weights is not None:
                attention.append(weights)
        pooled = self.masked_mean(hidden, valid)
        logits = self.classifier(self.head_norm(pooled)).squeeze(-1)
        out: dict[str, Any] = {"logits": logits, "prob": torch.sigmoid(logits)}
        if return_diagnostics:
            out.update(
                {
                    "u": u,
                    "z_pre_norm": z,
                    "hidden": hidden,
                    "pooled": pooled,
                    "attention": attention,
                    "components": components,
                    "retention": retention,
                }
            )
        return out
