from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch
from torch import nn
import torch.nn.functional as F

from ipat.model.baseline import (
    BaselineConfig,
    EventAssembler,
    SemiOrthogonalProjector,
)
from ipat.training.preprocessing import BaselinePreprocessingStats


@dataclass(frozen=True)
class RetentionConfig(BaselineConfig):
    gamma: float = 1.0

    def validate(self) -> None:
        super().validate()
        if self.gamma < 0.0:
            raise ValueError("MATH-05 requires gamma >= 0")


class RetentionScorer(nn.Module):
    """Event-level retention propensity from pre-LayerNorm projected events.

    The score is intentionally shared across heads and queries.  The bounded value r is a
    diagnostic propensity, not a calibrated probability, feature importance or explanation.
    The operational attention quantity is centred log-retention.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        # MATH-04 zero-effect initialisation.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        z_pre_norm: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eta = self.linear(z_pre_norm).squeeze(-1)
        log_r_raw = F.logsigmoid(eta)
        r_raw = torch.sigmoid(eta)

        valid_f = valid_mask.to(z_pre_norm.dtype)
        counts = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_log = (log_r_raw * valid_f).sum(dim=1, keepdim=True) / counts
        centered = (log_r_raw - mean_log) * valid_f

        # Padded locations are not events and are zeroed for diagnostics.
        eta_diag = eta * valid_f
        r_diag = r_raw * valid_f
        log_r_diag = log_r_raw * valid_f

        neg_inf = torch.full_like(centered, -torch.inf)
        pos_inf = torch.full_like(centered, torch.inf)
        max_valid = torch.where(valid_mask, centered, neg_inf).max(dim=1).values
        min_valid = torch.where(valid_mask, centered, pos_inf).min(dim=1).values
        contrast = max_valid - min_valid

        return {
            "eta": eta_diag,
            "r": r_diag,
            "log_r": log_r_diag,
            "centered_log_r": centered,
            "contrast": contrast,
        }


class RetentionAwareSelfAttention(nn.Module):
    """Scaled dot-product attention with additive centred log-retention bias.

    L_{bhis} = S_{bhis} + gamma * centered_log_r_{bs} + mask_{bs}

    The class deliberately mirrors IMP-01 StandardSelfAttention parameter names and module
    construction order so shared parameters can be matched exactly.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        gamma: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("n_heads must divide d_model")
        if gamma < 0.0:
            raise ValueError("MATH-05 requires gamma >= 0")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer("gamma", torch.tensor(float(gamma), dtype=torch.float32))

    def set_gamma(self, gamma: float) -> None:
        if gamma < 0.0:
            raise ValueError("MATH-05 requires gamma >= 0")
        self.gamma.fill_(float(gamma))

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        centered_log_retention: torch.Tensor,
        *,
        return_attention: bool = False,
        gamma_override: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t, _ = x.shape
        if centered_log_retention.shape != (b, t):
            raise ValueError("centered_log_retention must have shape [batch, time]")

        if gamma_override is None:
            gamma = self.gamma.to(dtype=x.dtype, device=x.device)
        elif isinstance(gamma_override, torch.Tensor):
            if gamma_override.numel() != 1:
                raise ValueError("gamma_override must be scalar")
            if float(gamma_override.detach().cpu()) < 0.0:
                raise ValueError("MATH-05 requires gamma >= 0")
            gamma = gamma_override.to(dtype=x.dtype, device=x.device)
        else:
            if gamma_override < 0.0:
                raise ValueError("MATH-05 requires gamma >= 0")
            gamma = torch.tensor(float(gamma_override), dtype=x.dtype, device=x.device)

        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        content_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        retention_bias = gamma * centered_log_retention[:, None, None, :]
        scores = content_scores + retention_bias
        key_valid = valid_mask[:, None, None, :]
        scores = scores.masked_fill(~key_valid, -torch.inf)
        weights = torch.softmax(scores, dim=-1)

        query_valid = valid_mask[:, None, :, None]
        weights_for_output = self.attn_dropout(weights) * query_valid
        context = torch.matmul(weights_for_output, v)
        context = context.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.out_proj(context) * valid_mask.unsqueeze(-1)

        if return_attention:
            return out, weights * query_valid
        return out, None


class RetentionAwarePreNormTransformerBlock(nn.Module):
    def __init__(self, cfg: RetentionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attention = RetentionAwareSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.dropout, cfg.gamma
        )
        self.drop1 = nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_dim, cfg.d_model),
        )
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        centered_log_retention: torch.Tensor,
        *,
        return_attention: bool = False,
        gamma_override: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, weights = self.attention(
            self.norm1(x),
            valid_mask,
            centered_log_retention,
            return_attention=return_attention,
            gamma_override=gamma_override,
        )
        x = (x + self.drop1(attn_out)) * valid_mask.unsqueeze(-1)
        x = (x + self.drop2(self.ff(self.norm2(x)))) * valid_mask.unsqueeze(-1)
        return x, weights


class RetentionAwareTransformer(nn.Module):
    """IMP-02 matched Transformer: IMP-01 plus retention-aware attention logits only."""

    def __init__(
        self,
        cfg: RetentionConfig,
        stats: BaselinePreprocessingStats,
    ) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.stats = stats

        # Shared modules are created in the same order as IMP-01, so resetting the same seed
        # yields identical shared initial parameters.
        self.assembler = EventAssembler(cfg, stats)
        self.projector = SemiOrthogonalProjector(cfg.d_in, cfg.d_model)
        self.blocks = nn.ModuleList(
            [RetentionAwarePreNormTransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.head_norm = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, 1)

        # Created last so it cannot perturb shared-parameter RNG initialisation.
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

    def load_imp01_shared_state(self, imp01_state: dict[str, torch.Tensor]) -> None:
        """Load only parameters/buffers shared with IMP-01, retaining IMP-02-only state."""
        ours = self.state_dict()
        copied = 0
        for name, value in imp01_state.items():
            if name in ours and ours[name].shape == value.shape:
                ours[name] = value.detach().clone()
                copied += 1
        if copied == 0:
            raise ValueError("No IMP-01-compatible state entries were found")
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
        out: dict[str, Any] = {
            "logits": logits,
            "prob": torch.sigmoid(logits),
        }
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
