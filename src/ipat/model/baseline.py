from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from ipat.training.preprocessing import BaselinePreprocessingStats


@dataclass(frozen=True)
class BaselineConfig:
    n_num: int = 6
    event_type_cardinality: int = 4
    nuisance_cardinality: int = 8
    numeric_embedding_dim: int = 2
    event_type_embedding_dim: int = 4
    nuisance_embedding_dim: int = 4
    n_time_harmonics: int = 4
    n_heads: int = 4
    n_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.10

    @property
    def d_num(self) -> int:
        return self.n_num * (1 + self.numeric_embedding_dim)

    @property
    def d_cat(self) -> int:
        return self.event_type_embedding_dim + self.nuisance_embedding_dim

    @property
    def d_time(self) -> int:
        # explicit timestamp, gap, relative position + periodic basis
        return 3 + self.n_time_harmonics

    @property
    def d_mask(self) -> int:
        return self.n_num

    @property
    def d_in(self) -> int:
        return self.d_num + self.d_cat + self.d_time + self.d_mask

    @property
    def d_model(self) -> int:
        # Frozen MATH-03 structural rule.
        return self.n_heads * math.ceil(self.d_in / self.n_heads)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ff_dim(self) -> int:
        return self.ff_multiplier * self.d_model

    def validate(self) -> None:
        if self.n_heads <= 0 or self.n_layers <= 0:
            raise ValueError("n_heads and n_layers must be positive")
        if self.d_model < self.d_in:
            raise ValueError("MATH-03 requires d_model >= d_in")
        if self.d_model % self.n_heads != 0:
            raise ValueError("n_heads must divide d_model")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must lie in [0, 1)")


class FixedIdentityNumericalEncoder(nn.Module):
    """Matched baseline numerical interface: explicit standardised scalar + learned expansion.

    This is intentionally not the adaptive MATH-02 mixture.  It preserves an explicit identity
    coordinate so IMP-03 can later replace only g_j while keeping the output interface fixed.
    """

    def __init__(self, cfg: BaselineConfig, stats: BaselinePreprocessingStats) -> None:
        super().__init__()
        if len(stats.numeric_mean) != cfg.n_num or len(stats.numeric_scale) != cfg.n_num:
            raise ValueError("Numerical preprocessing statistics do not match n_num")
        self.cfg = cfg
        self.register_buffer("mean", torch.tensor(stats.numeric_mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(stats.numeric_scale, dtype=torch.float32))
        self.weight = nn.Parameter(torch.empty(cfg.n_num, cfg.numeric_embedding_dim))
        self.bias = nn.Parameter(torch.zeros(cfg.n_num, cfg.numeric_embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        x_obs: torch.Tensor,
        missing_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed = valid_mask.unsqueeze(-1) & (~missing_mask)
        x_std = (x_obs - self.mean) / self.scale
        x_std = torch.where(observed, x_std, torch.zeros_like(x_std))

        expanded = x_std.unsqueeze(-1) * self.weight + self.bias
        expanded = torch.where(
            observed.unsqueeze(-1), expanded, torch.zeros_like(expanded)
        )
        block = torch.cat([x_std.unsqueeze(-1), expanded], dim=-1)
        return block.flatten(start_dim=-2), x_std


class CategoricalEncoder(nn.Module):
    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.event_padding_id = cfg.event_type_cardinality
        self.nuisance_padding_id = cfg.nuisance_cardinality
        self.event_embedding = nn.Embedding(
            cfg.event_type_cardinality + 1,
            cfg.event_type_embedding_dim,
            padding_idx=self.event_padding_id,
        )
        self.nuisance_embedding = nn.Embedding(
            cfg.nuisance_cardinality + 1,
            cfg.nuisance_embedding_dim,
            padding_idx=self.nuisance_padding_id,
        )

    def forward(
        self,
        event_type: torch.Tensor,
        nuisance_cat: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        event_ids = torch.where(
            valid_mask, event_type, torch.full_like(event_type, self.event_padding_id)
        )
        nuisance_ids = torch.where(
            valid_mask,
            nuisance_cat,
            torch.full_like(nuisance_cat, self.nuisance_padding_id),
        )
        return torch.cat(
            [self.event_embedding(event_ids), self.nuisance_embedding(nuisance_ids)],
            dim=-1,
        )


class TemporalEncoder(nn.Module):
    def __init__(self, cfg: BaselineConfig, stats: BaselinePreprocessingStats) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("timestamp_mean", torch.tensor(stats.timestamp_mean, dtype=torch.float32))
        self.register_buffer("timestamp_scale", torch.tensor(stats.timestamp_scale, dtype=torch.float32))
        self.register_buffer("delta_mean", torch.tensor(stats.delta_mean, dtype=torch.float32))
        self.register_buffer("delta_scale", torch.tensor(stats.delta_scale, dtype=torch.float32))
        frequencies = 2.0 ** torch.arange(cfg.n_time_harmonics, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)

    def forward(
        self,
        timestamp: torch.Tensor,
        delta: torch.Tensor,
        position: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        tau = (timestamp - self.timestamp_mean) / self.timestamp_scale
        gap = (delta - self.delta_mean) / self.delta_scale
        lengths = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        denom = (lengths - 1).clamp_min(1).to(position.dtype)
        relpos = position.clamp_min(0).to(torch.float32) / denom.to(torch.float32)
        periodic = torch.sin(tau.unsqueeze(-1) * self.frequencies)
        out = torch.cat(
            [tau.unsqueeze(-1), gap.unsqueeze(-1), relpos.unsqueeze(-1), periodic],
            dim=-1,
        )
        return out * valid_mask.unsqueeze(-1)


class EventAssembler(nn.Module):
    def __init__(self, cfg: BaselineConfig, stats: BaselinePreprocessingStats) -> None:
        super().__init__()
        self.cfg = cfg
        self.numeric = FixedIdentityNumericalEncoder(cfg, stats)
        self.categorical = CategoricalEncoder(cfg)
        self.temporal = TemporalEncoder(cfg, stats)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = batch["valid_mask"]
        num, x_std = self.numeric(batch["x_obs"], batch["missing_mask"], valid)
        cat = self.categorical(batch["event_type"], batch["nuisance_cat"], valid)
        time = self.temporal(batch["timestamp"], batch["delta"], batch["position"], valid)
        mask = batch["missing_mask"].to(num.dtype) * valid.unsqueeze(-1)
        u = torch.cat([num, cat, time, mask], dim=-1)
        u = u * valid.unsqueeze(-1)
        return u, {"x_std": x_std, "num": num, "cat": cat, "time": time, "mask": mask}


class SemiOrthogonalProjector(nn.Module):
    """Trainable W with exact W^T W = I through reduced QR on every forward pass."""

    def __init__(self, d_in: int, d_model: int) -> None:
        super().__init__()
        if d_model < d_in:
            raise ValueError("Semi-orthogonal projection requires d_model >= d_in")
        raw = torch.zeros(d_model, d_in)
        raw[:d_in, :] = torch.eye(d_in)
        self.raw = nn.Parameter(raw)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.d_in = d_in
        self.d_model = d_model

    def orthogonal_weight(self) -> torch.Tensor:
        q, _ = torch.linalg.qr(self.raw, mode="reduced")
        return q

    def forward(self, u: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        w = self.orthogonal_weight()
        z = F.linear(u, w, self.bias)
        return z * valid_mask.unsqueeze(-1)


class StandardSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention.

    IMP-02 will preserve this interface and add only the retention logit term.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("n_heads must divide d_model")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        key_valid = valid_mask[:, None, None, :]
        scores = scores.masked_fill(~key_valid, -torch.inf)
        weights = torch.softmax(scores, dim=-1)

        # Padded queries are not semantic events. Zero them for diagnostics/output.
        query_valid = valid_mask[:, None, :, None]
        weights_for_output = self.attn_dropout(weights) * query_valid
        context = torch.matmul(weights_for_output, v)
        context = context.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.out_proj(context) * valid_mask.unsqueeze(-1)

        if return_attention:
            diagnostic = weights * query_valid
            return out, diagnostic
        return out, None


class PreNormTransformerBlock(nn.Module):
    def __init__(self, cfg: BaselineConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attention = StandardSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
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
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, weights = self.attention(
            self.norm1(x), valid_mask, return_attention=return_attention
        )
        x = (x + self.drop1(attn_out)) * valid_mask.unsqueeze(-1)
        x = (x + self.drop2(self.ff(self.norm2(x)))) * valid_mask.unsqueeze(-1)
        return x, weights


class StandardTransformerBaseline(nn.Module):
    def __init__(
        self,
        cfg: BaselineConfig,
        stats: BaselinePreprocessingStats,
    ) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.stats = stats
        self.assembler = EventAssembler(cfg, stats)
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
                }
            )
        return out
