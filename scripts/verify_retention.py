from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from ipat.data.synthetic import load_hdf5
from ipat.data.torch_dataset import SyntheticTorchDataset
from ipat.model.baseline import BaselineConfig, StandardTransformerBaseline
from ipat.model.retention import RetentionAwareTransformer, RetentionConfig
from ipat.training.preprocessing import fit_baseline_preprocessing_stats


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="IMP-02 gamma=0 parity acceptance check")
    parser.add_argument("--train", default="data/processed/ipat_syn_v0.1/train.h5")
    parser.add_argument("--config", default="configs/model/retention_v0.1.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = RetentionConfig(**payload["model"])
    seed = int(payload["acceptance"]["seed"])
    split = load_hdf5(args.train)
    stats = fit_baseline_preprocessing_stats(split)
    dataset = SyntheticTorchDataset(split, indices=list(range(min(8, len(split["y"])))))
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    device = torch.device(args.device)
    batch = {k: v.to(device) for k, v in batch.items()}

    base_cfg = BaselineConfig(
        n_num=cfg.n_num,
        event_type_cardinality=cfg.event_type_cardinality,
        nuisance_cardinality=cfg.nuisance_cardinality,
        numeric_embedding_dim=cfg.numeric_embedding_dim,
        event_type_embedding_dim=cfg.event_type_embedding_dim,
        nuisance_embedding_dim=cfg.nuisance_embedding_dim,
        n_time_harmonics=cfg.n_time_harmonics,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_multiplier=cfg.ff_multiplier,
        dropout=0.0,
    )
    ret_cfg = replace(cfg, dropout=0.0, gamma=0.0)

    seed_everything(seed)
    baseline = StandardTransformerBaseline(base_cfg, stats).to(device)
    seed_everything(seed)
    retention = RetentionAwareTransformer(ret_cfg, stats).to(device)
    baseline.eval(); retention.eval()

    out_a = baseline(batch, return_diagnostics=True)
    out_b = retention(batch, return_diagnostics=True)
    logit_diff = float((out_a["logits"] - out_b["logits"]).abs().max().item())
    hidden_diff = float((out_a["hidden"] - out_b["hidden"]).abs().max().item())
    attn_diff = max(
        float((a - b).abs().max().item())
        for a, b in zip(out_a["attention"], out_b["attention"], strict=True)
    )
    if logit_diff != 0.0 or hidden_diff != 0.0 or attn_diff != 0.0:
        raise SystemExit(
            f"FAIL forward parity: logits={logit_diff}, hidden={hidden_diff}, attention={attn_diff}"
        )

    y = batch["y"].float()
    torch.nn.functional.binary_cross_entropy_with_logits(out_a["logits"], y).backward()
    torch.nn.functional.binary_cross_entropy_with_logits(out_b["logits"], y).backward()
    ret_params = dict(retention.named_parameters())
    max_grad_diff = 0.0
    for name, p in baseline.named_parameters():
        q = ret_params[name]
        if p.grad is None or q.grad is None:
            raise SystemExit(f"FAIL missing shared gradient: {name}")
        max_grad_diff = max(max_grad_diff, float((p.grad - q.grad).abs().max().item()))
    if max_grad_diff != 0.0:
        raise SystemExit(f"FAIL shared-gradient parity: max_abs_diff={max_grad_diff}")

    scorer_nonzero = sum(
        int(torch.count_nonzero(p.grad).item())
        for p in retention.retention_scorer.parameters()
        if p.grad is not None
    )
    if scorer_nonzero != 0:
        raise SystemExit("FAIL retention scorer received non-zero gradient at gamma=0")

    print("PASS gamma=0 forward parity: exact")
    print("PASS gamma=0 shared-gradient parity: exact")
    print("PASS gamma=0 retention-scorer gradient: zero")
    print(f"retention zero-init contrast max={out_b['retention']['contrast'].max().item():.6g}")


if __name__ == "__main__":
    main()
