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
from ipat.model.adaptive import FullIPATConfig, FullIPATTransformer
from ipat.training.preprocessing import fit_baseline_preprocessing_stats


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="IMP-03 full-IPAT trainability smoke test")
    parser.add_argument("--train", default="data/processed/ipat_syn_v0.1/train.h5")
    parser.add_argument("--config", default="configs/model/full_ipat_v0.1.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = FullIPATConfig(**payload["model"])
    acceptance = payload["acceptance"]
    seed = int(acceptance["seed"])
    seed_everything(seed)

    split = load_hdf5(args.train)
    stats = fit_baseline_preprocessing_stats(split)
    subset_size = min(int(acceptance["subset_size"]), int(split["y"].shape[0]))
    dataset = SyntheticTorchDataset(split, indices=list(range(subset_size)))
    batch = next(iter(DataLoader(dataset, batch_size=subset_size, shuffle=False)))
    device = torch.device(args.device)
    batch = {key: value.to(device) for key, value in batch.items()}

    model = FullIPATTransformer(
        replace(cfg, dropout=float(acceptance["dropout_override"])), stats
    ).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(acceptance["learning_rate"]),
        weight_decay=float(acceptance["weight_decay"]),
    )
    y = batch["y"].float()
    target_loss = float(acceptance["target_loss"])
    target_accuracy = float(acceptance["target_accuracy"])

    first_loss = None
    for step in range(1, int(acceptance["steps"]) + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        out = model(batch, return_diagnostics=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], y)
        loss.backward()
        optimiser.step()
        if first_loss is None:
            first_loss = float(loss.item())
        with torch.no_grad():
            accuracy = float(((out["logits"] >= 0).long() == batch["y"]).float().mean())
            contrast = float(out["retention"]["contrast"].mean())
            identity_weight = float(
                out["components"]["mixture_weights"][:, 0].mean()
            )
        if step == 1 or step % 25 == 0:
            print(
                f"step={step:03d} loss={loss.item():.6f} accuracy={accuracy:.3f} "
                f"retention_contrast={contrast:.6f} mean_identity_weight={identity_weight:.6f}"
            )
        if loss.item() <= target_loss and accuracy >= target_accuracy:
            print(
                f"PASS: full IPAT tiny-sample overfit at step {step}; "
                f"loss={loss.item():.6f}, accuracy={accuracy:.3f}, "
                f"first_loss={first_loss:.6f}, retention_contrast={contrast:.6f}, "
                f"mean_identity_weight={identity_weight:.6f}"
            )
            return
    raise SystemExit(
        f"FAIL: did not reach loss<={target_loss} and accuracy>={target_accuracy} "
        f"within {acceptance['steps']} steps"
    )


if __name__ == "__main__":
    main()
