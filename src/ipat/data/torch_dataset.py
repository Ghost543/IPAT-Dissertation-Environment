from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class SyntheticTorchDataset(Dataset):
    """Torch view of fixed-width DATA-01 arrays; no preprocessing is fitted here."""

    _FIELDS = (
        "x_obs",
        "missing_mask",
        "event_type",
        "nuisance_cat",
        "delta",
        "timestamp",
        "position",
        "valid_mask",
        "signal_mask",
        "contribution",
        "p_true",
        "y",
    )

    def __init__(self, split_data: Any, indices: list[int] | None = None) -> None:
        self.data = split_data
        n = int(split_data["y"].shape[0])
        self.indices = list(range(n)) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.indices[index]
        out: dict[str, torch.Tensor] = {}
        for name in self._FIELDS:
            value = self.data[name][row]
            tensor = torch.as_tensor(value)
            if name in {"x_obs", "delta", "timestamp", "contribution", "p_true"}:
                tensor = tensor.to(torch.float32)
            elif name in {"event_type", "nuisance_cat", "position", "y"}:
                tensor = tensor.to(torch.long)
            elif name in {"missing_mask", "valid_mask", "signal_mask"}:
                tensor = tensor.to(torch.bool)
            out[name] = tensor
        return out
