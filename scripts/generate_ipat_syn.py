from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ipat.data.synthetic import (
    IPATSyntheticConfig,
    apply_event_deletion,
    apply_missingness_corruption,
    apply_numerical_noise,
    apply_temporal_jitter,
    compute_train_numeric_scale,
    generate_position_sweep,
    generate_split,
    save_hdf5,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IPAT-SYN-v0.1.")
    parser.add_argument("--config", default="configs/data/ipat_syn_v0.1.yaml")
    parser.add_argument("--out", default="data/processed/ipat_syn_v0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = IPATSyntheticConfig(**payload["synthetic"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    debug_n = {"train": 256, "val": 64, "cal": 64, "test": 128}
    splits = {}
    for split in ("train", "val", "cal", "test"):
        data = generate_split(cfg, split, n_sequences=debug_n[split] if args.debug else None)
        splits[split] = data
        save_hdf5(data, out / f"{split}.h5")
        print(split, json.dumps(summarize(data), indent=2))

    scale = compute_train_numeric_scale(splits["train"])
    test = splits["test"]

    for rate in payload["corruptions"]["missingness_target_rates"]:
        save_hdf5(
            apply_missingness_corruption(test, float(rate), seed=10100 + int(rate * 100)),
            out / f"test_missing_{int(rate*100):02d}.h5",
        )
    for frac in payload["corruptions"]["event_deletion_fractions"]:
        save_hdf5(
            apply_event_deletion(test, float(frac), seed=10200 + int(frac * 100)),
            out / f"test_delete_{int(frac*100):02d}.h5",
        )
    for q in payload["corruptions"]["numerical_noise_q"]:
        save_hdf5(
            apply_numerical_noise(test, float(q), scale, seed=10300 + int(q * 100)),
            out / f"test_noise_{int(q*100):02d}.h5",
        )
    for q in payload["corruptions"]["temporal_jitter_q"]:
        save_hdf5(
            apply_temporal_jitter(test, float(q), seed=10400 + int(q * 100)),
            out / f"test_timejitter_{int(q*100):02d}.h5",
        )

    n_triplets = 64 if args.debug else int(payload["pilot"]["position_sweep_triplets"])
    save_hdf5(
        generate_position_sweep(cfg, n_triplets=n_triplets, seed=cfg.master_seed + 77),
        out / "position_sweep.h5",
    )

    manifest = {
        "dataset_id": cfg.dataset_id,
        "config_file": args.config,
        "debug": args.debug,
        "train_numeric_scale": scale.tolist(),
        "files": sorted(p.name for p in out.glob("*.h5")),
    }
    (out / "generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote dataset to {out.resolve()}")


if __name__ == "__main__":
    main()
