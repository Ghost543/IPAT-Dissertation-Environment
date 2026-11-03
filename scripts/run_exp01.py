from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ipat.experiments.exp01 import Exp01Config, run_exp01, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-01 matched tiny-sample sanity experiment")
    parser.add_argument("--train", default="data/processed/ipat_syn_v0.1/train.h5")
    parser.add_argument("--config", default="configs/experiments/exp01_tiny_sample_v0.1.yaml")
    parser.add_argument("--out", default="results/exp01/exp01_tiny_sample_v0.1.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = Exp01Config(experiment_id=payload["experiment_id"], **payload["protocol"])
    result = run_exp01(args.train, cfg, device=args.device)

    parity = result["initial_zero_effect_parity"]
    print(
        "Initial matched parity: "
        f"within_tolerance={parity['within_tolerance']} atol={parity['atol']:.1e} "
        f"max_logit_diff={parity['max_logit_abs_diff']:.3g} "
        f"max_attention_diff={parity['max_attention_abs_diff']:.3g}"
    )
    print(f"Subset indices: {result['subset_indices']}")
    print(
        f"Subset prevalence: {result['subset_positive_count']}/{result['subset_size']} "
        f"({result['subset_prevalence']:.3f})"
    )

    for name in ("baseline", "retention"):
        r = result[name]
        print(f"\n{name.upper()}")
        for row in r["trajectory"]:
            suffix = ""
            if "retention_contrast" in row:
                suffix = f" retention_contrast={row['retention_contrast']:.6f}"
            print(
                f"step={row['step']:03d} loss={row['loss']:.6f} "
                f"accuracy={row['accuracy']:.3f}{suffix}"
            )
        print(
            f"PASS={r['passed']} reached_step={r['reached_step']} "
            f"final_loss={r['final_loss']:.6f} final_accuracy={r['final_accuracy']:.3f}"
        )

    path = save_result(result, args.out)
    print(f"\nWrote: {path.resolve()}")
    if result["status"] != "PASS":
        raise SystemExit("EXP-01 FAIL: both matched models did not satisfy the tiny-sample gate")
    print("PASS EXP-01: both matched models satisfied the tiny-sample trainability gate.")
    print("Interpretation: implementation sanity only; no generalisation/superiority claim.")


if __name__ == "__main__":
    main()
