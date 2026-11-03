from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Works from PowerShell without editable-install or shell-specific PYTHONPATH changes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml

from ipat.experiments.exp03 import run_exp03
from ipat.experiments.exp03_contract import Exp03Config
from ipat.model.baseline import BaselineConfig


def main():
    parser = argparse.ArgumentParser(description="EXP-03 matched informative-event position sweep")
    parser.add_argument("--config", default="configs/experiments/exp03_position_sweep_v0.1.yaml")
    parser.add_argument("--train", default="data/processed/ipat_syn_v0.1/train.h5")
    parser.add_argument("--val", default="data/processed/ipat_syn_v0.1/val.h5")
    parser.add_argument("--sweep", default="data/processed/ipat_syn_v0.1/position_sweep.h5")
    parser.add_argument("--out-dir", default="results/exp03/v0.1")
    parser.add_argument("--device", default="cpu", help="cpu (reproducible default) or cuda")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed runs")
    parser.add_argument("--preflight-only", action="store_true", help="Validate data without training")
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    result = run_exp03(args.train, args.val, args.sweep, args.out_dir,
                      Exp03Config(**payload["protocol"]), BaselineConfig(**payload["model"]),
                      device=args.device, resume=args.resume, preflight_only=args.preflight_only)
    print(f"{result['status']}: {Path(args.out_dir).resolve()}")
    print("Review the saved evidence before freezing EXP-03. No superiority claim is automatic.")


if __name__ == "__main__":
    main()
