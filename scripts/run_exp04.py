"""Evaluate accepted EXP-04 protocols and automatically generate plots/HTML."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import yaml
from ipat.experiments.exp04 import run_exp04, read_signed
from ipat.experiments.exp04_contract import Exp04Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/exp04_corruption_retention_v0.1.yaml")
    parser.add_argument("--parent-dir", default="results/exp03/v0.1")
    parser.add_argument("--data-dir", default="data/processed/ipat_syn_v0.1")
    parser.add_argument("--out-dir", default="results/exp04/v0.1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rebuild figures from an existing signed summary; no inference")
    args = parser.parse_args()
    if args.report_only:
        from ipat.experiments.exp04_report import write_report
        summary = read_signed(Path(args.out_dir)/"exp04_corruption_retention_v0.1.json")
        write_report(summary, args.out_dir)
        print(f"Report rebuilt: {Path(args.out_dir).resolve()/'exp04_report.html'}")
        return
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    if payload.get("experiment_id") != "EXP-04" or payload.get("primary_endpoint") != "raw_nll_oriented_macro_change":
        raise ValueError("Wrong EXP-04 configuration/primary endpoint")
    result = run_exp04(args.parent_dir, args.data_dir, args.out_dir, Exp04Config(**payload["protocol"]),
                      device=args.device, resume=args.resume, preflight_only=args.preflight_only)
    print(f"{result['status']}: {Path(args.out_dir).resolve()}")
    print("Execution success does not establish the research hypothesis. Review the evidence before freezing.")


if __name__ == "__main__":
    main()
