"""Run from the dissertation project root after local verifier acceptance."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ipat.experiments.cal01 import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True,help='Completed EXP-05 pilot directory')
    p.add_argument('--cache',type=Path,required=True,help='Frozen BPI17 packed cache')
    p.add_argument('--output',type=Path,required=True,help='New CAL-01 output directory')
    a=p.parse_args()
    result=run(a.parent,a.cache,a.output)
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':
        raise SystemExit('CAL-01 produced review cases; preserve outputs and review before test evaluation')

if __name__=='__main__':main()
