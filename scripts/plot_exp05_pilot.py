import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ipat.experiments.exp05_plots import plots
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args();plots(a.results)
