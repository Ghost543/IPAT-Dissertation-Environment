"""Run from the dissertation project; paths are relative to its root."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ipat.experiments.exp05 import configure_device,load_inputs,preflight,pilot


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','pilot'])
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/exp05_pilot_v0.1.json');p.add_argument('--device',default='auto')
    p.add_argument('--microbatch',type=int,default=64);p.add_argument('--source-review');p.add_argument('--preflight');p.add_argument('--resume',action='store_true')
    a=p.parse_args();c=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    # Budget changes require a new protocol; these defaults are not HPO CLI switches.
    if c['epochs']!=20 or c['seeds']!=[1001,1002,1003] or c['batch_size']!=64:raise ValueError('Accepted budget changed; version protocol explicitly')
    if not 1<=a.microbatch<=64:raise ValueError('Microbatch must be 1..64')
    c['microbatch']=a.microbatch;device=configure_device(a.device)
    if a.mode=='preflight':
        s,t,_=load_inputs(a.cache,c,validation=False);preflight(s,t,c,a.output,device)
    else:
        if not a.source_review or not a.preflight:p.error('pilot requires --source-review and --preflight')
        pilot(a.cache,c,a.output,device,a.source_review,a.preflight,a.resume)

if __name__=='__main__':main()
