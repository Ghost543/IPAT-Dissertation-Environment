"""Evaluate the approved EXP-06 protocol using frozen EXP-05/CAL-01 outputs."""
import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class Tee:
    def __init__(self,stream,path):self.stream,self.path,self.file=stream,path,None
    def write(self,text):
        self.stream.write(text)
        if self.file is None and self.path.parent.is_dir():self.file=self.path.open('a',encoding='utf-8')
        if self.file is not None:self.file.write(text);self.file.flush()
        return len(text)
    def flush(self):
        self.stream.flush()
        if self.file is not None:self.file.flush()
    def close(self):
        if self.file is not None:self.file.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','calibration','cache','output','verification'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists. Retain it and choose a new timestamped directory.')
    from ipat.experiments.exp06 import run
    tee=Tee(sys.stdout,a.output/'console.txt')
    try:
        with contextlib.redirect_stdout(tee),contextlib.redirect_stderr(tee):
            try:
                result=run(a.parent,a.calibration,a.cache,a.output,a.verification)
                print(json.dumps(result,indent=2))
            except BaseException:
                traceback.print_exc();raise SystemExit(1)
    finally:tee.close()


if __name__=='__main__':main()
