"""Install ROB-01 additions only; back up differing earlier ROB-01 files."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path.cwd());a=p.parse_args()
    root=a.project.resolve();kit=Path(__file__).resolve().parent
    for n in ('pyproject.toml','src/ipat/data/bpi17.py','src/ipat/experiments/exp05.py',
              'src/ipat/experiments/cal01.py','src/ipat/experiments/exp06.py',
              'src/ipat/experiments/exp06_core.py','scripts/verify_exp06.py'):
        if not (root/n).is_file():raise SystemExit('Missing installed prerequisite: '+n)
    inventory=json.loads((kit/'PACKAGE_SHA256.json').read_text())
    for n,h in inventory.items():
        file=(kit/n).resolve()
        if kit not in file.parents or not file.is_file() or sha(file)!=h:raise SystemExit('Package integrity failure: '+n)
    allowed={*(f'src/ipat/experiments/{n}.py' for n in ('rob01','rob01_core','rob01_corruptions','rob01_plots','rob01_binding')),
        'scripts/run_rob01.py','scripts/verify_rob01.py','scripts/Run_ROB01.ps1','tests/test_rob01.py',
        'configs/rob01_protocol_v0.1.json','docs/rob01/ROB01_Protocol_v0.1.md','docs/rob01/IMPLEMENTATION_RECORD.md'}
    if not allowed.issubset(inventory):raise SystemExit('Package missing required additions')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');written=[];backups=[]
    for n in sorted(allowed):
        src,dest=kit/n,root/n
        if dest.exists():
            if src.read_bytes()==dest.read_bytes():continue
            backup=root/'backups'/('rob01_'+stamp)/n;backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(dest,backup);backups.append(str(backup))
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);written.append(n)
    print(json.dumps(dict(status='INSTALLED',files=written,backups=backups,parent_modules_modified=False,dependencies_changed=False),indent=2))

if __name__=='__main__':main()
