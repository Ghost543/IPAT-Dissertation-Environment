"""Full local synthetic verifier. Does not read real BPI17 data or results."""
import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    from ipat.experiments.exp06_core import ARMS,SEEDS,sha,write_json,require,load_npz,read_json
    try:
        suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_rob01.py')
        with (a.output/'fixture_tests.txt').open('w',encoding='utf-8') as log:tested=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        require(tested.wasSuccessful() and tested.testsRun>=26,'Synthetic unit tests failed or incomplete')
        import numpy as np
        import torch
        import joblib
        from ipat.experiments.exp05 import make_model,predict,save_checkpoint,configure_device,environment
        from ipat.experiments.exp05_references import tree,prefix_summaries
        from ipat.experiments.exp06 import test_inference
        from ipat.experiments.rob01 import evaluate_grid,code_hashes,dependencies
        spec=importlib.util.spec_from_file_location('rob01_fixture',root/'tests/test_rob01.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        state,train,config=mod.fixture(30,7001);_,data,_=mod.fixture(4,7002)
        torch.set_num_threads(1);device=configure_device('cpu')
        parent=a.output/'synthetic_checkpoints';parent.mkdir();out=a.output/'synthetic_fixture';out.mkdir()
        identity='SYNTHETIC_ROB01_ONLY';rows=[];expected={};guards=0
        for arm in ARMS:
            for seed in SEEDS:
                d=parent/f'{arm}_{seed}';d.mkdir()
                if arm=='GBDT':
                    model=tree(seed).fit(prefix_summaries(train,state),train.arrays['y'])
                    joblib.dump(model,d/'selected.joblib');file=d/'selected.joblib'
                    raw=model.decision_function(prefix_summaries(data,state));epoch=None
                else:
                    model=make_model(arm,state,seed,config['model']);epoch=1
                    save_checkpoint(d/'selected.pt',dict(model=model.state_dict(),identity=identity,epoch=epoch));file=d/'selected.pt'
                    raw,_=predict(model,data,device,config['microbatch'])
                row=dict(arm=arm,seed=seed,temperature={1001:.8,1002:1.,1003:1.2}[seed],
                    selected_epoch=epoch,checkpoint_sha256=sha(file))
                rows.append(row);expected[arm,seed]=dict(raw_logits=np.asarray(raw))
                if arm=='T00' and seed==1001:
                    for badidentity,badhash in [('wrong',sha(file)),(identity,'wrong')]:
                        try:test_inference(d,arm,seed,state,config,badidentity,epoch,data,device,badhash)
                        except ValueError:guards+=1
                        else:raise AssertionError('Wrong checkpoint accepted')
        before={str(p):sha(p) for p in parent.rglob('*') if p.is_file()}
        result=evaluate_grid(parent,state,config,identity,device,rows,data,out,expected,fixture_only=True)
        zero=read_json(out/'zero_replay.json')
        rr=[read_json(p) for p in (out/'conditions').glob('*/*/result.json')]
        checks=dict(unit_tests=True,all_18_checkpoint_paths=len(zero)==18,
            all_324_corrupt_evaluations=len(rr)==result['corrupted_evaluations']==324,
            zero_corruption_replay=all(r['max_logit_error']<=1e-6 for r in zero),
            immutability=before=={p:sha(p) for p in before} and all(all(r['checks'].values()) for r in rr),
            saved_metric_recompute=result['status']=='PASS',negative_guards=guards==2,
            figure_exports=len(result['figures'])==7 and all((out/'plots'/f'{n}.{e}').is_file() for n in result['figures'] for e in ('png','pdf','svg')),
            offline_html='data:image/svg+xml;base64,' in (out/'plots/report.html').read_text())
        require(all(checks.values()),'Incomplete integration verification')
        receipt=dict(status='PASS',fixture_only=True,scope='Synthetic checkpoints only; no BPI17 access',
            unit_tests_run=tested.testsRun,checks=checks,environment=environment(device),code_sha256=code_hashes(),dependency_sha256=dependencies())
        write_json(a.output/'rob01_runner_acceptance.json',receipt)
        print(json.dumps(receipt,indent=2))
    except BaseException as error:
        write_json(a.output/'rob01_runner_failure.json',dict(status='FAILED',error_type=type(error).__name__,error=str(error)))
        raise

if __name__=='__main__':main()
