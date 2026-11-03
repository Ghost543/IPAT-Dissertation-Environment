"""Verify EXP-06 on synthetic checkpoints only, before real test access."""
import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    project=Path(__file__).resolve().parents[1]
    from ipat.experiments.exp06_core import (ARMS,SEEDS,sha,write_json,save_npz,metadata,
        prediction_record,evaluate_record,replay,require,load_partition)
    try:
        suite=unittest.defaultTestLoader.discover(str(project/'tests'),pattern='test_exp06.py')
        with (a.output/'fixture_tests.txt').open('w',encoding='utf-8') as log:
            tested=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        require(tested.wasSuccessful(),'Unit tests failed; inspect fixture_tests.txt')
        import numpy as np
        import torch
        import joblib
        from ipat.experiments.exp05 import make_model,predict,save_checkpoint,configure_device,environment
        from ipat.experiments.exp05_references import tree,prefix_summaries
        from ipat.experiments.cal01 import evaluate_checkpoint
        from ipat.experiments.exp06 import test_inference,finish_reports,code_hashes
        spec=importlib.util.spec_from_file_location('exp06_fixture',project/'tests/test_exp06.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        state,train,c=mod.fixture(48,800)
        _,val,_=mod.fixture(24,801);_,cal,_=mod.fixture(24,802);_,test,_=mod.fixture(24,803)
        torch.set_num_threads(1);device=configure_device('cpu')
        parent=a.output/'synthetic_checkpoints';parent.mkdir()
        out=a.output/'synthetic_fixture';out.mkdir()
        identity='SYNTHETIC_EXP06_ONLY';records=[]
        for arm in ARMS:
            for seed in SEEDS:
                d=parent/f'{arm}_{seed}';d.mkdir()
                if arm=='GBDT':
                    # This training is confined to synthetic verification, not the evaluator.
                    model=tree(seed).fit(prefix_summaries(train,state),train.arrays['y'])
                    joblib.dump(model,d/'selected.joblib')
                    v=model.decision_function(prefix_summaries(val,state))
                    expected_cal=model.decision_function(prefix_summaries(cal,state))
                else:
                    model=make_model(arm,state,seed,c['model'])
                    save_checkpoint(d/'selected.pt',dict(model=model.state_dict(),identity=identity,epoch=1))
                    v,_=predict(model,val,device,c['microbatch'])
                    expected_cal,_=predict(model,cal,device,c['microbatch'])
                rawcal,parent_checks=evaluate_checkpoint(d,arm,seed,state,c,identity,1,val,cal,device,v)
                error=replay(rawcal,expected_cal,'synthetic calibration')
                checkpoint=d/('selected.joblib' if arm=='GBDT' else 'selected.pt')
                raw,checks=test_inference(d,arm,seed,state,c,identity,1,test,device,sha(checkpoint))
                if arm=='T00' and seed==1001:
                    for bad_identity,bad_hash in [('wrong',sha(checkpoint)),(identity,'wrong')]:
                        try:test_inference(d,arm,seed,state,c,bad_identity,1,test,device,bad_hash)
                        except ValueError:pass
                        else:raise AssertionError('Invalid checkpoint was accepted')
                    try:evaluate_checkpoint(d,arm,seed,state,c,identity,1,val,cal,device,v+1.)
                    except ValueError:pass
                    else:raise AssertionError('Invalid parent replay was accepted')
                # Known positive temperatures exercise application only; no solver called.
                t={1001:.8,1002:1.,1003:1.2}[seed]
                arrays=prediction_record(metadata(test),raw,t)
                target=out/f'{arm}_{seed}';target.mkdir()
                save_npz(target/'test_predictions.npz',**arrays)
                row=dict(arm=arm,seed=seed,temperature=t,**evaluate_record(arrays),checks=checks,
                    parent_checks=parent_checks,calibration_replay_max_logit_error=error,
                    prediction_sha256=sha(target/'test_predictions.npz'))
                write_json(target/'result.json',row);records.append(row)
                print(f'Synthetic checkpoint verified: {arm}/{seed}',flush=True)
        finish_reports(out,fixture_only=True)
        expected=['nll','paired','factorial','calibration_changes','reliability','bin_support','confusion',
                  'precision_recall','length_nll','length_recall','length_distribution']
        checks=dict(unit_tests=True,all_18_checkpoint_paths=len(records)==18,parent_replays=True,
            exact_test_replay=all(r['checks']['exact_test_replay'] for r in records),
            immutability=all(r['checks']['model_state_unchanged'] and r['checks']['checkpoint_file_unchanged'] and r['checks']['input_arrays_unchanged'] for r in records),
            negative_guards=True,saved_metric_recompute=True,
            figure_exports=all((out/'plots'/f'{n}.{ext}').is_file() for n in expected for ext in ('png','pdf','svg')),
            offline_html='data:image/svg+xml;base64,' in (out/'plots/report.html').read_text())
        require(all(checks.values()),'Incomplete synthetic verification')
        result=dict(status='PASS',fixture_only=True,scope='Synthetic verification only; no BPI17 test access',
            unit_tests_run=tested.testsRun,checks=checks,environment=environment(device),code_sha256=code_hashes())
        write_json(a.output/'exp06_runner_acceptance.json',result)
        print(json.dumps(result,indent=2))
    except BaseException as e:
        write_json(a.output/'exp06_runner_failure.json',dict(status='FAILED',error_type=type(e).__name__,error=str(e)))
        raise


if __name__=='__main__':main()
