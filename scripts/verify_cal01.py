"""Synthetic local acceptance: no real BPI17 partition or pilot is accessed."""
import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    project=Path(__file__).resolve().parents[1]
    suite=unittest.defaultTestLoader.discover(str(project/'tests'),pattern='test_cal01.py')
    with (a.output/'fixture_tests.txt').open('w',encoding='utf-8') as log:
        tested=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    if not tested.wasSuccessful():raise SystemExit('Numerical tests failed; inspect fixture_tests.txt')
    import numpy as np
    import torch
    import joblib
    from ipat.experiments.exp05 import make_model,predict,save_checkpoint,write_json,configure_device,environment
    from ipat.experiments.exp05_references import tree,prefix_summaries
    from ipat.experiments.exp05_metrics import metrics
    from ipat.experiments.cal01 import evaluate_checkpoint,ARMS,SEEDS
    from ipat.experiments.cal01_core import fit_temperature,scale_logits,load_partition
    from ipat.experiments.cal01_plots import plots
    from ipat.data.bpi17 import logical_hash,sha256
    # Existing EXP-05 fixture builds only synthetic arrays, never reads real data.
    spec=importlib.util.spec_from_file_location('cal01_exp05_fixture',project/'tests/test_exp05.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    torch.set_num_threads(1);device=configure_device('cpu')
    state,val,c=mod.fixture(24,703);_,cal,_=mod.fixture(24,704)
    root=a.output/'synthetic_fixture';root.mkdir()
    cache=a.output/'fixture_cache';cache.mkdir()
    np.savez_compressed(cache/'calibration.npz',**cal.arrays)
    if len(load_partition(cache,'calibration',logical_hash(cal.arrays)))!=len(cal):raise AssertionError('Cache reload')
    try:load_partition(cache,'calibration','wrong-hash')
    except ValueError:pass
    else:raise AssertionError('Modified cache hash accepted')
    rows=[];identity='SYNTHETIC_CAL01_ONLY'
    for seed in SEEDS:
        for arm in ARMS:
            d=root/f'{arm}_{seed}';d.mkdir()
            if arm=='GBDT':
                model=tree(seed).fit(prefix_summaries(val,state),val.arrays['y'])
                joblib.dump(model,d/'selected.joblib')
                saved=model.decision_function(prefix_summaries(val,state))
            else:
                model=make_model(arm,state,seed,c['model'])
                save_checkpoint(d/'selected.pt',dict(model=model.state_dict(),identity=identity,epoch=1))
                saved,_=predict(model,val,device,c['microbatch'])
                if arm=='T00' and seed==SEEDS[0]:
                    try:evaluate_checkpoint(d,arm,seed,state,c,'wrong-identity',1,val,cal,device,saved)
                    except ValueError:pass
                    else:raise AssertionError('Changed checkpoint identity accepted')
            raw,checks=evaluate_checkpoint(d,arm,seed,state,c,identity,1,val,cal,device,saved)
            if arm=='T00' and seed==SEEDS[0]:
                try:evaluate_checkpoint(d,arm,seed,state,c,identity,1,val,cal,device,saved+1.)
                except ValueError:pass
                else:raise AssertionError('Changed parent predictions accepted')
            fit=fit_temperature(cal.arrays['y'],raw);scaled=scale_logits(raw,fit['temperature'])
            before=metrics(cal.arrays['y'],raw);after=metrics(cal.arrays['y'],scaled)
            if before['confusion_matrix']!=after['confusion_matrix']:raise AssertionError('Changed decisions')
            np.savez_compressed(d/'calibration_predictions.npz',raw_logits=raw,scaled_logits=scaled,y=cal.arrays['y'])
            rows.append(dict(arm=arm,seed=seed,fit=fit,raw=before,scaled=after,checks=checks))
    write_json(root/'cal01_summary.json',dict(fixture_only=True,runs=rows,status='FIXTURE',scope='SYNTHETIC ONLY'))
    figures=plots(root)
    exports=all((root/'plots'/(name+'.'+ext)).is_file() for name,_ in figures for ext in ('png','pdf','svg'))
    checks=dict(numerical_tests=tested.wasSuccessful(),all_18_checkpoint_paths=len(rows)==18,
        model_and_file_immutability=all(r['checks']['model_state_unchanged'] and r['checks']['checkpoint_file_unchanged'] for r in rows),
        calibration_replay_exact=all(r['checks']['calibration_replay_exact'] for r in rows),
        cache_mismatch_rejected=True,checkpoint_identity_mismatch_rejected=True,validation_mismatch_rejected=True,
        class_decisions_unchanged=True,figure_exports=exports,offline_html=(root/'plots/report.html').is_file())
    if not all(checks.values()):raise AssertionError(checks)
    result=dict(status='PASS',scope='SYNTHETIC local runner acceptance only; no BPI17 temperature fits',
        unit_tests_run=tested.testsRun,checks=checks,environment=environment(device),
        cal01_code_sha256={name:sha256(project/'src/ipat/experiments'/name) for name in ('cal01.py','cal01_core.py','cal01_plots.py')})
    write_json(a.output/'cal01_runner_acceptance.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
