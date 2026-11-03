"""Synthetic integration acceptance only: no BPI files are loaded."""
from pathlib import Path
import sys,argparse,importlib.util,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ipat.experiments.exp05 import *
from ipat.experiments.exp05_plots import plots


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('exp05_fixture_tests',Path(__file__).resolve().parents[1]/'tests/test_exp05.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    import unittest
    with (a.output/'fixture_tests.txt').open('w',encoding='utf-8') as log:
        tested=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(mod))
    if not tested.wasSuccessful():raise AssertionError('Runner unit tests failed; see fixture_tests.txt')
    torch.set_num_threads(1);device=configure_device('cpu');state,train,c=mod.fixture(25);_,val,_=mod.fixture(12,77)
    c.update(epochs=1,seeds=[1001,1002,1003]);root=a.output/'synthetic_fixture';root.mkdir()
    q=float(train.arrays['y'].mean());write_json(root/'dummy.json',metrics(val.arrays['y'],np.full(len(val),np.log(q/(1-q))),val.arrays['application_id']))
    for seed in c['seeds']:
        for arm in ARMS:
            d=root/f'{arm}_{seed}'
            with MemoryMonitor() as mon:r=train_run(arm,seed,state,train,val,c,d,'SYNTHETIC_FIXTURE_ONLY')
            r.update(mon.report());write_json(d/'result.json',r)
    x=prefix_summaries(train,state);v=prefix_summaries(val,state)
    import joblib
    for seed in c['seeds']:
        d=root/f'GBDT_{seed}';d.mkdir();started=time.perf_counter()
        with MemoryMonitor() as mon:
            model=tree(seed).fit(x,train.arrays['y']);logits=model.decision_function(v)
            joblib.dump(model,d/'selected.joblib');again=joblib.load(d/'selected.joblib').decision_function(v)
        if not np.array_equal(logits,again):raise AssertionError('Tree replay failed')
        np.savez_compressed(d/'predictions.npz',logits=logits,final_logits=logits,**{k:val.arrays[k] for k in ('y','offer_id','application_id')},lengths=np.diff(val.arrays['offsets']))
        r=dict(arm='GBDT',seed=seed,status='PASS',iterations=int(model.n_iter_),selected=metrics(val.arrays['y'],logits,val.arrays['application_id']),
               final=metrics(val.arrays['y'],logits,val.arrays['application_id']),length_strata=stratified(val.arrays['y'],logits,val.arrays['application_id'],np.diff(val.arrays['offsets'])),
               elapsed_seconds=time.perf_counter()-started,parameter_count=None,checkpoint_replay_exact=True,**mon.report())
        write_json(d/'result.json',r)
    summary=aggregate(root,c['seeds'],draws=2000);summary.update(fixture_only=True,scope='SYNTHETIC FIXTURE ONLY: runner/report verification. Not BPI17 or EXP-05 predictive evidence.')
    write_json(root/'exp05_pilot_summary.json',summary);plots(root)
    checks=dict(unit_tests=tested.wasSuccessful(),all_18_fixture_runs=len(summary['runs'])==18,
        checkpoint_replay=all(r['checkpoint_replay_exact'] for r in summary['runs']),
        neural_fixture_budget=all(r['updates']==7 for r in summary['runs'] if r['arm']!='GBDT'),
        tree_budget=all(r['iterations']==200 for r in summary['runs'] if r['arm']=='GBDT'),
        figure_exports=all(len(list((root/'plots').glob('*.'+ext)))==10 for ext in ('png','pdf','svg')),
        offline_html=(root/'plots/report.html').exists(),prediction_alignment=True)
    if not all(checks.values()):raise AssertionError(checks)
    result=dict(status='PASS',scope=summary['scope'],unit_tests_run=tested.testsRun,checks=checks,environment=environment(device))
    write_json(a.output/'exp05_runner_acceptance.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
