"""ROB-01 frozen inference. Never trains, fits preprocessing or fits temperatures."""
from pathlib import Path
import importlib
import time
import numpy as np
from .exp06_core import (ARMS,SEEDS,sha,read_json,write_json,save_npz,load_npz,require,
    metadata,aligned,disjoint,load_partition,replay,evaluate_record)
from .rob01_corruptions import SourceIndex,conditions,condition_id
from .rob01_core import save_record


def project_root(): return Path(__file__).resolve().parents[3]


def code_hashes():
    root=project_root()
    names=['src/ipat/experiments/'+n+'.py' for n in ('rob01','rob01_corruptions','rob01_core','rob01_plots','rob01_binding')]
    names+=['scripts/run_rob01.py','scripts/verify_rob01.py','tests/test_rob01.py',
        'configs/rob01_protocol_v0.1.json','docs/rob01/ROB01_Protocol_v0.1.md']
    return {n:sha(root/n) for n in names}


def verify_receipt(path):
    r=read_json(path)
    require(r.get('status')=='PASS' and r.get('fixture_only') is True,'Run local synthetic ROB-01 verification first')
    needed=('unit_tests','all_18_checkpoint_paths','all_324_corrupt_evaluations','zero_corruption_replay',
        'immutability','saved_metric_recompute','negative_guards','figure_exports','offline_html')
    require(all(r.get('checks',{}).get(k) is True for k in needed),'Incomplete local verifier')
    require(r.get('code_sha256')==code_hashes(),'ROB-01 changed since local verification')
    require(r.get('dependency_sha256')==dependencies(),'Dependency files changed since verification')


def dependencies():
    from .exp06 import CAL_CODE,code_hashes as exp06_hashes
    d={'exp06/'+k:v for k,v in exp06_hashes().items()}
    for name in CAL_CODE:
        d['src/ipat/experiments/'+name]=sha(project_root()/'src/ipat/experiments'/name)
    # Module hashes only, with no device or runtime mutation.
    for module in ('ipat.model.baseline','ipat.model.adaptive','ipat.model.retention','ipat.model.bpi17',
                   'ipat.data.bpi17','ipat.experiments.exp05','ipat.experiments.exp05_io',
                   'ipat.experiments.exp05_metrics','ipat.experiments.exp05_references','ipat.experiments.exp05_plots'):
        d[module]=sha(Path(importlib.import_module(module).__file__))
    return d


def validate_clean(clean, rows, data, spec):
    """Bind the already reviewed clean evidence and each saved prediction."""
    from .exp06 import code_hashes as exp06_hashes
    ca=read_json(clean/'exp06_acceptance.json');cs=read_json(clean/'exp06_summary.json');cm=read_json(clean/'exp06_manifest.json')
    require(ca.get('status')==cs.get('status')=='PASS' and cs.get('fixture_only') is False,'Real clean evaluation required')
    needed=('protocol_bound','parents_bound','cohort_integrity','frozen_execution','replay_verified',
            'all_18_pairs','numerical_validity','reports_complete','inputs_unchanged')
    require(all(ca.get('checks',{}).get(k) is True for k in needed),'Incomplete clean acceptance')
    for name,expected in spec['reviewed_evidence_sha256'].items():
        require(sha(clean/name)==expected,'Reviewed clean evidence changed: '+name)
    for key,name in [('summary_sha256','exp06_summary.json'),('manifest_sha256','exp06_manifest.json'),('checksums_sha256','checksums.json')]:
        require(ca[key]==sha(clean/name),'Clean evidence digest mismatch: '+name)
    require(cm['code_sha256']==exp06_hashes(),'EXP-06 implementation differs from accepted clean evaluation')
    inventory=read_json(clean/'checksums.json')['outputs']
    for name,h in inventory.items():
        p=(clean/name).resolve();require(clean in p.parents,'Unsafe clean output inventory path')
        require(sha(p)==h,'Clean output changed: '+name)
    require(len(cs['runs'])==18 and len(cm['frozen_runs'])==18,'Clean run count differs')
    lookup={(r['arm'],r['seed']):r for r in cs['runs']}
    frozen={(r['arm'],r['seed']):r for r in cm['frozen_runs']}
    require(set(lookup)==set(frozen)=={(a,s) for a in ARMS for s in SEEDS},'Clean run coverage differs')
    meta=metadata(data);predictions={}
    require(len(data)==8613 and len(np.unique(meta['application_id']))==6356,'Real test cohort differs')
    require(cs['n']==8613 and cs['applications']==6356 and cs['fitted_on_test'] is False,'Clean summary scope differs')
    for row in rows:
        arm,seed=row['arm'],row['seed'];r=lookup[arm,seed];fr=frozen[arm,seed]
        for k in ('temperature','selected_epoch','checkpoint_sha256','temperature_sha256'):
            require(row[k]==r[k]==fr[k],'Clean/parent binding differs: '+k)
        file=clean/f'{arm}_{seed}'/'test_predictions.npz'
        require(sha(file)==r['prediction_sha256'],'Clean predictions changed')
        a=load_npz(file);aligned(a,meta)
        from .cal01_core import scale_logits
        require(np.array_equal(scale_logits(a['raw_logits'],row['temperature']),a['scaled_logits']),'Clean temperature mismatch')
        for k,v in evaluate_record(a).items():require(r[k]==v,'Clean metric/prediction mismatch')
        predictions[arm,seed]=a
    return cm,predictions


def evaluate_grid(parent,state,config,identity,device,rows,data,output,expected_clean=None,fixture_only=False):
    """Used identically by real runner and synthetic integration verifier."""
    from .exp06 import test_inference
    from .rob01_plots import finish
    output=Path(output);index=SourceIndex(data,state);lengths=metadata(data)['lengths']
    # Both family zero transformations must be byte-identical to the clean arrays.
    for family in ('missingness','deletion'):
        for rep in (1,2,3):index.transform(family,0,rep)
    zero=[]
    for row in rows:
        arm,seed=row['arm'],row['seed']
        raw,checks=test_inference(parent/f'{arm}_{seed}',arm,seed,state,config,identity,
            row.get('selected_epoch'),data,device,row['checkpoint_sha256'])
        error=replay(raw,expected_clean[arm,seed]['raw_logits'],'EXP-06 clean') if expected_clean is not None else 0.
        # Anchor changes to the original saved clean predictions, not tolerance-shifted reruns.
        anchor=expected_clean[arm,seed]['raw_logits'] if expected_clean is not None else raw
        save_record(output/'clean'/f'{arm}_{seed}',data,anchor,dict(row,checks=checks,clean_replay_max_logit_error=error),lengths)
        zero.append(dict(arm=arm,seed=seed,max_logit_error=error,checks=checks))
        print(f'Zero corruption verified: {arm}/{seed}',flush=True)
    write_json(output/'zero_replay.json',zero)
    count=0
    for family,severity,rep in conditions():
        name=condition_id(family,severity,rep);directory=output/'conditions'/name;directory.mkdir(parents=True)
        corrupted,audit,plan=index.transform(family,severity,rep)
        save_npz(directory/'plan.npz',**plan);audit['plan_sha256']=sha(directory/'plan.npz')
        write_json(directory/'audit.json',audit)
        for row in rows:
            arm,seed=row['arm'],row['seed'];started=time.perf_counter()
            raw,checks=test_inference(parent/f'{arm}_{seed}',arm,seed,state,config,identity,
                row.get('selected_epoch'),corrupted,device,row['checkpoint_sha256'])
            save_record(directory/f'{arm}_{seed}',corrupted,raw,
                dict(row,family=family,severity=severity,replicate=rep,checks=checks,
                     elapsed_seconds=time.perf_counter()-started,corruption_logical_sha256=audit['output_logical_sha256']),lengths)
            count+=1
            print(f'[{count}/324] {name}: {arm}/{seed}',flush=True)
        del corrupted
    require(count==324,'Incomplete condition grid')
    return finish(output,fixture_only)


def run(parent,calibration,cache,clean,output,verification):
    from .exp06 import preflight,input_files,CAL_CODE
    from .exp05 import digest
    from .rob01_binding import CONFIG_SHA,PROTOCOL_SHA
    parent,calibration,cache,clean,output=[Path(p).resolve() for p in (parent,calibration,cache,clean,output)]
    require(not output.exists(),'Use a fresh timestamped output directory; retain earlier attempts')
    for p in (parent,calibration,cache,clean):
        require(p!=output and p not in output.parents and output not in p.parents,'Output overlaps input')
    output.mkdir(parents=True,exist_ok=False);phase='preflight'
    try:
        verify_receipt(verification)
        root=project_root();cfg=root/'configs/rob01_protocol_v0.1.json';doc=root/'docs/rob01/ROB01_Protocol_v0.1.md'
        require(sha(cfg)==CONFIG_SHA and sha(doc)==PROTOCOL_SHA,'Accepted protocol/configuration changed')
        spec=read_json(cfg)
        modules=code_hashes();deps=dependencies()
        paths=input_files(parent,calibration,cache)
        paths += [p for p in clean.rglob('*') if p.is_file() and p.name!='console.txt']
        inputs={str(p):sha(p) for p in paths}
        manifest,state,device,rows,partitions=preflight(parent,calibration,cache)
        data=load_partition(cache,'test',manifest['config']['logical_hashes']['test'])
        disjoint(dict(partitions,test=metadata(data)))
        cm,predictions=validate_clean(clean,rows,data,spec)
        require(cm['environment']==manifest['environment'],'Clean and parent environments differ')
        receipt=dict(version='ROB01-v0.1',protocol_accepted='2026-09-23',scope=spec['scope'],
            parent=str(parent),calibration=str(calibration),cache=str(cache),clean=str(clean),
            code_sha256=modules,dependency_sha256=deps,input_sha256=inputs,config=spec,
            verification_sha256=sha(verification),environment=manifest['environment'],frozen_runs=rows,
            training=False,temperature_fitting=False,threshold_fitting=False,checkpoint_selection=False)
        write_json(output/'rob01_manifest.json',receipt)
        phase='corruption_and_inference'
        evaluate_grid(parent,state,manifest['config'],digest(manifest),device,rows,data,output,predictions)
        phase='final_integrity'
        require(inputs=={p:sha(p) for p in inputs},'Frozen input changed during run')
        require(modules==code_hashes() and deps==dependencies(),'Source changed during run')
        checks=dict(protocol_bound=True,parents_bound=True,clean_replay=True,all_324_evaluations=True,
            paired_source_corruptions=True,original_length_strata=True,saved_metrics_recomputed=True,
            reports_complete=True,inputs_unchanged=True,frozen_execution=True)
        outputs={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='console.txt'}
        write_json(output/'checksums.json',dict(inputs=inputs,outputs=outputs))
        acceptance=dict(status='PASS',scope='Execution validity only; scientific advantage assessed separately',checks=checks,
            summary_sha256=sha(output/'rob01_summary.json'),manifest_sha256=sha(output/'rob01_manifest.json'),checksums_sha256=sha(output/'checksums.json'))
        write_json(output/'rob01_acceptance.json',acceptance);return acceptance
    except BaseException as error:
        try:
            if (output/'rob01_summary.json').exists():
                r=read_json(output/'rob01_summary.json');r['status']='FAILED';write_json(output/'rob01_summary.json',r)
            write_json(output/'rob01_failure.json',dict(status='FAILED',phase=phase,error_type=type(error).__name__,error=str(error)))
        except Exception:pass
        raise
