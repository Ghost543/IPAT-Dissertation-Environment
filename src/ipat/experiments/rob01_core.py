"""Saved-prediction ROB-01 metrics; paired, hierarchical aggregation."""
from pathlib import Path
import numpy as np
from .exp06_core import (ARMS,SEEDS,VARIANTS,sha,read_json,write_json,save_npz,load_npz,
    require,metadata,aligned,prediction_record,evaluate_record,flatten,csv_write,
    cluster_draws,cluster_interval)
from .exp05_metrics import losses
from .cal01_core import scale_logits
from .rob01_corruptions import FAMILIES,REPLICATES,conditions,condition_id


def save_record(directory, data, raw, row, original_lengths):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    meta=metadata(data);actual=meta['lengths'].copy()
    meta['lengths']=np.asarray(original_lengths,dtype=np.int64)
    a=prediction_record(meta,raw,row['temperature'])
    a['corrupted_lengths']=actual
    save_npz(directory/'predictions.npz',**a)
    record=dict(row,**evaluate_record(a),prediction_sha256=sha(directory/'predictions.npz'))
    write_json(directory/'result.json',record)
    return record


def load_record(directory, ref=None):
    directory=Path(directory);r=read_json(directory/'result.json');a=load_npz(directory/'predictions.npz')
    require(r['prediction_sha256']==sha(directory/'predictions.npz'),'Saved prediction digest changed')
    if ref is not None: aligned(a,ref)
    require(np.array_equal(scale_logits(a['raw_logits'],r['temperature']),a['scaled_logits']), 'Temperature application differs')
    for k,v in evaluate_record(a).items(): require(r[k]==v,'Metric recomputation mismatch: '+k)
    require(a['corrupted_lengths'].shape==a['lengths'].shape and
            (a['corrupted_lengths']>0).all() and (a['corrupted_lengths']<=a['lengths']).all(),'Corrupted lengths invalid')
    return r,a


def hierarchical(values):
    """values[(family,severity,replicate)] -> scalar/vector. Equal family weight."""
    return np.mean([np.mean([np.mean([values[f,s,r] for r in REPLICATES],axis=0)
        for s in levels],axis=0) for f,levels in FAMILIES.items()],axis=0)


def aggregate(root,fixture_only=False):
    root=Path(root);clean={};observations={};rows=[];metric_rows=[];length_rows=[];audit_rows=[];reference=None
    for arm in ARMS:
        for seed in SEEDS:
            r,a=load_record(root/'clean'/f'{arm}_{seed}',reference)
            reference=a if reference is None else reference
            require((r['arm'],r['seed'])==(arm,seed),'Wrong clean identity')
            clean[arm,seed]=(r,a)
            for v in VARIANTS:
                metric_rows.append(dict(arm=arm,seed=seed,family='clean',severity=0,replicate=0,variant=v,**flatten(r[v])))
    for f,s,rep in conditions():
        name=condition_id(f,s,rep)
        audit=read_json(root/'conditions'/name/'audit.json')
        require((audit['family'],audit['severity'],audit['replicate'])==(f,s,rep),'Wrong condition audit')
        audit_rows.append({k:audit[k] for k in ('family','severity','replicate','eligible','selected','realised_rate','applications_without_eligible')})
        require(audit['plan_sha256']==sha(root/'conditions'/name/'plan.npz'),'Corruption plan changed')
        for arm in ARMS:
            for seed in SEEDS:
                r,a=load_record(root/'conditions'/name/f'{arm}_{seed}',reference)
                cr,ca=clean[arm,seed]
                require((r['arm'],r['seed'],r['family'],r['severity'],r['replicate'])==(arm,seed,f,s,rep),'Wrong result identity')
                require(r['temperature']==cr['temperature'] and r['checkpoint_sha256']==cr['checkpoint_sha256'],'Frozen predictor binding changed')
                require(a['corrupted_lengths'].tolist()==audit['corrupted_offer_lengths'],'Length/audit mismatch')
                require(r['corruption_logical_sha256']==audit['output_logical_sha256'],'Input/audit mismatch')
                rows.append(r)
                for v in VARIANTS:
                    delta=losses(ca['y'],ca[v+'_logits'])-losses(a['y'],a[v+'_logits'])
                    observations[arm,seed,v,f,s,rep]=delta
                    metric_rows.append(dict(arm=arm,seed=seed,family=f,severity=s,replicate=rep,variant=v,
                        delta_rob_nll=float(delta.mean()),calibration_deterioration=r[v]['ece']-cr[v]['ece'],**flatten(r[v])))
                    length_rows.extend(dict(arm=arm,seed=seed,family=f,severity=s,replicate=rep,variant=v,
                        original_length_stratum=k,**flatten(m)) for k,m in r[v+'_length'].items())
    apps,inv,draws=cluster_draws(reference['application_id'],2000,70102)
    save_npz(root/'bootstrap_draws.npz',application_id=apps,indices=draws)
    macro={};comparisons={};overall={}
    for arm in ARMS:
        overall[arm]={}
        for v in VARIANTS:
            arr=np.stack([hierarchical({(f,s,r):observations[arm,seed,v,f,s,r] for f,s,r in conditions()}) for seed in SEEDS])
            macro[arm,v]=arr
            overall[arm][v]=cluster_interval(arr,inv,draws)
    for v in VARIANTS:
        for arm in ARMS[1:]:
            comparisons[v+':'+arm+'-T00']=cluster_interval(macro[arm,v]-macro['T00',v],inv,draws)
        for f,levels in FAMILIES.items():
            for s in levels:
                d=np.stack([np.mean([observations['T11',seed,v,f,s,r]-observations['T00',seed,v,f,s,r]
                    for r in REPLICATES],axis=0) for seed in SEEDS])
                comparisons[f'{v}:T11-T00:{f}:{s}']=cluster_interval(d,inv,draws)
    result=dict(version='ROB01-v0.1',status='REPORT_PENDING',fixture_only=fixture_only,
        scope='Synthetic fixture only' if fixture_only else 'Post-clean-review controlled stress test on existing held-out applications',
        n=len(reference['y']),applications=len(apps),corrupted_evaluations=len(rows),
        primary='raw:T11-T00',primary_sign='Positive favours T11: less NLL deterioration',
        delta_rob_sign='Clean NLL minus corrupt NLL; negative means deterioration',
        cd_sign='Corrupt ECE minus clean ECE; positive means worse calibration',
        aggregation='Average replicates, then severities within family, then equal-weight families; within seed first',
        inference_scope='Descriptive intervals conditional on frozen predictors AND these corruption realisations; no training-seed or corruption-population inference',
        bootstrap=dict(draws=2000,seed=70102,generator='PCG64',file_sha256=sha(root/'bootstrap_draws.npz')),
        mean_oriented_changes=overall,comparisons=comparisons,
        conditions=[dict(family=f,severity=s,replicate=r) for f,s,r in conditions()],
        metric_rows=metric_rows,length_rows=length_rows,corruption_counts=audit_rows,
        fitting=False,length_strata='Original clean lengths',training_seeds=list(SEEDS),corruption_replicates=list(REPLICATES))
    csv_write(root/'corruption_counts.csv',audit_rows)
    csv_write(root/'metrics.csv',metric_rows);csv_write(root/'length_metrics.csv',length_rows)
    csv_write(root/'comparisons.csv',[dict(contrast=k,mean=v['mean_delta'],lower=v['percentile_95'][0],upper=v['percentile_95'][1],
        **{f'seed_{s}':x for s,x in zip(SEEDS,v['by_seed'])}) for k,v in comparisons.items()])
    write_json(root/'rob01_summary.json',result)
    return result
