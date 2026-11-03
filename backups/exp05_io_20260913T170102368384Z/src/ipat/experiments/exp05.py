"""Accepted EXP-05 pilot: training/validation only, paired budgets and recoverable state."""
from pathlib import Path
import hashlib, importlib, json, os, platform, random, time
import numpy as np
import torch
from torch.nn import functional as F
from ipat.data.bpi17 import PackedOffers,collate_bpi17,logical_hash,canonical,sha256
from ipat.model.bpi17 import MODELS,config_from_state,model_stats
from .exp05_references import PilotGRU,prefix_summaries,tree
from .exp05_metrics import metrics,stratified,paired_cluster_interval

ARMS=('T00','T10','T01','T11','GRU')


def digest(obj):return hashlib.sha256(canonical(obj).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)


def save_checkpoint(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)


def load_checkpoint(path):
    # Checkpoints here contain tensors and built-in types, never arbitrary model objects.
    return torch.load(path,map_location='cpu',weights_only=True)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])


def permutation(n,seed,epoch):return np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(n)


def load_split(cache,split,expected):
    if split not in ('train','validation'):raise ValueError('EXP-05 prohibits calibration/test predictive loading')
    d=PackedOffers.load(Path(cache)/(split+'.npz'))
    if logical_hash(d.arrays)!=expected:raise ValueError('Frozen logical cache hash mismatch: '+split)
    return d


def load_inputs(cache,config,validation=True):
    state=json.loads((Path(cache)/'preprocessing.json').read_text(encoding='utf-8-sig'))
    if digest(state)!=config['preprocessing_logical_sha256']:raise ValueError('Preprocessing content changed')
    train=load_split(cache,'train',config['logical_hashes']['train'])
    val=load_split(cache,'validation',config['logical_hashes']['validation']) if validation else None
    if val is not None and set(train.arrays['application_id'])&set(val.arrays['application_id']):raise ValueError('Application leakage')
    return state,train,val


def make_model(arm,state,seed,settings):
    seed_all(seed);cfg=config_from_state(state,**settings);stats=model_stats(state)
    return (MODELS[ARMS.index(arm)] if arm!='GRU' else PilotGRU)(cfg,stats)


def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def assert_shared(state,seed,settings):
    base=make_model('T00',state,seed,settings).state_dict();base_params=dict(make_model('T00',state,seed,settings).named_parameters())
    names=sorted(base_params);hashes={}
    for arm in ARMS[:4]:
        model=make_model(arm,state,seed,settings);other=model.state_dict()
        for k in names:
            if k not in other or not torch.equal(base[k],other[k]):raise AssertionError('Shared initialisation mismatch: '+arm+' '+k)
        hashes[arm]=tensor_hash({k:other[k] for k in names})
    return hashes


def batch(dataset,indices,device):
    x,y,_=collate_bpi17([dataset[int(i)] for i in indices]);return {k:v.to(device) for k,v in x.items()},y.to(device)


def predict(model,dataset,device,microbatch=64,diagnostics=False):
    was=model.training;rng=rng_state();model.eval();out=[];ident=[];ret=[]
    try:
        with torch.no_grad():
            for start in range(0,len(dataset),microbatch):
                x,_=batch(dataset,range(start,min(start+microbatch,len(dataset))),device)
                r=model(x,return_diagnostics=diagnostics) if not isinstance(model,PilotGRU) else model(x);out.extend(r['logits'].cpu().double().tolist())
                if diagnostics and 'components' in r and 'mixture_weights' in r['components']:
                    ident.append(float(r['components']['mixture_weights'][...,0].mean().cpu()))
                # Retention output names are inspected by the caller's frozen model contract.
                if diagnostics and 'retention' in r:
                    v=r['retention']['r'];ret.extend(v[x['valid_mask']].detach().cpu().reshape(-1).tolist())
    finally:model.train(was);restore_rng(rng)
    values=np.asarray(out)
    if not np.isfinite(values).all():raise ValueError('Non-finite prediction')
    return values,dict(mean_identity_weight=float(np.mean(ident)) if ident else None,
                       mean_retention=float(np.mean(ret)) if ret else None)


def optimiser(model,c):
    return torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=c['weight_decay'])


def train_run(arm,seed,state,train,val,c,directory,identity,device='cpu',resume=False,stop_after=None):
    """stop_after is a fixture-only interruption hook, never exposed by the pilot CLI."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    model=make_model(arm,state,seed,c['model']).to(device);initial=tensor_hash(model.state_dict());opt=optimiser(model,c)
    start_epoch=0;offset=0;step=0;history=[];best=float('inf');best_epoch=None;partial={'loss_sum':0.,'n':0,'norm_sum':0.,'updates':0,'clipped':0};elapsed=0.
    checkpoint=directory/'resume.pt';seed_all(seed+9000)
    if resume and checkpoint.exists():
        s=load_checkpoint(checkpoint)
        if s['identity']!=identity:raise ValueError('Resume configuration/code/data/runtime differs')
        model.load_state_dict(s['model']);opt.load_state_dict(s['optimiser']);restore_rng(s['rng'])
        start_epoch=s['epoch'];offset=s['offset'];step=s['step'];history=s['history'];best=s['best'];best_epoch=s['best_epoch'];partial=s['partial'];elapsed=s['elapsed']
        if offset and s['permutation']!=permutation(len(train),seed,start_epoch).tolist():raise ValueError('Permutation mismatch')
    elif checkpoint.exists():raise FileExistsError('Use --resume for an existing run')
    started=time.perf_counter()
    def persist(epoch,offset,order):
        save_checkpoint(checkpoint,dict(identity=identity,model=model.state_dict(),optimiser=opt.state_dict(),rng=rng_state(),
            epoch=epoch,offset=offset,step=step,history=history,best=best,best_epoch=best_epoch,partial=partial,
            permutation=order.tolist(),elapsed=elapsed+time.perf_counter()-started))
    for epoch in range(start_epoch,c['epochs']):
        model.train();order=permutation(len(train),seed,epoch)
        for start in range(offset,len(train),c['batch_size']):
            ii=order[start:start+c['batch_size']];opt.zero_grad(set_to_none=True);total=0.
            for j in range(0,len(ii),c['microbatch']):
                x,y=batch(train,ii[j:j+c['microbatch']],device);r=model(x);loss=F.binary_cross_entropy_with_logits(r['logits'],y,reduction='sum')
                if not torch.isfinite(loss):raise ValueError('Non-finite loss')
                (loss/len(ii)).backward();total+=float(loss.detach())
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm'],error_if_nonfinite=True))
            opt.step();step+=1;partial['loss_sum']+=total;partial['n']+=len(ii);partial['norm_sum']+=norm;partial['updates']+=1;partial['clipped']+=int(norm>c['clip_norm'])
            end=start+len(ii)
            if step%c['checkpoint_every']==0 or stop_after==step:persist(epoch,end,order)
            if stop_after==step:return {'status':'INTERRUPTED_FOR_FIXTURE','step':step}
        tr,diag=predict(model,train,device,c['microbatch'],True);vl,_=predict(model,val,device,c['microbatch'])
        vm=metrics(val.arrays['y'],vl,val.arrays['application_id']);tm=metrics(train.arrays['y'],tr)
        entry=dict(epoch=epoch+1,step=step,train_nll=tm['nll'],validation_nll=vm['nll'],optimisation_train_nll=partial['loss_sum']/partial['n'],
                   mean_gradient_norm=partial['norm_sum']/partial['updates'],clipped_fraction=partial['clipped']/partial['updates'],**diag)
        history.append(entry)
        if vm['nll']<best:
            best=vm['nll'];best_epoch=epoch+1;save_checkpoint(directory/'selected.pt',{'model':model.state_dict(),'epoch':best_epoch,'identity':identity})
        write_json(directory/'history.json',history)
        print(f'{arm} seed={seed} epoch={epoch+1:02d} train_nll={tm["nll"]:.6f} val_nll={vm["nll"]:.6f}',flush=True)
        offset=0;partial={'loss_sum':0.,'n':0,'norm_sum':0.,'updates':0,'clipped':0};persist(epoch+1,0,order)
    # On a completed resume, these are generated again from the checkpoint, without more fitting.
    save_checkpoint(directory/'final.pt',{'model':model.state_dict(),'epoch':c['epochs'],'identity':identity})
    final,_=predict(model,val,device,c['microbatch']);s=load_checkpoint(directory/'selected.pt');model.load_state_dict(s['model'])
    selected,_=predict(model,val,device,c['microbatch']);again,_=predict(model,val,device,c['microbatch'])
    if not np.array_equal(selected,again):raise AssertionError('Checkpoint prediction replay differs')
    np.savez_compressed(directory/'predictions.npz',logits=selected,final_logits=final,y=val.arrays['y'],
                        offer_id=val.arrays['offer_id'],application_id=val.arrays['application_id'],lengths=np.diff(val.arrays['offsets']))
    result=dict(arm=arm,seed=seed,status='PASS',epochs=c['epochs'],updates=step,selected_epoch=best_epoch,initial_state_sha256=initial,
        parameter_count=sum(p.numel() for p in model.parameters()),elapsed_seconds=elapsed+time.perf_counter()-started,
        selected=metrics(val.arrays['y'],selected,val.arrays['application_id']),final=metrics(val.arrays['y'],final,val.arrays['application_id']),
        length_strata=stratified(val.arrays['y'],selected,val.arrays['application_id'],np.diff(val.arrays['offsets'])),
        checkpoint_replay_exact=True)
    write_json(directory/'result.json',result);return result


def environment(device):
    import sklearn
    modules=['ipat.model.baseline','ipat.model.adaptive','ipat.model.retention','ipat.model.bpi17','ipat.data.bpi17',
             'ipat.experiments.exp05','ipat.experiments.exp05_metrics','ipat.experiments.exp05_references','ipat.experiments.exp05_plots']
    return dict(python=platform.python_version(),platform=platform.platform(),torch=str(torch.__version__),numpy=np.__version__,sklearn=sklearn.__version__,device=str(device),
        device_name=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else platform.processor(),
        cpu_threads=torch.get_num_threads(),code_sha256={k:sha256(importlib.import_module(k).__file__) for k in modules})


def configure_device(device):
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if str(device).startswith('cuda') and not torch.cuda.is_available():raise ValueError('Requested CUDA unavailable')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    return device


def provenance_review(path,config):
    """Read a completed, evidence-backed source review. No default verified flag."""
    p=Path(path);r=json.loads(p.read_text())
    if r.get('status')!='VERIFIED' or r.get('source_sha256')!=config['source_sha256']:raise ValueError('DATA-02 source review incomplete')
    for k in ('official_url','repository_version','terms_url','reviewer','review_date','permitted_use_basis'):
        if not isinstance(r.get(k),str) or not r[k].strip():raise ValueError('Source review missing '+k)
    for k in ('repository_evidence','terms_evidence'):
        entry=r[k];f=p.parent/entry['path']
        if sha256(f)!=entry['sha256']:raise ValueError('Source review evidence hash mismatch')
    return {'receipt_sha256':sha256(p),'record':r}


def aggregate(root,seeds,draws=2000):
    root=Path(root);runs=[];arrays={}
    for arm in (*ARMS,'GBDT'):
        for seed in seeds:
            d=root/f'{arm}_{seed}';runs.append(json.loads((d/'result.json').read_text()))
            with np.load(d/'predictions.npz',allow_pickle=False) as f:arrays[arm,seed]={k:f[k] for k in f.files}
    d=json.loads((root/'dummy.json').read_text());ref=arrays['T00',seeds[0]]
    for a in arrays.values():
        for k in ('y','offer_id','application_id','lengths'):
            if not np.array_equal(ref[k],a[k]):raise AssertionError('Prediction alignment failed')
    if len(set(ref['offer_id']))!=len(ref['offer_id']):raise AssertionError('Duplicate validation offers')
    summary={}
    for arm in (*ARMS,'GBDT'):
        scores=[r['selected']['nll'] for r in runs if r['arm']==arm]
        summary[arm]=dict(nll_by_seed=scores,mean=float(np.mean(scores)),sample_std=float(np.std(scores,ddof=1)),
            identical_seed_predictions=all(np.array_equal(arrays[arm,seeds[0]]['logits'],arrays[arm,s]['logits']) for s in seeds[1:]))
    comparisons={}
    for arm in ('T10','T01','T11','GRU','GBDT'):
        comparisons[arm]=paired_cluster_interval(ref['y'],[arrays[arm,s]['logits'] for s in seeds],[arrays['T00',s]['logits'] for s in seeds],ref['application_id'],draws)
    interaction=[summary['T11']['nll_by_seed'][i]-summary['T10']['nll_by_seed'][i]-summary['T01']['nll_by_seed'][i]+summary['T00']['nll_by_seed'][i] for i in range(len(seeds))]
    result=dict(status='PASS',scope='EXP-05 validation-only pilot; checkpoint selection uses validation; no final-test claim',
        runs=runs,dummy=d,summary=summary,paired_nll=comparisons,interaction_by_seed=interaction,
        checkpoints_selected_on_validation=True,predictive_partitions=['train','validation'],
        all_runs_improve_dummy={r['arm']+'_'+str(r['seed']):r['selected']['nll']<d['nll'] for r in runs})
    write_json(root/'exp05_pilot_summary.json',result);return result


class MemoryMonitor:
    """Sample process RSS; GPU allocation peaks reported separately."""
    def __enter__(self):
        import psutil,threading
        self.process=psutil.Process();self.peak=self.process.memory_info().rss;self.stop=threading.Event()
        def watch():
            while not self.stop.wait(.05):self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.peak=max(self.peak,self.process.memory_info().rss)
    def report(self):return dict(peak_process_rss_bytes=self.peak,rss_sampling_seconds=.05,
                                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)


def preflight(state,train,c,root,device):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);results=[]
    longest=int(np.argmax(np.diff(train.arrays['offsets'])))
    for arm in ARMS:
        m=make_model(arm,state,999,c['model']).to(device);opt=optimiser(m,c);seed_all(999+9000)
        started=time.perf_counter()
        with MemoryMonitor() as mon:
            for step in range(50):
                ii=permutation(len(train),999,step)[:c['batch_size']]
                if step==0:ii[0]=longest
                opt.zero_grad(set_to_none=True)
                for j in range(0,len(ii),c['microbatch']):
                    x,y=batch(train,ii[j:j+c['microbatch']],device);loss=F.binary_cross_entropy_with_logits(m(x)['logits'],y,reduction='sum')/len(ii)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite preflight loss')
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),c['clip_norm'],error_if_nonfinite=True);opt.step()
            if str(device).startswith('cuda'):torch.cuda.synchronize()
        results.append(dict(arm=arm,updates=50,elapsed_seconds=time.perf_counter()-started,**mon.report()))
        print('Preflight completed: '+arm,flush=True)
    receipt=dict(status='PASS',scope='Train-only resource diagnostic; disposable seed/weights; not EXP-05 results',
        configuration=c,environment=environment(device),longest_training_prefix=int(np.diff(train.arrays['offsets']).max()),
        runs=results,shared_initialisation={str(s):assert_shared(state,s,c['model']) for s in c['seeds']})
    write_json(root/'exp05_preflight.json',receipt);return receipt


def pilot(cache,c,root,device,review_path,preflight_path,resume=False):
    review=provenance_review(review_path,c);pf=json.loads(Path(preflight_path).read_text())
    if pf.get('status')!='PASS' or pf['configuration']!=c or pf['environment']!=environment(device):raise ValueError('Run matching resource preflight first')
    state,train,val=load_inputs(cache,c);env=environment(device)
    manifest=dict(config=c,environment=env,source_review=review,protocol_status='Accepted by user 2026-09-13',
        configuration_sha256=digest(c),predictive_partitions=['train','validation'])
    ident=digest(manifest);root=Path(root)
    if root.exists():
        if not resume or json.loads((root/'run_manifest.json').read_text())!=manifest:raise ValueError('Existing output: use matching --resume')
    else:root.mkdir(parents=True);write_json(root/'run_manifest.json',manifest)
    q=float(train.arrays['y'].mean());dummy_logit=float(np.log(q)-np.log1p(-q))
    write_json(root/'dummy.json',metrics(val.arrays['y'],np.full(len(val),dummy_logit),val.arrays['application_id']))
    write_json(root/'initialisation.json',{str(s):assert_shared(state,s,c['model']) for s in c['seeds']})
    for seed in c['seeds']:
        for arm in ARMS:
            d=root/f'{arm}_{seed}'
            with MemoryMonitor() as mon:
                result=train_run(arm,seed,state,train,val,c,d,ident,device,resume)
            result.update(mon.report());write_json(d/'result.json',result)
    x=prefix_summaries(train,state);v=prefix_summaries(val,state)
    import joblib
    for seed in c['seeds']:
        d=root/f'GBDT_{seed}';d.mkdir(exist_ok=True);started=time.perf_counter()
        with MemoryMonitor() as mon:
            model=tree(seed);model.fit(x,train.arrays['y']);logits=model.decision_function(v)
            joblib.dump(model,d/'selected.joblib');again=joblib.load(d/'selected.joblib').decision_function(v)
        if not np.array_equal(logits,again):raise AssertionError('Tree checkpoint replay failed')
        np.savez_compressed(d/'predictions.npz',logits=logits,final_logits=logits,y=val.arrays['y'],offer_id=val.arrays['offer_id'],
                            application_id=val.arrays['application_id'],lengths=np.diff(val.arrays['offsets']))
        result=dict(arm='GBDT',seed=seed,status='PASS',iterations=int(model.n_iter_),elapsed_seconds=time.perf_counter()-started,parameter_count=None,
            selected=metrics(val.arrays['y'],logits,val.arrays['application_id']),final=metrics(val.arrays['y'],logits,val.arrays['application_id']),
            length_strata=stratified(val.arrays['y'],logits,val.arrays['application_id'],np.diff(val.arrays['offsets'])),checkpoint_replay_exact=True,**mon.report())
        write_json(d/'result.json',result)
    result=aggregate(root,c['seeds'],c['bootstrap_draws'])
    from .exp05_plots import plots
    plots(root)
    checks=dict(all_planned_runs=len(result['runs'])==18,all_replay=all(r['checkpoint_replay_exact'] for r in result['runs']),
        exact_neural_budget=all(r['updates']==c['epochs']*((len(train)+c['batch_size']-1)//c['batch_size']) for r in result['runs'] if r['arm']!='GBDT'),
        exact_tree_budget=all(r['iterations']==200 for r in result['runs'] if r['arm']=='GBDT'),
        prediction_alignment=True,heldout_predictive_access_disabled=True,source_review_recorded=True)
    if not all(checks.values()):raise AssertionError(checks)
    write_json(root/'exp05_acceptance.json',dict(status='PASS',checks=checks,scope='Execution validity, not hypothesis acceptance',
        summary_sha256=sha256(root/'exp05_pilot_summary.json'),config_sha256=digest(c)))
    return result
