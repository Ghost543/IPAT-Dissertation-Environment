"""Build and verify BPI17 tensors from the pinned contract and split manifest.

Writes a fresh output directory. No EXP-05 training or performance evaluation.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import hashlib
import html
import inspect
import json
from pathlib import Path
import platform
import sys

# Run directly from the dissertation project root; installed package also works.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from ipat.data.bpi17 import (NUM,CAT,SPLITS,load_source,fit_preprocessing,save_cache,
    PackedOffers,collate_numpy,logical_hash,sha256,canonical)
from ipat.model.bpi17 import config_from_state,model_stats,MODELS,BPI17FullIPAT
from ipat.model import baseline,adaptive,retention
from ipat.training import preprocessing


def tensor_batch(examples,padding=0):
    x,y,meta=collate_numpy(examples,extra_padding=padding)
    return {k:torch.from_numpy(v) for k,v in x.items()},torch.from_numpy(y)


def verify_models(state,examples):
    torch.set_num_threads(1)
    cfg=config_from_state(state,dropout=0.,asinh_scales=(),gamma=0.)
    stats=model_stats(state);batch,y=tensor_batch(examples)
    models=[];outs=[];grads=[]
    for cls in MODELS:
        torch.manual_seed(37);m=cls(cfg,stats);models.append(m)
        out=m(batch,return_diagnostics=True);outs.append(out)
        torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],y).backward()
        grads.append({k:p.grad.detach().clone() for k,p in m.named_parameters() if p.grad is not None})
    base=dict(models[0].named_parameters())
    shared=max(float((base[k]-dict(m.named_parameters())[k]).detach().abs().max()) for m in models[1:] for k in base)
    forward=max(float((outs[0]['logits']-o['logits']).detach().abs().max()) for o in outs[1:])
    attention=max(float((a-b).detach().abs().max()) for o in outs[1:] for a,b in zip(outs[0]['attention'],o['attention']))
    gradient=max(float((g-grads[i][k]).abs().max()) for i in range(1,4) for k,g in grads[0].items())
    cfg_normal=config_from_state(state,dropout=0.)
    padded,_=tensor_batch(examples,5);padding_error=0.
    for cls in MODELS:
        torch.manual_seed(37);m=cls(cfg_normal,stats).eval()
        with torch.no_grad():padding_error=max(padding_error,float((m(batch)['logits']-m(padded)['logits']).abs().max()))
    torch.manual_seed(37);m=BPI17FullIPAT(cfg_normal,stats)
    out=m(batch,return_diagnostics=True);w=m.projector.orthogonal_weight()
    gram=float((w.T@w-torch.eye(cfg_normal.d_in)).detach().abs().max())
    reconstructed=(out['z_pre_norm']-m.projector.bias)@w
    valid=batch['valid_mask'];reconstruction=float((reconstructed[valid]-out['u'][valid]).detach().abs().max())
    observed=valid.unsqueeze(-1)&~batch['missing_mask']
    deriv=float(out['components']['mixture_derivative'][observed].detach().min())
    expected=(batch['x_obs']-m.assembler.numeric.mean)/m.assembler.numeric.scale
    standardisation=float((out['components']['x_std'][observed]-expected[observed]).detach().abs().max())
    before=m.classifier.weight.detach().clone();opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],y);loss.backward()
    finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None)
    opt.step();changed=not torch.equal(before,m.classifier.weight)
    metrics=dict(shared_parameter_max_difference=shared,identity_gamma_zero_logit_max_difference=forward,
        identity_gamma_zero_attention_max_difference=attention,identity_gamma_zero_shared_gradient_max_difference=gradient,
        padding_logit_max_difference=padding_error,projection_gram_max_error=gram,
        projection_reconstruction_max_error=reconstruction,minimum_observed_derivative=deriv,
        standardisation_max_difference=standardisation)
    gates=dict(shared_initialisation=shared==0.,limiting_forward_parity=forward==0.,
        limiting_attention_parity=attention==0.,limiting_gradient_parity=gradient==0.,
        padding_invariance=padding_error<=1e-6,projection_gram=gram<=1e-6,
        projection_reconstruction=reconstruction<=1e-5,derivative_floor=deriv>=cfg_normal.identity_floor-1e-6,
        no_double_standardisation=standardisation==0.,finite_backward_and_parameter_update=finite and changed)
    return metrics,gates,asdict(cfg_normal)|{'d_in':cfg_normal.d_in,'d_model':cfg_normal.d_model}


def render(state,parts,examples,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    missing=np.array([parts[s]['numeric_missing_fraction'] for s in SPLITS])*100
    im=ax[0,0].imshow(missing,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax[0,0].set_xticks(range(6),NUM,rotation=35,ha='right',fontsize=8);ax[0,0].set_yticks(range(4),SPLITS)
    ax[0,0].set_title('Missing numerical cells in packed prefixes (%)')
    for i in range(4):
        for j in range(6):ax[0,0].text(j,i,f'{missing[i,j]:.1f}',ha='center',va='center',color='white' if missing[i,j]>60 else 'black')
    unknown=np.array([parts[s]['categorical_unknown_fraction'] for s in SPLITS])*100
    ax[0,1].imshow(unknown,cmap='Oranges',aspect='auto',vmin=0,vmax=max(float(unknown.max()),1.))
    ax[0,1].set_xticks(range(7),CAT,rotation=35,ha='right',fontsize=8);ax[0,1].set_yticks(range(4),SPLITS)
    ax[0,1].set_title('Categories unseen during fitting (%)')
    for i in range(4):
        for j in range(7):ax[0,1].text(j,i,f'{unknown[i,j]:.2f}',ha='center',va='center',fontsize=8)
    # Short examples make field missingness and event padding separately visible.
    x,_,_=collate_numpy(examples[:2],extra_padding=2)
    grid=np.where(x['valid_mask'][...,None],np.where(x['missing_mask'],1,0),2)
    from matplotlib.colors import ListedColormap
    ax[1,0].imshow(grid[0].T,cmap=ListedColormap(['#3e9b78','#e5ad42','#c8cdd2']),vmin=0,vmax=2,aspect='auto')
    ax[1,0].set_yticks(range(6),NUM,fontsize=8);ax[1,0].set_xlabel('Event slot (zero-based)')
    ax[1,0].set_title('One prefix: green observed / amber missing / grey padding')
    vals=[n['count'] for n in state['numeric']]
    ax[1,1].bar(range(6),vals,color='#3979ad');ax[1,1].set_xticks(range(6),NUM,rotation=35,ha='right',fontsize=8)
    ax[1,1].set_ylabel('Observed values used to fit statistics');ax[1,1].set_title('Training fitting counts; overlapping prefixes deduplicated')
    for i,v in enumerate(vals):ax[1,1].text(i,v,f'{v:,}',ha='center',va='bottom',fontsize=8)
    ax[1,1].set_ylim(0,max(vals)*1.12)
    fig.suptitle('BPI17 schema adapter — measured encoding diagnostics; no model performance',fontsize=13)
    for ext in ('png','svg','pdf'):fig.savefig(directory/f'adapter_diagnostics.{ext}',dpi=160)
    plt.close(fig)
    svg=(directory/'adapter_diagnostics.svg').read_text();svg=svg[svg.index('<svg'):]
    (directory/'adapter_diagnostics.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>BPI17 adapter diagnostics</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px}svg{width:100%;height:auto}pre{white-space:pre-wrap}</style><h1>BPI17 schema adapter diagnostics</h1><p>All figures use measured adapter outputs. Missingness is measured over packed prefix event occurrences; preprocessing counts use unique training source events (static attributes once per application).</p>'+svg+'<p>PAD=0, MISSING=1, UNK=2. Unknown categories are a lossy encoding; future raw category identity is not guaranteed. No model performance is evaluated.</p></html>',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--contract',type=Path,default=Path(__file__).resolve().parents[1]/'configs/bpi17_contract_v0.1.json')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--plots',action='store_true')
    a=p.parse_args()
    if a.output_dir.exists():raise SystemExit('Output directory already exists; use a fresh path.')
    print('Checking source and manifest, then extracting admitted prefixes...',flush=True)
    apps=load_source(a.source,a.manifest,a.contract)
    print('Fitting training-only preprocessing...',flush=True);state=fit_preprocessing(apps)
    a.output_dir.mkdir(parents=True)
    parts=save_cache(apps,state,a.output_dir/'cache')
    checks={}
    expected={'train':19071,'validation':7871,'calibration':6117,'test':8613}
    sets={s:{x.application_id for x in apps if x.split==s} for s in SPLITS}
    checks['application_disjoint']=all(not sets[s]&sets[t] for i,s in enumerate(SPLITS) for t in SPLITS[i+1:])
    checks['offer_counts_match_contract']=all(parts[s]['offers']==expected[s] for s in SPLITS)
    checks['no_unknown_training_categories']=all(x==0 for x in parts['train']['categorical_unknown_fraction'])
    checks['all_training_scales_positive']=all(n['scale']>0 for n in state['numeric'])
    print('Checking cache roundtrips and frozen model interfaces...',flush=True)
    ds=PackedOffers.load(a.output_dir/'cache/train.npz')
    checks['all_cache_roundtrips']=all(logical_hash(PackedOffers.load(a.output_dir/f'cache/{s}.npz').arrays)==parts[s]['logical_sha256'] for s in SPLITS)
    with a.manifest.open(newline='',encoding='utf-8') as f: manifest_rows={r['offer_id']:r for r in csv.DictReader(f)}
    provenance=True
    for split in SPLITS:
        cache=PackedOffers.load(a.output_dir/f'cache/{split}.npz').arrays
        for i,oid in enumerate(cache['offer_id']):
            rr=manifest_rows[str(oid)]
            provenance &= (rr['split']==split and rr['application_id']==str(cache['application_id'][i]) and int(rr['label'])==int(cache['y'][i]) and int(rr['prefix_length'])==int(cache['offsets'][i+1]-cache['offsets'][i]))
    checks['every_cached_offer_matches_manifest']=bool(provenance)
    lengths=np.diff(ds.arrays['offsets']);indices=list(dict.fromkeys([0,1,int(lengths.argmax()),int(lengths.argmin())]))
    examples=[ds[i] for i in indices]
    metrics,model_checks,cfg=verify_models(state,examples);checks.update(model_checks)
    # A second deterministic encoding of every admitted split checks logical cache identity.
    from ipat.data.bpi17 import pack_split
    checks['deterministic_encoding']=all(logical_hash(pack_split(apps,state,s))==parts[s]['logical_sha256'] for s in SPLITS)
    sources={m.__name__:sha256(inspect.getfile(m)) for m in (baseline,adaptive,retention,preprocessing)}
    report=dict(version='IPAT-BPI17-adapter-v0.1',status='PASS' if all(checks.values()) else 'REVIEW_REQUIRED',
        scope='Adapter and integration acceptance only; no EXP-05 predictive results.',
        source_sha256=sha256(a.source),manifest_sha256=sha256(a.manifest),contract_sha256=sha256(a.contract),
        preprocessing_sha256=sha256(a.output_dir/'cache/preprocessing.json'),
        runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),device='cpu'),
        frozen_module_sha256=sources,adapter_module_sha256={name:sha256(inspect.getfile(sys.modules[name])) for name in ("ipat.data.bpi17","ipat.model.bpi17")},builder_sha256=sha256(Path(__file__)),checks=checks,model_metrics=metrics,validation_config=cfg,partitions=parts,
        training_unique_events=state['train_unique_events'],training_applications=state['train_applications'],
        limitations=['Local fixture-test output is separate evidence.','Passing integration checks does not verify historical field availability or source licensing/version.','One optimiser step is an engineering diagnostic, not EXP-05.','Projection and parity checks cover the recorded runtime/configuration.'])
    (a.output_dir/'adapter_acceptance.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if a.plots:render(state,parts,[ds[0],ds[1]],a.output_dir)
    print(json.dumps({k:report[k] for k in ('status','scope','checks','model_metrics')},indent=2))
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
