"""Rebuild all figures from saved results; never imports a dataset or model."""
from pathlib import Path
import csv,html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from .exp05_metrics import probabilities


def plots(root):
    root=Path(root);s=json.loads((root/'exp05_pilot_summary.json').read_text());out=root/'plots';out.mkdir(exist_ok=True)
    runs=s['runs'];arms=list(s['summary']);seeds=sorted({r['seed'] for r in runs});panels=[]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name,title):
        label='SYNTHETIC FIXTURE ONLY' if s.get('fixture_only') else 'validation development estimates'
        fig.suptitle(title+'\nEXP-05 v0.1 | '+label+' | '+str(seeds),fontsize=11)
        fig.tight_layout(rect=(0,0,1,.93))
        for ext in ('png','pdf','svg'):fig.savefig(out/(name+'.'+ext),dpi=150)
        plt.close(fig);panels.append((title,(out/(name+'.svg')).read_text()))
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,arm in zip(axes.flat,arms):
        for seed in seeds:
            p=root/f'{arm}_{seed}'/'history.json'
            if not p.exists():continue
            h=json.loads(p.read_text());line=ax.plot([r['epoch'] for r in h],[r['validation_nll'] for r in h],label=f'val {seed}')[0]
            ax.plot([r['epoch'] for r in h],[r['train_nll'] for r in h],ls='--',color=line.get_color(),alpha=.6)
            run=next(r for r in runs if r['arm']==arm and r['seed']==seed);ep=run['selected_epoch'];ax.scatter([ep],[h[ep-1]['validation_nll']],color=line.get_color(),s=20)
        ax.set(title=arm,xlabel='Epoch',ylabel='NLL (lower better)')
        if arm!='GBDT':ax.legend(fontsize=7)
        else:ax.text(.1,.5,'Fixed 200 boosting iterations\nNo epoch selection',transform=ax.transAxes)
    save(fig,'learning','Solid: validation; dashed: eval-mode train; dots: selected epochs')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i,arm in enumerate(arms):
        vals=s['summary'][arm]['nll_by_seed'];axes[0].scatter([i]*len(vals),vals);axes[0].plot([i-.15,i+.15],[np.mean(vals)]*2,c='black')
        if arm!='T00':
            d=s['paired_nll'][arm];lo,hi=d['percentile_95'];axes[1].plot([lo,hi],[i,i],c='grey');axes[1].scatter(d['mean_delta'],i)
    axes[0].axhline(s['dummy']['nll'],ls=':',color='black',label='Train-prevalence dummy');axes[0].legend();axes[0].set(xticks=range(len(arms)),xticklabels=arms,ylabel='NLL (lower better)')
    axes[1].axvline(0,c='black',lw=.7);axes[1].set(yticks=range(len(arms)),yticklabels=arms,xlabel='NLL minus T00; negative favours comparator')
    save(fig,'paired','All seeds and descriptive application-cluster intervals')
    fig,ax=plt.subplots(figsize=(6,4));mat=np.array([[s['summary'][k]['mean'] for k in ('T00','T01')],[s['summary'][k]['mean'] for k in ('T10','T11')]])
    im=ax.imshow(mat,cmap='Blues');fig.colorbar(im,ax=ax,label='Mean NLL');ax.set(xticks=[0,1],xticklabels=['Standard attention','Retention'],yticks=[0,1],yticklabels=['Identity','Adaptive'])
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{mat[i,j]:.4f}',ha='center',va='center',color='black')
    save(fig,'factorial','2 x 2 comparison; interaction per seed: '+', '.join(f'{v:.4f}' for v in s['interaction_by_seed']))
    first=seeds[0];pred={}
    for arm in arms:
        with np.load(root/f'{arm}_{first}'/'predictions.npz',allow_pickle=False) as f:pred[arm]={k:f[k] for k in f.files}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for c,ax in enumerate(axes):
        for arm in arms:
            a=pred[arm];p=probabilities(a['logits']);precision,recall,_=precision_recall_curve(a['y']==c,p if c else 1-p);ax.plot(recall,precision,label=arm)
        ax.axhline(float(np.mean(a['y']==c)),ls=':',c='black');ax.set(title=f'Class {c}',xlabel='Recall',ylabel='Precision',ylim=(0,1.02));ax.legend(fontsize=7)
    save(fig,'precision_recall',f'Precision-recall curves; prespecified display seed {first}; dotted: prevalence')
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for ax,arm in zip(axes.flat,arms):
        r=next(r for r in runs if r['arm']==arm and r['seed']==first);cm=np.array(r['selected']['confusion_matrix']);ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        ax.set(title=arm,xlabel='Predicted',ylabel='True',xticks=[0,1],yticks=[0,1])
    save(fig,'confusion',f'Confusion matrices, threshold 0.5; display seed {first}')
    fig,axes=plt.subplots(1,3,figsize=(13,4));groups=['1-10','11-20','21-40','41+']
    for arm in arms:
        rr=[r for r in runs if r['arm']==arm]
        axes[0].plot(groups,[np.mean([r['length_strata'][g]['nll'] for r in rr]) if rr[0]['length_strata'][g]['n'] else np.nan for g in groups],marker='o',label=arm)
    a=pred['T00'];lengths=a['lengths'];edges=np.arange(1,max(lengths)+3)-.5
    for c in (0,1):axes[1].hist(lengths[a['y']==c],bins=edges,alpha=.5,label=f'class {c}')
    order=np.sort(lengths);axes[2].step(order,np.arange(1,len(order)+1)/len(order));axes[2].set(xlabel='Prefix length',ylabel='Cumulative offer fraction')
    axes[0].set(ylabel='Mean seed NLL',xlabel='Length stratum');axes[0].legend(fontsize=7);axes[1].set(xlabel='Prefix length',ylabel='Offer count');axes[1].legend()
    save(fig,'length','Length diagnostics: complete tail and class composition retained')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for c,ax in enumerate(axes):
        for arm in arms:
            rr=[r for r in runs if r['arm']==arm];values=[]
            for g in groups:
                vals=[r['length_strata'][g].get('classes',{}).get(str(c),{}).get('recall') for r in rr]
                values.append(np.mean(vals) if all(v is not None for v in vals) else np.nan)
            ax.plot(groups,values,marker='o',label=arm)
        ax.set(title=f'Class {c}',xlabel='Length stratum',ylabel='Mean seed recall',ylim=(0,1.02));ax.legend(fontsize=7)
    save(fig,'length_recall','Recall by length and class; missing points mean no class support')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in arms:
        r=next(r for r in runs if r['arm']==arm and r['seed']==first);bins=[b for b in r['selected']['reliability'] if b['count']]
        axes[0].plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],marker='o',label=arm)
        axes[1].plot([b['confidence'] for b in bins],[b['count'] for b in bins],marker='o',label=arm)
    axes[0].plot([.5,1],[.5,1],ls=':',c='black');axes[0].set(xlabel='Top-label confidence',ylabel='Bin accuracy');axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Top-label confidence',ylabel='Bin count')
    save(fig,'reliability',f'Uncalibrated reliability, 15 fixed bins; display seed {first}')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,arm in enumerate(arms):
        rr=[r for r in runs if r['arm']==arm];axes[0].scatter([i]*len(rr),[r['elapsed_seconds'] for r in rr]);axes[1].scatter([i]*len(rr),[r.get('peak_process_rss_bytes',float('nan'))/2**20 for r in rr])
    for ax in axes:ax.set(xticks=range(len(arms)),xticklabels=arms)
    axes[0].set_ylabel('Run elapsed seconds (including evaluation)');axes[1].set_ylabel('Process peak RSS MiB (sampled; not incremental model RAM)')
    save(fig,'resources','Resources: tree/GRU differ in architecture and workload')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in arms:
        for seed in seeds:
            p=root/f'{arm}_{seed}'/'history.json'
            if not p.exists():continue
            h=json.loads(p.read_text())
            for ax,key in zip(axes,['mean_identity_weight','mean_retention']):
                if h[0].get(key) is not None:ax.plot([r['epoch'] for r in h],[r[key] for r in h],label=f'{arm}/{seed}')
    for ax,label in zip(axes,['Mean identity mixture weight','Mean latent retention propensity']):
        ax.set(xlabel='Epoch',ylabel=label);ax.legend(fontsize=6)
    save(fig,'diagnostics','Training-prefix diagnostics only; not information percentages or causal importance')
    with (root/'metrics_by_run.csv').open('w',newline='',encoding='utf-8') as f:
        cols=['arm','seed','nll','balanced_accuracy','macro_f1','auroc','brier','ece','elapsed_seconds','parameter_count'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in runs:w.writerow({k:r.get(k,r['selected'].get(k)) for k in cols})
    details=html.escape(json.dumps({'scope':s['scope'],'summary':s['summary'],'interaction_by_seed':s['interaction_by_seed']},indent=2))
    body=''.join('<section><h2>'+html.escape(title)+'</h2>'+svg+'</section>' for title,svg in panels)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>EXP-05 pilot</title><style>body{font:16px system-ui;max-width:1250px;margin:2rem auto;padding:1rem;background:#fafafa}section{background:white;margin:2rem 0;padding:1rem}svg{width:100%;height:auto}pre{white-space:pre-wrap}</style><h1>EXP-05 controlled pilot</h1><p>'+html.escape(s['scope'])+'</p><details><summary>Numerical summary</summary><pre>'+details+'</pre></details>'+body,encoding='utf-8')
