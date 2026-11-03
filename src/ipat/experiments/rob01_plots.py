"""Offline report from recomputed ROB-01 results; no data/model fitting."""
import base64
import html
from pathlib import Path
import numpy as np
from .exp06_core import ARMS,VARIANTS,write_json,require
from .rob01_corruptions import FAMILIES,conditions,condition_id
from .exp06_core import read_json


def plots(root,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'plots';out.mkdir(exist_ok=True)
    rows=result['metric_rows'];lengths=result['length_rows'];images=[]
    label='SYNTHETIC FIXTURE ONLY' if result['fixture_only'] else 'Existing held-out cohort; frozen models and temperatures'
    def save(fig,name,title):
        fig.suptitle(title+'\nROB-01 v0.1 | '+label,fontsize=12)
        fig.tight_layout(rect=(0,0,1,.90))
        for ext in ('png','pdf','svg'):fig.savefig(out/f'{name}.{ext}',dpi=150,bbox_inches='tight')
        plt.close(fig);images.append((name,title))
    def mean(items,key):
        vals=[r[key] for r in items if r.get(key) is not None]
        return np.mean(vals) if vals else np.nan
    for name,key,title,ylab in [
        ('nll','nll','Absolute NLL under incomplete observations','NLL (lower is better)'),
        ('deterioration','delta_rob_nll','Oriented NLL change from clean','Clean minus corrupt NLL (negative: deterioration)'),
        ('calibration','calibration_deterioration','Calibration deterioration; fixed clean temperatures','Corrupt minus clean ECE (positive: worse)')]:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for i,(family,levels) in enumerate(FAMILIES.items()):
            for j,v in enumerate(VARIANTS):
                ax=axes[i,j]
                for arm in ARMS:
                    yy=[]
                    for s in (0,*levels):
                        rr=[r for r in rows if r['arm']==arm and r['variant']==v and
                            r['family']==('clean' if s==0 else family) and r['severity']==s]
                        yy.append(0 if s==0 and name!='nll' else mean(rr,key))
                    ax.plot([0,*levels],yy,marker='o',label=arm)
                ax.set_title(f'{family} / {v}');ax.set_xlabel('Requested probability on eligible units');ax.set_ylabel(ylab)
                if name!='nll':ax.axhline(0,color='grey',lw=.8)
                ax.legend(fontsize=8)
        save(fig,name,title+'; means over seeds and corruption replicates')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,v in zip(axes,VARIANTS):
        for i,arm in enumerate(ARMS[1:]):
            r=result['comparisons'][v+':'+arm+'-T00'];lo,hi=r['percentile_95']
            ax.plot([lo,hi],[i,i],color='grey');ax.scatter(r['mean_delta'],i)
            ax.scatter(r['by_seed'],[i+.12]*3,marker='x',s=20,color='black')
        ax.set_yticks(range(5),ARMS[1:]);ax.axvline(0,color='black',lw=.8);ax.set_title(v)
        ax.set_xlabel('Difference in oriented NLL change (positive favours comparator)')
    save(fig,'paired','Macro robustness comparisons; descriptive 95% application-cluster intervals; x = seed means')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,(family,levels) in enumerate(FAMILIES.items()):
        for cls in (0,1):
            ax=axes[i,cls]
            for arm in ARMS:
                yy=[mean([r for r in rows if r['arm']==arm and r['variant']=='raw' and
                    r['family']==('clean' if s==0 else family) and r['severity']==s],f'classes.{cls}.recall') for s in (0,*levels)]
                ax.plot([0,*levels],yy,marker='o',label=arm)
            ax.set_ylim(0,1);ax.set_title(f'{family} / class {cls}');ax.set_xlabel('Requested probability');ax.set_ylabel('Recall');ax.legend(fontsize=8)
    save(fig,'recall','Classwise recall at threshold 0.5; raw/scaled decisions identical')
    fig,axes=plt.subplots(2,2,figsize=(12,8));bins=['1-10','11-20','21-40','41+']
    for i,family in enumerate(FAMILIES):
        for j,v in enumerate(VARIANTS):
            ax=axes[i,j]
            for arm in ARMS:
                yy=[mean([r for r in lengths if r['arm']==arm and r['variant']==v and r['family']==family and
                    r['severity']==.4 and r['original_length_stratum']==b],'nll') for b in bins]
                ax.plot(bins,yy,marker='o',label=arm)
            ax.set_title(f'{family} / {v} / severity 0.40');ax.set_ylabel('Mean NLL');ax.set_xlabel('Original clean prefix length');ax.legend(fontsize=8)
    save(fig,'length_nll','Length diagnostic at highest severity; original groups retained; means over seeds/replicates')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,(f,levels) in zip(axes,FAMILIES.items()):
        for rep in (1,2,3):
            aa=[read_json(root/'conditions'/condition_id(f,s,rep)/'audit.json') for s in levels]
            ax.plot(levels,[a['realised_rate'] if a['realised_rate'] is not None else np.nan for a in aa],marker='o',label=f'replicate {rep}')
        ax.plot([0,.4],[0,.4],':',color='black');ax.set_title(f);ax.set_xlabel('Requested probability');ax.set_ylabel('Realised fraction of eligible source units');ax.legend()
    save(fig,'corruption_rates','Actual corruption; denominators exclude existing missing values / protected events')
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>ROB-01 report</title>',
          '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px}img{width:100%}p{line-height:1.6}</style>',
          '<h1>ROB-01 controlled robustness report</h1>',f'<p>{html.escape(label)}</p>',
          '<p>Negative oriented NLL change means deterioration; positive paired advantage favours the comparator. Positive ECE change means worse calibration. Intervals condition on fixed models and fixed corruption realisations. No refitting. No causal or population-wide robustness claim.</p>',
          '<p>Each curve averages three training seeds and three corruption replicates. Tables retain every seed, replicate and length/class count. The 324 evaluations are dependent, not 324 independent experiments. GBDT seed repetitions may be identical.</p>']
    for name,title in images:
        uri=base64.b64encode((out/f'{name}.svg').read_bytes()).decode()
        body += [f'<h2>{html.escape(title)}</h2>',f'<img alt="{html.escape(title)}" src="data:image/svg+xml;base64,{uri}">']
    body.append('</html>');(out/'report.html').write_text('\n'.join(body),encoding='utf-8')
    require(all((out/f'{n}.{e}').is_file() for n,_ in images for e in ('png','pdf','svg')),'Missing plot export')
    return [n for n,_ in images]


def finish(root,fixture_only=False):
    from .rob01_core import aggregate
    result=aggregate(root,fixture_only)
    result['figures']=plots(root,result)
    result['status']='PASS'
    write_json(Path(root)/'rob01_summary.json',result)
    return result
