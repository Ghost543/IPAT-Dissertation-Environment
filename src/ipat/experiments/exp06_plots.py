"""Scientific exports and a standalone offline HTML report from saved EXP-06 data."""
import base64
import html
from pathlib import Path
import numpy as np
from .exp06_core import ARMS, SEEDS, read_json, load_npz


def plots(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    root=Path(root);s=read_json(root/'exp06_summary.json')
    rr={(r['arm'],r['seed']):r for r in s['runs']}
    chosen={a:load_npz(root/f'{a}_1001'/'test_predictions.npz') for a in ARMS}
    ref=chosen['T00'];out=root/'plots';out.mkdir(exist_ok=True)
    scope='SYNTHETIC FIXTURE ONLY' if s['fixture_only'] else 'Held-out clean test'
    figures=[];colours={a:f'C{i}' for i,a in enumerate(ARMS)}
    def save(fig,name,title,seed_scope='All seeds: 1001, 1002, 1003'):
        fig.suptitle(f'{title}\nEXP-06 v0.1 | {scope} | {seed_scope}',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.90))
        for ext in ('png','pdf','svg'):
            fig.savefig(out/f'{name}.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig);figures.append((name,title))
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,v in zip(axes,('raw','scaled')):
        for i,a in enumerate(ARMS):
            vals=[rr[a,seed][v]['nll'] for seed in SEEDS]
            ax.scatter(i+np.array([-.1,0,.1]),vals,color=colours[a])
            ax.plot([i-.18,i+.18],[np.mean(vals)]*2,'k-',lw=2)
        ax.axhline(s['dummy']['nll'],color='black',ls=':',label='Unchanged train-prevalence dummy')
        ax.set(xticks=range(6),xticklabels=ARMS,ylabel='Offer-mean NLL (lower is better)',title=v.capitalize())
        ax.legend(fontsize=8)
    save(fig,'nll','Dots: individual seeds; bars: mean score, not an ensemble')
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,v in zip(axes[:2],('raw','scaled')):
        for i,a in enumerate(ARMS[1:]):
            r=s['comparisons'][f'{v}:{a}-T00'];lo,hi=r['percentile_95']
            ax.plot([lo,hi],[i,i],color='grey');ax.scatter(r['mean_delta'],i,color=colours[a])
        ax.axvline(0,color='black',lw=1);ax.set(yticks=range(5),yticklabels=ARMS[1:],title=v.capitalize(),xlabel='NLL minus T00; negative favours comparator')
    for i,a in enumerate(ARMS):
        r=s['comparisons'][f'calibration:{a}:scaled-raw'];lo,hi=r['percentile_95']
        axes[2].plot([lo,hi],[i,i],color='grey');axes[2].scatter(r['mean_delta'],i,color=colours[a])
    axes[2].axvline(0,color='black',lw=1)
    axes[2].set(yticks=range(6),yticklabels=ARMS,title='Within-model calibration',xlabel='Scaled minus raw NLL; negative favours scaling')
    save(fig,'paired','Paired means and descriptive 95% application-cluster intervals')
    fig,ax=plt.subplots(figsize=(8,5))
    cells=np.array([[s['summary'][a]['raw']['nll']['mean'] for a in ('T00','T01')],
                    [s['summary'][a]['raw']['nll']['mean'] for a in ('T10','T11')]])
    im=ax.imshow(cells,cmap='Blues');fig.colorbar(im,ax=ax,label='Raw NLL (lower is better)')
    for i in range(2):
        for j in range(2):ax.text(j,i,f'{cells[i,j]:.6f}',ha='center',va='center')
    ax.set(xticks=[0,1],xticklabels=['Standard attention','Retention'],yticks=[0,1],yticklabels=['Identity','Adaptive'])
    inter=s['comparisons']['raw:factorial_interaction']
    ax.set_xlabel('Interaction per seed: '+', '.join(f'{x:.6f}' for x in inter['by_seed'])+'\nColour range is narrow; read the numeric differences.')
    save(fig,'factorial','Raw 2 × 2 comparison; descriptive interaction')
    fig,axes=plt.subplots(1,4,figsize=(17,5))
    for ax,key in zip(axes,('nll','brier','ece','ece_equal_mass')):
        for i,a in enumerate(ARMS):
            for j,seed in enumerate(SEEDS):
                r=rr[a,seed];ax.scatter(i+(j-1)*.12,r['scaled'][key]-r['raw'][key],color=f'C{j}',label=str(seed) if i==0 else None)
        ax.axhline(0,color='black',lw=1);ax.set(xticks=range(6),xticklabels=ARMS,title=key,ylabel='Scaled minus raw (lower is better)')
        ax.tick_params(axis='x',rotation=45);ax.legend(fontsize=8)
    save(fig,'calibration_changes','Held-out changes from frozen temperatures; improvements are not guaranteed')
    for support in (False,True):
        fig,axes=plt.subplots(2,3,figsize=(13,8))
        for ax,a in zip(axes.flat,ARMS):
            for v in ('raw','scaled'):
                bins=[b for b in rr[a,1001][v]['reliability'] if b['count']]
                ax.plot([b['confidence'] for b in bins],[b['count' if support else 'accuracy'] for b in bins],'o-',label=v)
            if not support:
                ax.plot([.5,1],[.5,1],'k:');ax.set(ylim=(0,1.02))
            ax.set(title=a,xlim=(.48,1.02),xlabel='Mean top-label confidence',ylabel='Offer count' if support else 'Bin accuracy')
            ax.legend(fontsize=8)
        save(fig,'bin_support' if support else 'reliability','15 fixed-bin counts' if support else 'Top-label reliability; diagonal indicates agreement','Display seed 1001')
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for ax,a in zip(axes.flat,ARMS):
        cm=np.asarray(rr[a,1001]['raw']['confusion_matrix']);den=cm.sum(axis=1,keepdims=True)
        rates=np.divide(cm,den,out=np.zeros_like(cm,dtype=float),where=den!=0)
        ax.imshow(rates,vmin=0,vmax=1,cmap='Blues')
        for i in range(2):
            for j in range(2):
                txt=f'{cm[i,j]}\n{100*rates[i,j]:.1f}% of class' if den[i,0] else f'{cm[i,j]}\nNo class support'
                ax.text(j,i,txt,ha='center',va='center',color='white' if rates[i,j]>.6 else 'black',fontsize=9)
        ax.set(title=a,xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='True')
    save(fig,'confusion','Threshold 0.5; raw/scaled decisions identical; colour shows row proportions','Display seed 1001')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,c in zip(axes,(0,1)):
        target=ref['y']==c
        if len(np.unique(target))==2:
            for a in ARMS:
                prob=chosen[a]['raw_probability'];score=prob if c else 1-prob
                precision,recall,_=precision_recall_curve(target,score)
                ax.plot(recall,precision,label=a,color=colours[a])
        else:ax.text(.1,.5,'Ranking undefined: one class absent',transform=ax.transAxes)
        ax.axhline(float(target.mean()),color='black',ls=':',label='Test class prevalence')
        ax.set(title=f'Class {c}',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=8)
    save(fig,'precision_recall','Raw ranking; positive temperature scaling preserves order in exact arithmetic','Display seed 1001')
    strata=('1-10','11-20','21-40','41+')
    masks=[(ref['lengths']>=lo)&((ref['lengths']<=hi) if hi else True) for lo,hi in ((1,10),(11,20),(21,40),(41,None))]
    counts=[f'{name}\nn={int(m.sum())}; 0/1={int(((ref["y"]==0)&m).sum())}/{int(((ref["y"]==1)&m).sum())}' for name,m in zip(strata,masks)]
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for ax,v in zip(axes,('raw','scaled')):
        for a in ARMS:
            vals=[np.mean([rr[a,seed][v+'_length'][k]['nll'] for seed in SEEDS]) if rr[a,1001][v+'_length'][k]['n'] else np.nan for k in strata]
            ax.plot(range(4),vals,'o-',label=a,color=colours[a])
        ax.set(xticks=range(4),xticklabels=counts,ylabel='Mean seed NLL (lower is better)',title=v.capitalize());ax.legend(fontsize=8)
    save(fig,'length_nll','Complete-prefix strata; counts show test class composition')
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    for ax,c in zip(axes,('0','1')):
        for a in ARMS:
            vals=[]
            for k in strata:
                ms=[rr[a,seed]['raw_length'][k] for seed in SEEDS]
                vals.append(np.mean([m['classes'][c]['recall'] for m in ms]) if all(m['n'] and m['classes'][c]['recall'] is not None for m in ms) else np.nan)
            ax.plot(range(4),vals,'o-',label=a,color=colours[a])
        ax.set(xticks=range(4),xticklabels=counts,ylim=(0,1.03),ylabel='Mean seed recall',title='Class '+c);ax.legend(fontsize=8)
    save(fig,'length_recall','Class recall by length; unchanged by scaling; gaps mean absent support')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    bins=np.arange(ref['lengths'].min()-.5,ref['lengths'].max()+1.5)
    for c in (0,1):axes[0].hist(ref['lengths'][ref['y']==c],bins=bins,alpha=.55,label='Class '+str(c))
    axes[0].set(xlabel='Full prefix length',ylabel='Offer count');axes[0].legend()
    x=np.sort(ref['lengths']);axes[1].step(x,np.arange(1,len(x)+1)/len(x),where='post')
    axes[1].set(xlabel='Full prefix length',ylabel='Cumulative offer fraction',ylim=(0,1.02))
    save(fig,'length_distribution','Full length distribution; tail retained','Test cohort, independent of model seed')
    contents=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>EXP-06 clean evaluation</title>',
      '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 22px;color:#172233}img{width:100%}table{border-collapse:collapse;width:100%}th,td{padding:8px;border:1px solid #ddd;text-align:left}summary{cursor:pointer;font-weight:600}details{margin:24px 0}a{color:#185fa5}.note{background:#eef4fa;padding:16px}</style></head><body>',
      '<h1>EXP-06: frozen clean evaluation</h1><p class="note">'+html.escape(scope)+'. All model weights, thresholds and temperatures are frozen. Negative findings remain valid results. Three seeds are not three independent datasets.</p>',
      f'<p>{s["n"]:,} offers from {s["applications"]:,} applications. Primary contrast: raw T11 minus T00 NLL. Negative favours T11.</p>',
      '<p>Intervals resample whole applications and are descriptive, conditional on the fitted predictors. Reliability panels use top-label confidence. Temperature scaling cannot repair class recall at the fixed decision threshold. GBDT seed duplicates are disclosed in the summary.</p>',
      '<table><tr><th>Arm / seed</th><th>T</th><th>Raw NLL</th><th>Scaled NLL</th><th>Raw ECE</th><th>Scaled ECE</th></tr>']
    for r in s['runs']:
        contents.append(f'<tr><td>{r["arm"]}/{r["seed"]}</td><td>{r["temperature"]:.6f}</td><td>{r["raw"]["nll"]:.6f}</td><td>{r["scaled"]["nll"]:.6f}</td><td>{r["raw"]["ece"]:.6f}</td><td>{r["scaled"]["ece"]:.6f}</td></tr>')
    contents.append('</table>')
    for name,title in figures:
        img=base64.b64encode((out/(name+'.svg')).read_bytes()).decode('ascii')
        contents.append(f'<details open><summary>{html.escape(title)}</summary><img src="data:image/svg+xml;base64,{img}" alt="{html.escape(title)}"><p><a href="{name}.png">PNG</a> · <a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p></details>')
    contents.append('</body></html>')
    from .exp05_io import atomic_write
    page='\n'.join(contents).encode('utf-8');atomic_write(out/'report.html',lambda f:f.write(page))
    return figures
