"""Offline plots from saved CAL-01 outputs, with fitting-sample scope explicit."""
import html
import json
from pathlib import Path
import numpy as np
from .cal01_core import nll, scale_logits, T_MIN, T_MAX


def plots(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    s = json.loads((root/'cal01_summary.json').read_text(encoding='utf-8'))
    rows = s['runs']
    arms = list(dict.fromkeys(r['arm'] for r in rows))
    seeds = sorted(set(r['seed'] for r in rows))
    colours = {seed:f'C{i}' for i,seed in enumerate(seeds)}
    out = root/'plots'
    out.mkdir(exist_ok=True)
    figures = []
    scope = 'SYNTHETIC FIXTURE ONLY' if s.get('fixture_only') else 'Calibration FIT partition; no test evaluation'
    def save(fig,name,title):
        fig.suptitle(title+'\nCAL-01 v0.1 | '+scope,fontsize=13)
        fig.tight_layout(rect=(0,0,1,.91))
        for extension in ('png','pdf','svg'):
            fig.savefig(out/(name+'.'+extension),dpi=160,bbox_inches='tight')
        plt.close(fig)
        figures.append((name,title))
    fig,ax = plt.subplots(figsize=(10,5))
    for i,arm in enumerate(arms):
        rs = [r for r in rows if r['arm']==arm]
        for offset,r in zip(np.linspace(-.12,.12,len(rs)),rs):
            ax.scatter(i+offset,r['fit']['temperature'],color=colours[r['seed']],
                       label=str(r['seed']) if i==0 else None,
                       marker='x' if r['fit']['status']!='INTERIOR' else 'o')
    ax.axhline(1,color='black',linestyle='--',label='T=1: unchanged')
    ax.set(xticks=range(len(arms)),xticklabels=arms,yscale='log',ylabel='Temperature (log scale)')
    ax.legend()
    save(fig,'temperatures','All seed temperatures; x marks a solution requiring review')
    fig,axes = plt.subplots(1,4,figsize=(17,5))
    for ax,key in zip(axes,('nll','brier','ece','ece_equal_mass')):
        for i,arm in enumerate(arms):
            rs = [r for r in rows if r['arm']==arm]
            for r in rs:
                ax.scatter(i,r['scaled'][key]-r['raw'][key],s=25,color=colours[r['seed']],
                           label=str(r['seed']) if i==0 else None)
        ax.axhline(0,color='black',linewidth=1)
        ax.set(xticks=range(len(arms)),xticklabels=arms,title=key,ylabel='Scaled minus raw (lower is better)')
        ax.tick_params(axis='x',rotation=45)
        ax.legend(fontsize=8)
    save(fig,'fit_metric_changes','Paired changes on the same data used to fit T; optimistic fitting diagnostics')
    display_seed = min(r['seed'] for r in rows)
    chosen = [r for r in rows if r['seed']==display_seed]
    fig,axes = plt.subplots(2,3,figsize=(13,8))
    for ax,r in zip(axes.flat,chosen):
        for stage,label in (('raw','Raw'),('scaled','Temperature scaled')):
            bins = [b for b in r[stage]['reliability'] if b['count']]
            ax.plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],'o-',label=label)
        ax.plot([.5,1],[.5,1],'k:')
        ax.set(title=r['arm'],xlim=(.48,1.02),ylim=(0,1.02),xlabel='Mean top-label confidence',ylabel='Bin accuracy')
        ax.legend(fontsize=8)
    save(fig,'reliability',f'Before/after reliability; 15 fixed bins; display seed {display_seed}')
    fig,axes = plt.subplots(2,3,figsize=(13,8))
    for ax,r in zip(axes.flat,chosen):
        for stage,label in (('raw','Raw'),('scaled','Temperature scaled')):
            bins = [b for b in r[stage]['reliability'] if b['count']]
            ax.plot([b['confidence'] for b in bins],[b['count'] for b in bins],'o-',label=label)
        ax.set(title=r['arm'],xlabel='Mean top-label confidence',ylabel='Number of offers')
        ax.legend(fontsize=8)
    save(fig,'bin_support',f'Reliability bin counts; display seed {display_seed}')
    fig,axes = plt.subplots(2,3,figsize=(13,8))
    temperatures = np.geomspace(T_MIN,T_MAX,101)
    for ax,r in zip(axes.flat,chosen):
        with np.load(root/f"{r['arm']}_{r['seed']}"/'calibration_predictions.npz',allow_pickle=False) as a:
            losses = [nll(a['y'],scale_logits(a['raw_logits'],t)) for t in temperatures]
        ax.plot(temperatures,losses)
        ax.axvline(1,color='grey',linestyle='--')
        ax.scatter([r['fit']['temperature']],[r['fit']['fitted_nll']],color='black')
        ax.set(xscale='log',xlabel='Temperature',ylabel='Fit NLL',title=r['arm'])
    save(fig,'temperature_objective',f'Objective profiles; dots are fitted solutions; display seed {display_seed}')
    contents = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>CAL-01 report</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 24px}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>',
        '<h1>CAL-01 temperature fitting</h1><p><strong>'+html.escape(scope)+'</strong></p>',
        '<p>Temperature changes confidence. Model weights and decisions at threshold 0.5 stay fixed. Fit-sample changes are not independent calibration benefits. ECE may worsen even when fitting NLL improves.</p>',
        '<table><tr><th>Arm / seed</th><th>T</th><th>Raw NLL</th><th>Fit NLL</th><th>Solver status</th></tr>']
    for r in rows:
        contents.append(f"<tr><td>{html.escape(r['arm'])}/{r['seed']}</td><td>{r['fit']['temperature']:.6f}</td><td>{r['raw']['nll']:.6f}</td><td>{r['scaled']['nll']:.6f}</td><td>{html.escape(r['fit']['status'])}</td></tr>")
    contents.append('</table>')
    for name,title in figures:
        contents.append(f'<h2>{html.escape(title)}</h2><img src="{name}.svg" alt="{html.escape(title)}"><p><a href="{name}.png">PNG</a> | <a href="{name}.pdf">PDF</a></p>')
    contents.append('</html>')
    (out/'report.html').write_text('\n'.join(contents),encoding='utf-8')
    return figures
