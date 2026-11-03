"""Offline HTML exploration plus PNG/SVG/PDF figures from saved EXP-04 metrics.

No network, CDN, external JavaScript library, or extra fitting. The HTML embeds
its figures and reliability data. The same runner invokes this after evaluation.
"""
from pathlib import Path
import csv
import html
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ipat.experiments.exp04_contract import FAMILIES, CONDITIONS, MODELS, PANELS, METRICS, macro_change
from ipat.experiments.exp03_contract import describe, file_hash

COLOURS = {"baseline": "#64748b", "retention": "#007f86", "full_ipat": "#b35112"}
LABELS = {"baseline": "Standard Transformer", "retention": "Retention-only", "full_ipat": "Full IPAT"}
TITLES = {"missing": "Numerical missingness", "delete": "Event deletion", "noise": "Numerical noise", "timejitter": "Temporal jitter"}


def write_report(summary, out_dir):
    folder = Path(out_dir)
    figures = folder/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    scope = ("Mean and approximate 95% t interval across training seeds; fixed datasets and corruption realisations."
             if len(summary["seeds"]) > 1 else "Single development seed; no confidence interval. Not research evidence.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 160})
    images, titles = {}, {}
    def save(fig, stem, title):
        fig.suptitle(title, fontsize=16, fontweight="bold")
        fig.text(.02, .015, f"{summary['status']} | {scope}", fontsize=8, color="#475569")
        fig.tight_layout(rect=(0, .055, 1, .94))
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(figures/f"{stem}.{suffix}", metadata={"Creator": "IPAT EXP-04"} if suffix == "pdf" else None)
        plt.close(fig)
        svg = (figures/f"{stem}.svg").read_text(encoding="utf-8")
        images[stem] = svg[svg.index("<svg"):]
        titles[stem] = title
    for panel, metric in [("raw", m) for m in ("nll", "auroc", "accuracy", "brier", "ece")] + [("temperature_scaled", "ece"), ("raw", "nll_change")]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (family, levels) in zip(axes.flat, FAMILIES.items()):
            x = np.array([.05 if family == "missing" else 0., *levels])
            conds = ["clean", *(f"{family}_{round(s*100):02d}" for s in levels)]
            for model in MODELS:
                if metric == "nll_change":
                    runs = [r for r in summary["runs"] if r["model"] == model]
                    stats = [describe([r["conditions"]["clean"][panel]["nll"]-r["conditions"][c][panel]["nll"] for r in runs]) for c in conds]
                else:
                    stats = [summary["aggregate"]["models"][model]["absolute"][c][panel][metric] for c in conds]
                y = np.array([v["mean"] for v in stats])
                lo = np.array([v["mean_95ci_t"][0] if v["mean_95ci_t"] else v["mean"] for v in stats])
                hi = np.array([v["mean_95ci_t"][1] if v["mean_95ci_t"] else v["mean"] for v in stats])
                ax.plot(x, y, "o-", label=LABELS[model], color=COLOURS[model], lw=2, ms=4)
                ax.fill_between(x, lo, hi, color=COLOURS[model], alpha=.12)
            ax.set_title(TITLES[family])
            ax.set_xticks(x, ["Clean\n(5% expected)" if family == "missing" else "Clean", *[f"{s:g}" for s in levels]])
            ax.set_xlabel("Target total missing rate" if family == "missing" else "Deletion fraction" if family == "delete" else "Noise scale q")
            ax.set_ylabel("NLL clean − corrupted (higher is better)" if metric == "nll_change" else metric.upper())
            ax.grid(alpha=.2)
            if metric == "nll_change":
                ax.axhline(0, color="#334155", lw=.8, ls="--")
        axes.flat[0].legend(fontsize=8)
        title = f"{'Fixed-temperature' if panel != 'raw' else 'Raw'} {metric.replace('_', ' ').upper()} across corruptions"
        save(fig, f"{panel}_{metric}", title)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, item) in enumerate(summary["aggregate"]["paired_primary_contrasts"].items()):
        vals = np.asarray(item["values"])
        ax.scatter(vals, i+np.linspace(-.09, .09, len(vals)), color="#007f86", s=24, alpha=.75)
        interval = item["mean_95ci_t"]
        if interval:
            ax.plot(interval, [i, i], color="#b35112", lw=3)
        ax.scatter(item["mean"], i, marker="D", color="#b35112", zorder=3)
    names = list(summary["aggregate"]["paired_primary_contrasts"])
    ax.set_yticks(range(len(names)), [n.replace("_minus_", " minus ").replace("full_ipat", "Full IPAT") for n in names])
    ax.axvline(0, color="#475569", ls="--")
    ax.set_xlabel("Difference in macro NLL change; positive favours the first model")
    ax.grid(axis="x", alpha=.2)
    save(fig, "paired_primary", "Primary paired comparisons — dots are individual training seeds")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, panel in zip(axes, PANELS):
        for i, model in enumerate(MODELS):
            item = summary["aggregate"]["models"][model]["macro_change"][panel]["ece"]
            ax.scatter(i+np.linspace(-.06, .06, len(item["values"])), item["values"], color=COLOURS[model], s=24)
            if item["mean_95ci_t"]:
                ax.plot([i, i], item["mean_95ci_t"], color=COLOURS[model], lw=3)
            ax.scatter(i, item["mean"], marker="D", color="black")
        ax.set_xticks(range(3), [LABELS[m].replace(" ", "\n") for m in MODELS])
        ax.set_title(panel.replace("_", " ").title())
        ax.axhline(0, color="#475569", ls="--")
        ax.set_ylabel("Macro ECE change; positive means deterioration")
        ax.grid(axis="y", alpha=.2)
    save(fig, "macro_calibration", "Calibration deterioration — equal weighting across corruption families")
    with (folder/"metrics_by_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "condition", "panel", *METRICS])
        for r in summary["runs"]:
            for c in CONDITIONS:
                for p in PANELS:
                    writer.writerow([r["model"], r["seed"], c, p, *[r["conditions"][c][p][m] for m in METRICS]])
    with (folder/"macro_changes_by_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "panel", "metric", "macro_change", "positive_means", *FAMILIES])
        for r in summary["runs"]:
            for p in PANELS:
                for m in METRICS:
                    item = macro_change(r["conditions"], p, m)
                    writer.writerow([r["model"], r["seed"], p, m, item["mean"], item["positive_means"], *item["per_family"].values()])
    reliability = [{"model": r["model"], "seed": r["seed"], "temperature": r["temperature"],
                    "conditions": {c: {p: {k: r["conditions"][c][p][k] for k in ("ece", "reliability")}
                                        for p in PANELS} for c in CONDITIONS}} for r in summary["runs"]]
    # JSON is embedded as data only; escape HTML delimiters and script terminators.
    embedded = json.dumps(reliability, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    chart_options = "".join(f'<option value="{k}">{html.escape(t)}</option>' for k, t in titles.items())
    chart_sections = "".join(f'<div class="chart" id="chart-{k}" hidden>{v}</div>' for k, v in images.items())
    cards = "".join(f'<article><h3>{LABELS[m]}</h3><strong>{summary["aggregate"]["models"][m]["macro_change"]["raw"]["nll"]["mean"]:+.5f}</strong><p>Mean raw NLL change<br>Higher is better</p></article>' for m in MODELS)
    contrast_rows = "".join(f'<tr><td>{html.escape(name)}</td><td>{v["mean"]:+.6f}</td><td>{v["secondary_signed_rank"]["pvalue"]:.6f}</td><td>{v["secondary_signed_rank"]["holm_pvalue"]:.6f}</td></tr>' for name,v in summary["aggregate"]["paired_primary_contrasts"].items())
    flags = "".join(f'<li>{html.escape(f)}</li>' for f in summary["review_flags"]) or '<li>No temperature boundary flags.</li>'
    page = '''<!doctype html><html lang="en-GB"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IPAT · EXP-04 corruption evaluation</title><style>
:root{font-family:system-ui,sans-serif;color:#17263b;background:#f3f6f8}body{max-width:1180px;margin:auto;padding:28px}header{border-top:6px solid #007f86;padding:24px;background:#fff;border-radius:10px}h1{font-size:32px;margin:8px 0}h2{font-size:23px}p{line-height:1.6}.badge{font-size:12px;font-weight:700;letter-spacing:.08em;color:#995015}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}article,section{background:white;padding:22px;border-radius:10px;margin-bottom:20px}article strong{font-size:32px}article h3{font-size:16px}select,button{padding:10px;font:inherit;border:1px solid #94a3b8;border-radius:5px;background:white;margin:6px}button{cursor:pointer}.chart svg{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #e2e8f0}.scroll{overflow:auto}.note{font-size:14px;color:#475569}code{overflow-wrap:anywhere}#reliability svg{width:100%;max-width:750px;height:auto}@media(max-width:700px){body{padding:12px}.cards{grid-template-columns:1fr}h1{font-size:26px}}@media print{select,button{display:none}.chart[hidden]{display:block}section{break-inside:avoid}}
</style><header><div class="badge">__STATUS__ · RESULTS REQUIRE INTERPRETATION</div><h1>EXP-04 · What survives corruption?</h1><p>Frozen-checkpoint comparison of predictive performance and calibration. Numerical missingness, event deletion, numerical noise and temporal jitter are evaluated independently.</p><p class="note">__RUNS__ checkpoints · __SEEDS__ paired training seeds · __COUNT__ primary test predictions. No model retraining.</p></header>
<div class="cards">__CARDS__</div><section><h2>Explore the measured changes</h2><label for="chart-select">Figure</label><select id="chart-select">__OPTIONS__</select><p class="note">__SCOPE__ NLL change: positive means improvement. ECE change: positive means deterioration. Consider absolute performance alongside changes.</p>__CHARTS__</section>
<section><h2>Reliability explorer</h2><p>One model and training seed at a time. The diagonal represents perfect agreement between confidence and observed accuracy. Empty bins are omitted. Temperature was fitted once on clean calibration data and held fixed.</p><label>Model <select id="model"></select></label><label>Seed <select id="seed"></select></label><label>Condition <select id="condition"></select></label><button id="export-reliability">Download this SVG</button><div id="reliability"></div><p id="reliability-note" class="note"></p></section>
<section><h2>Primary paired comparisons</h2><p class="note">Positive mean favours the first model. Secondary two-sided sign-flip tests on midranks; Holm adjustment across these three comparisons. Null inference assumes symmetric, independent seed differences.</p><div class="scroll"><table><thead><tr><th>Comparison</th><th>Mean advantage</th><th>Unadjusted p</th><th>Holm p</th></tr></thead><tbody>__CONTRASTS__</tbody></table></div></section>
<section><h2>Interpretation and provenance</h2><ul>__FLAGS__</ul><p>Clean data already contains 5% expected numerical missingness. Deleted events retain surviving source time fields. Labels refer to the original sequence. Distinct severity files are not nested perturbations.</p><p>Retention diagnostics are descriptive, not causal. Undefined scores are represented by NaN in prediction archives and null in JSON, with eligibility counts. Confidence intervals do not include uncertainty from resampling datasets or corruption realisations.</p><p>Full numerical evidence is in the adjacent JSON, CSV and prediction archives. Static figures are supplied as PNG, SVG and PDF.</p><p class="note">Fingerprint: <code>__FINGERPRINT__</code><br>Built from summary SHA-256: <code>__SUMMARYHASH__</code></p></section>
<script id="reliability-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('reliability-data').textContent);
const names={baseline:'Standard Transformer',retention:'Retention-only',full_ipat:'Full IPAT'};
const el=id=>document.getElementById(id);
function options(id,values){for(const v of values){const o=document.createElement('option');o.value=v;o.textContent=names[v]||v;el(id).append(o)}}
options('model',[...new Set(data.map(x=>x.model))]);options('seed',[...new Set(data.map(x=>x.seed))]);options('condition',Object.keys(data[0].conditions));
function chooseChart(){document.querySelectorAll('.chart').forEach(n=>n.hidden=n.id!=='chart-'+el('chart-select').value)}
el('chart-select').addEventListener('change',chooseChart);chooseChart();
function drawReliability(){const row=data.find(x=>x.model===el('model').value&&x.seed===Number(el('seed').value));const c=row.conditions[el('condition').value];const X=v=>70+600*v,Y=v=>390-320*v;
let svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 460" role="img" aria-label="Reliability diagram"><rect width="740" height="460" fill="white"/><text x="70" y="27" font-family="sans-serif" font-size="16">'+names[row.model]+' · seed '+row.seed+' · '+el('condition').value+'</text>';
for(let i=0;i<=5;i++){let v=i/5;svg+='<path d="M'+X(v)+' 70V390M70 '+Y(v)+'H670" stroke="#e2e8f0"/><text x="'+X(v)+'" y="415" text-anchor="middle" font-size="12">'+v.toFixed(1)+'</text><text x="55" y="'+(Y(v)+4)+'" text-anchor="end" font-size="12">'+v.toFixed(1)+'</text>'}
svg+='<path d="M70 390L670 70" stroke="#64748b" stroke-dasharray="5 5"/>';
for(const [panel,colour] of [['raw','#007f86'],['temperature_scaled','#b35112']]){const bins=c[panel].reliability.filter(b=>b.count>0);svg+='<polyline points="'+bins.map(b=>X(b.confidence)+','+Y(b.accuracy)).join(' ')+'" fill="none" stroke="'+colour+'" stroke-width="2"/>';for(const b of bins){svg+='<circle cx="'+X(b.confidence)+'" cy="'+Y(b.accuracy)+'" r="5" fill="'+colour+'"><title>'+panel+'; bin '+b.bin+'; n='+b.count+'; confidence='+b.confidence.toFixed(4)+'; accuracy='+b.accuracy.toFixed(4)+'</title></circle>'}}
svg+='<text x="365" y="450" text-anchor="middle" font-size="14">Top-label confidence</text><text x="20" y="225" transform="rotate(-90 20 225)" text-anchor="middle" font-size="14">Observed accuracy</text><text x="85" y="58" fill="#007f86">Raw</text><text x="170" y="58" fill="#b35112">Fixed temperature</text></svg>';
el('reliability').innerHTML=svg;el('reliability-note').textContent='T = '+row.temperature.temperature.toFixed(5)+'; raw ECE = '+c.raw.ece.toFixed(5)+'; scaled ECE = '+c.temperature_scaled.ece.toFixed(5)+'. Hover over a point for its bin count.'}
for(const id of ['model','seed','condition'])el(id).addEventListener('change',drawReliability);drawReliability();
el('export-reliability').addEventListener('click',()=>{const blob=new Blob([el('reliability').innerHTML],{type:'image/svg+xml'});const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='reliability_'+el('model').value+'_'+el('seed').value+'_'+el('condition').value+'.svg';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)});
</script></html>'''
    values = {"STATUS":summary["status"], "RUNS":str(len(summary["runs"])), "SEEDS":str(len(summary["seeds"])),
        "COUNT":f'{summary["primary_test_prediction_count"]:,}', "CARDS":cards, "OPTIONS":chart_options,
        "SCOPE":scope, "CHARTS":chart_sections, "CONTRASTS":contrast_rows, "FLAGS":flags,
        "FINGERPRINT":summary["fingerprint"], "SUMMARYHASH":file_hash(folder/"exp04_corruption_retention_v0.1.json"), "DATA":embedded}
    for key, value in values.items():
        page = page.replace(f"__{key}__", value)
    (folder/"exp04_report.html").write_text(page, encoding="utf-8")
    from ipat.experiments.exp03 import atomic_json
    outputs = [folder/"exp04_report.html", folder/"metrics_by_seed.csv", folder/"macro_changes_by_seed.csv", *sorted(figures.glob("*"))]
    atomic_json(folder/"report_manifest.json", {"summary_sha256": values["SUMMARYHASH"],
        "files": {p.relative_to(folder).as_posix(): file_hash(p) for p in outputs},
        "figure_count": len(images), "offline_html": True})
