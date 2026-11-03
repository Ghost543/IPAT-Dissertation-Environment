"""Audit an explicit BPI17 contract; no tensor preprocessing or model fitting.

Standard-library preflight; --plots additionally needs matplotlib.
CSV retains all candidates, including application-boundary exclusions.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import html
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dt(s):
    t = datetime.fromisoformat(s.replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError('Timestamp requires a timezone')
    return t.astimezone(timezone.utc)


def attributes(element):
    return {a.get('key'): a.get('value') for a in element if a.get('key') is not None}


def candidates(path):
    """Yield one application at a time; never expose later fields as inputs."""
    seen_apps, seen_offers = set(), set()
    with gzip.open(path, 'rb') as f:
        parser = ET.iterparse(f, events=('start', 'end'))
        _, root = next(parser)
        for action, node in parser:
            if action != 'end' or node.tag.split('}')[-1] != 'trace':
                continue
            aid = attributes(node).get('concept:name')
            if not aid or aid in seen_apps:
                raise ValueError('Missing/duplicate application ID')
            seen_apps.add(aid)
            events = [attributes(e) for e in node if e.tag.split('}')[-1] == 'event']
            previous, offers = None, []
            for index, e in enumerate(events):
                t = dt(e['time:timestamp'])
                if previous is not None and t <= previous:
                    raise ValueError('Events must be strictly increasing; do not silently reorder')
                previous = t
                if e.get('concept:name') != 'O_Create Offer':
                    continue
                oid = e.get('EventID')
                if not oid or oid in seen_offers:
                    raise ValueError('Missing/duplicate offer ID')
                seen_offers.add(oid)
                if e.get('Accepted') not in ('true', 'false'):
                    raise ValueError('Non-Boolean recorded Accepted target')
                offers.append(dict(application_id=aid, offer_id=oid,
                    creation_time=t.isoformat(), label=int(e['Accepted'] == 'true'),
                    prefix_length=index+1))
            yield aid, len(events), offers
            node.clear()
            root.clear()


def assign_application(offers, contract):
    boundaries = [dt(x) for x in contract['split']['boundaries']]
    names = contract['split']['names']
    if boundaries != sorted(set(boundaries)) or len(names) != len(boundaries)+1:
        raise ValueError('Invalid split definition')
    provisional = [names[bisect_right(boundaries, dt(o['creation_time']))] for o in offers]
    crossing = len(set(provisional)) > 1
    return [dict(o, provisional_split=s, split='excluded' if crossing else s,
                 exclusion_reason='application_crosses_boundary' if crossing else '')
            for o, s in zip(offers, provisional)]


def summarise(rows):
    lengths = sorted(r['prefix_length'] for r in rows)
    counts = Counter(r['label'] for r in rows)
    def quantile(p):
        x = (len(lengths)-1)*p
        i = int(x)
        return lengths[i] + (lengths[min(i+1, len(lengths)-1)]-lengths[i])*(x-i)
    if not rows:
        return {'offers': 0, 'applications': 0, 'labels': {'0': 0, '1': 0}}
    strata = {}
    for name, lo, hi in [('1–10',1,10), ('11–20',11,20), ('21–40',21,40), ('41+',41,float('inf'))]:
        subset = [r for r in rows if lo <= r['prefix_length'] <= hi]
        c = Counter(r['label'] for r in subset)
        strata[name] = {'offers': len(subset), 'labels': {'0': c[0], '1': c[1]}}
    return dict(offers=len(rows), applications=len({r['application_id'] for r in rows}),
        labels={'0': counts[0], '1': counts[1]}, positive_fraction=counts[1]/len(rows),
        length=dict(min=lengths[0], median=quantile(.5), mean=statistics.fmean(lengths),
                    p95=quantile(.95), p99=quantile(.99), max=lengths[-1]),
        length_histogram=dict(sorted(Counter(lengths).items())), length_strata=strata)


def preflight(path, contract):
    sha = digest(path)
    if sha != contract['source_sha256']:
        raise ValueError('Source SHA-256 mismatch; do not apply this contract to another archive')
    rows, app_count, event_count = [], 0, 0
    for _, events, offers in candidates(path):
        app_count += 1
        event_count += events
        rows.extend(assign_application(offers, contract))
    actual = dict(applications=app_count, events=event_count, offers=len(rows))
    names = contract['split']['names']
    groups = {s: [r for r in rows if r['split'] == s] for s in names+['excluded']}
    sets = {s: {r['application_id'] for r in groups[s]} for s in groups}
    disjoint = all(not sets[a] & sets[b] for i, a in enumerate(sets) for b in list(sets)[i+1:])
    # Independently express date membership as half-open intervals, not bisect.
    dates = [dt(x) for x in contract['split']['boundaries']]
    membership = True
    for i, s in enumerate(names):
        membership &= all((i == 0 or dt(r['creation_time']) >= dates[i-1]) and
                          (i == len(names)-1 or dt(r['creation_time']) < dates[i]) for r in groups[s])
    checks = dict(source_hash_matches=True, source_counts_match=actual == contract['expected_source_counts'],
        parser_ids_targets_timestamps_valid=True,
        candidate_accounting=sum(len(v) for v in groups.values()) == len(rows),
        application_disjoint=disjoint, nonempty_splits=all(groups[s] for s in names),
        both_labels_in_each_split=all({r['label'] for r in groups[s]} == {0,1} for s in names),
        retained_date_membership=bool(membership),
        positive_prefix_lengths=all(r['prefix_length'] >= 1 for r in rows))
    return dict(contract_version=contract['version'], scope='Dataset contract preflight only; no model results or tensor acceptance.',
        status='PASS' if all(checks.values()) else 'REVIEW_REQUIRED', source_sha256=sha,
        source_counts=actual, checks=checks, candidate=summarise(rows),
        partitions={s: summarise(groups[s]) for s in groups},
        readiness='Adapter acceptance and user-local replication pending; source review limitations remain.'), rows


def plot(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    names = ['train','validation','calibration','test']
    colours = ['#2878b5','#e07b39','#39977a','#9a67ad']
    for s,c in zip(names,colours):
        h = report['partitions'][s]['length_histogram']
        xs = sorted(map(int,h)); total=sum(h.values()); running=0; ys=[]
        for x in xs:
            running += h.get(x, h.get(str(x), 0)); ys.append(running/total)
        ax[0,0].step(xs,ys,where='post',label=s,color=c)
    ax[0,0].set(xlabel='Complete prefix length (events)',ylabel='Cumulative fraction',title='Length distribution by admitted partition',ylim=(0,1.02))
    ax[0,0].legend(fontsize=8)
    positives = [report['partitions'][s]['positive_fraction']*100 for s in names]
    ax[0,1].bar(names,positives,color=colours)
    for i,v in enumerate(positives): ax[0,1].text(i,v+1,f'{v:.1f}%',ha='center')
    ax[0,1].set(ylabel='Recorded Accepted=true (%)',ylim=(0,100),title='Natural class prevalence; no resampling')
    strata=report['candidate']['length_strata']
    left=[v['labels']['0'] for v in strata.values()];right=[v['labels']['1'] for v in strata.values()]
    ax[1,0].bar(list(strata),left,label='false / 0',color='#d7793e')
    ax[1,0].bar(list(strata),right,bottom=left,label='true / 1',color='#2878b5')
    ax[1,0].set(xlabel='Prefix length stratum',ylabel='Candidate offers',title='Length and label jointly (before exclusions)')
    ax[1,0].legend(fontsize=8)
    groups=names+['excluded']; counts=[report['partitions'][s]['offers'] for s in groups]
    ax[1,1].bar(groups,counts,color=colours+['#999999'])
    for i,v in enumerate(counts):ax[1,1].text(i,v+max(counts)*.015,f'{v:,}',ha='center',fontsize=9)
    ax[1,1].set(ylabel='Offers',title='All candidates accounted for',ylim=(0,max(counts)*1.17))
    ax[1,1].tick_params(axis='x',labelsize=8)
    fig.suptitle('BPI17 contract preflight v0.1 — descriptive evidence, no model results',fontsize=14)
    for a in ax.flat:a.spines[['top','right']].set_visible(False)
    for ext in ['png','svg','pdf']:fig.savefig(directory/f'contract_preflight.{ext}',dpi=170)
    plt.close(fig)
    # Inline SVG gives an offline, self-contained report without external scripts.
    svg=(directory/'contract_preflight.svg').read_text(encoding='utf-8')
    svg=svg[svg.index('<svg'):]
    body=''.join(f'<tr><td>{html.escape(s)}</td><td>{v["applications"]:,}</td><td>{v["offers"]:,}</td><td>{v["labels"]["0"]:,}</td><td>{v["labels"]["1"]:,}</td></tr>' for s,v in report['partitions'].items())
    (directory/'contract_preflight.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>BPI17 contract preflight</title><style>body{font:16px system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#203040}svg{width:100%;height:auto}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}table{border-collapse:collapse;width:100%}details{margin:24px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>BPI17 dataset contract preflight</h1><p>Descriptive evidence only. Complete prefixes and natural class proportions are retained. All offers of boundary-crossing applications are excluded.</p>'+svg+'<table><tr><th>Partition</th><th>Applications</th><th>Offers</th><th>False / 0</th><th>True / 1</th></tr>'+body+'</table><details><summary>Scope and acceptance evidence</summary><pre>'+html.escape(json.dumps(report,indent=2))+'</pre></details><p>Tensor adapter and local replication remain pending. Field availability and target semantics are limited to the retrospective archive.</p></html>',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--contract',type=Path,default=Path(__file__).resolve().parents[1]/'dataset_contract.json')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--plots',action='store_true')
    args=p.parse_args()
    contract=json.loads(args.contract.read_text(encoding='utf-8'))
    report,rows=preflight(args.source,contract)
    report['contract_sha256']=digest(args.contract)
    report['script_sha256']=digest(Path(__file__))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    csv_path=args.output_dir/'offer_split_manifest.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['application_id','offer_id','creation_time','label','prefix_length','provisional_split','split','exclusion_reason'])
        writer.writeheader();writer.writerows(rows)
    report['manifest_sha256']=digest(csv_path)
    (args.output_dir/'contract_preflight.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    if args.plots:plot(report,args.output_dir)
    print(json.dumps({k:report[k] for k in ['status','scope','source_counts','checks','readiness']},indent=2))
    if report['status'] != 'PASS':raise SystemExit(1)


if __name__ == '__main__':main()
