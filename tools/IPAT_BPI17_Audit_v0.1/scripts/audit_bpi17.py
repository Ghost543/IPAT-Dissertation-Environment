"""Read-only BPI17 source audit. No fitting, splitting or model evaluation.

Uses only the Python standard library; optional plots require matplotlib.
An offer prefix ends at its O_Create Offer event in original source order.
The 2016-10-20 boundary is an EBES reference diagnostic, not a frozen IPAT split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

NUM = ['RequestedAmount', 'FirstWithdrawalAmount', 'NumberOfTerms',
       'MonthlyCost', 'CreditScore', 'OfferedAmount']
CAT = ['Action', 'org:resource', 'concept:name', 'EventOrigin',
       'lifecycle:transition', 'LoanGoal', 'ApplicationType']
CUTOFF = '2016-10-20T00:00:00+00:00'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def attrs(element):
    return {c.get('key'): c.get('value') for c in element
            if c.tag.split('}')[-1] != 'event' and c.get('key') is not None}


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timezone missing')
    return result


def describe(values):
    if not values:
        return {'n': 0}
    a = sorted(values)
    def quantile(p):
        x = (len(a) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return a[lo] + (a[hi] - a[lo]) * (x - lo)
    return dict(n=len(a), min=a[0], median=quantile(.5), p95=quantile(.95),
                p99=quantile(.99), max=a[-1], mean=statistics.fmean(a))


def analyse_trace(trace):
    ta = attrs(trace)
    events = [attrs(e) for e in trace if e.tag.split('}')[-1] == 'event']
    times, invalid = [], 0
    for e in events:
        try:
            times.append(timestamp(e['time:timestamp']))
        except (KeyError, ValueError, TypeError):
            times.append(None)
            invalid += 1
    decreases = sum(a > b for a, b in zip(times, times[1:]) if a and b)
    ties = sum(a == b for a, b in zip(times, times[1:]) if a and b)
    offers, available, prefix_missing, prefix_total = [], Counter(), Counter(), 0
    for i, e in enumerate(events):
        for k in NUM + CAT:
            value = ta.get(k) if k in ('RequestedAmount', 'LoanGoal', 'ApplicationType') else e.get(k)
            if value is not None and value != '':
                available[k] += 1
        if e.get('concept:name') == 'O_Create Offer':
            next_e = events[i+1] if i+1 < len(events) else {}
            oid = e.get('EventID')
            later = [x for x in events[i+1:] if x.get('OfferID') == oid]
            later_accept = [x for x in later if x.get('concept:name') == 'O_Accepted']
            offers.append(dict(offer_id=oid, label=e.get('Accepted'),
                time=times[i].isoformat() if times[i] else None, prefix_length=i+1,
                next_event_is_created=next_e.get('concept:name') == 'O_Created',
                next_id_matches=bool(oid) and next_e.get('OfferID') == oid,
                later_accepted_event=bool(later_accept),
                outcome_events=[x.get('concept:name') for x in later if x.get('concept:name') in
                                ('O_Accepted', 'O_Refused', 'O_Cancelled')],
                creation_fields={k: ta.get(k) if k == 'RequestedAmount' else e.get(k) for k in NUM},
                selected=e.get('Selected')))
            prefix_total += i+1
            prefix_missing.update({k: i+1-available[k] for k in NUM + CAT})
    return ta, events, times, invalid, decreases, ties, offers, prefix_missing, prefix_total


def audit(path, cutoff=CUTOFF):
    counts, event_names, fields, trace_fields = Counter(), Counter(), Counter(), Counter()
    types, cat_values = defaultdict(Counter), defaultdict(set)
    app_ids, offer_ids = set(), set()
    app_lengths, prefix_lengths, offers_per_app = [], [], []
    labels, agreement, split, cross_labels = Counter(), Counter(), Counter(), Counter()
    missing = Counter(); prefix_cells = 0
    creation_values = defaultdict(list); zeros = Counter(); nonfinite = Counter()
    months = Counter(); cross_apps = 0; cross_offers = 0
    minimum, maximum = None, None
    metadata = {}; cutoff_dt = timestamp(cutoff)
    with gzip.open(path, 'rb') as stream:
        context = ET.iterparse(stream, events=('start', 'end'))
        _, root = next(context)
        for event, element in context:
            if event != 'end':
                continue
            tag = element.tag.split('}')[-1]
            if tag != 'trace':
                if element.get('key') in ('meta_concept:named_events_total', 'meta_general:traces_total'):
                    metadata[element.get('key')] = int(element.get('value'))
                continue
            ta, es, ts, bad, desc, ties, offers, miss, total = analyse_trace(element)
            counts['applications'] += 1; counts['events'] += len(es)
            counts['invalid_timestamps'] += bad; counts['adjacent_timestamp_decreases'] += desc
            counts['adjacent_timestamp_ties'] += ties
            aid = ta.get('concept:name')
            counts['missing_application_ids'] += not bool(aid)
            counts['duplicate_application_ids'] += aid in app_ids
            app_ids.add(aid); app_lengths.append(len(es)); offers_per_app.append(len(offers))
            counts['applications_without_offers'] += not offers
            trace_fields.update(ta.keys())
            counts['exact_duplicate_events_within_application'] += len(es)-len({json.dumps(e,sort_keys=True) for e in es})
            for e, t in zip(es, ts):
                event_names[e.get('concept:name', '<MISSING>')] += 1
                fields.update(e.keys())
                if t:
                    minimum = min(minimum,t) if minimum else t
                    maximum = max(maximum,t) if maximum else t
                for k in CAT:
                    v = ta.get(k) if k in ('LoanGoal','ApplicationType') else e.get(k)
                    if v is not None: cat_values[k].add(v)
            for child in element:
                if child.tag.split('}')[-1] == 'event':
                    for a in child: types[a.get('key')][a.tag.split('}')[-1]] += 1
            app_sides=set()
            for o in offers:
                oid=o['offer_id']; counts['offers'] += 1
                counts['missing_offer_ids'] += not bool(oid)
                counts['duplicate_offer_ids'] += oid in offer_ids; offer_ids.add(oid)
                counts['next_offer_id_mismatches'] += not o['next_id_matches']
                counts['next_event_not_O_Created'] += not o['next_event_is_created']
                labels[str(o['label'])] += 1
                counts['invalid_offer_labels'] += o['label'] not in ('true','false')
                agreement[str(o['label'])+'|later_O_Accepted='+str(o['later_accepted_event'])] += 1
                counts['offers_without_terminal_offer_event'] += not o['outcome_events']
                prefix_lengths.append(o['prefix_length'])
                if o['time']:
                    side='before' if timestamp(o['time']) < cutoff_dt else 'on_or_after'
                    split[side] += 1; app_sides.add(side); months[o['time'][:7]] += 1
                for k,v in o['creation_fields'].items():
                    if v is not None:
                        try:
                            n=float(v)
                            if math.isfinite(n): creation_values[k].append(n); zeros[k] += n == 0
                            else: nonfinite[k] += 1
                        except ValueError: nonfinite[k] += 1
            if len(app_sides)>1:
                cross_apps += 1; cross_offers += len(offers)
                cross_labels.update(o['label'] for o in offers)
            missing.update(miss); prefix_cells += total
            element.clear(); root.clear()
    gates={
        'gzip_and_xml_complete': True,
        'event_count_matches_embedded_metadata': counts['events']==metadata.get('meta_concept:named_events_total'),
        'unique_application_ids': counts['duplicate_application_ids']==0 and counts['missing_application_ids']==0,
        'unique_offer_ids': counts['duplicate_offer_ids']==0 and counts['missing_offer_ids']==0,
        'timestamps_parse': counts['invalid_timestamps']==0,
        'source_order_nondecreasing': counts['adjacent_timestamp_decreases']==0,
        'offer_labels_boolean': counts['invalid_offer_labels']==0,
        'creation_id_matches_next_offer_id': counts['next_offer_id_mismatches']==0,
    }
    return dict(audit_version='v0.1', scope='Raw-source and candidate-prefix audit only; no model results or frozen training split.',
        source_sha256=sha256(path), source_bytes=path.stat().st_size,
        counts=dict(counts), embedded_metadata=metadata, integrity_gates=gates,
        integrity_status='PASS' if all(gates.values()) else 'REVIEW_REQUIRED',
        readiness_status='REVIEW_REQUIRED',
        event_time_range=[minimum.isoformat() if minimum else None,maximum.isoformat() if maximum else None],
        application_length=describe(app_lengths), candidate_prefix_length=describe(prefix_lengths),
        offers_per_application=dict(sorted(Counter(offers_per_app).items())),
        application_length_histogram=dict(sorted(Counter(app_lengths).items())),
        candidate_prefix_length_histogram=dict(sorted(Counter(prefix_lengths).items())),
        candidate_labels=dict(labels), label_vs_later_event=dict(agreement),
        trace_field_presence=dict(trace_fields), event_field_presence=dict(fields), event_field_types=dict(types),
        event_name_counts=dict(event_names), categorical_cardinalities={k:len(v) for k,v in cat_values.items()},
        candidate_prefix_event_occurrences=prefix_cells,
        candidate_prefix_missing_counts=dict(missing),
        candidate_prefix_missing_rates={k:v/prefix_cells for k,v in missing.items()} if prefix_cells else {},
        creation_numeric_summary={k:dict(describe(creation_values[k]),zero_count=zeros[k],invalid_count=nonfinite[k]) for k in NUM},
        offer_creation_months=dict(sorted(months.items())),
        reference_boundary=dict(cutoff=cutoff,offer_counts=dict(split),crossing_applications=cross_apps,
                                offers_in_crossing_applications=cross_offers,labels_in_crossing_applications=dict(cross_labels)),
        unresolved=['Verify full terms and exact repository version from official source.',
                    'Resolve Accepted attribute versus later O_Accepted semantics before target freeze.',
                    'Freeze application-disjoint chronological split, including label-availability policy.',
                    'Determine whether numeric zero values (especially CreditScore) are genuine or sentinels.',
                    'Pin schema and prediction-time field availability; no backward filling from future events.'])


def render(report, output):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    hist=report['candidate_prefix_length_histogram']
    ax[0,0].bar([int(x) for x in hist],list(hist.values()),color='#24788e')
    ax[0,0].set(xlabel='Events through offer creation',ylabel='Offer prefixes',title='Candidate sequence lengths')
    labels=report['candidate_labels']; ax[0,1].bar(list(labels),list(labels.values()),color=['#24788e','#db934b'][:len(labels)])
    ax[0,1].set(xlabel='Recorded Accepted attribute',ylabel='Offers',title='Candidate labels: semantics under review')
    rates=report['candidate_prefix_missing_rates']
    ax[1,0].barh(NUM,[rates.get(k,0)*100 for k in NUM],color='#24788e')
    ax[1,0].set(xlabel='Absent cells (%)',xlim=(0,100),title='Raw missingness across candidate prefixes')
    months=report['offer_creation_months']; ax[1,1].bar(list(months),list(months.values()),color='#24788e')
    ax[1,1].tick_params(axis='x',rotation=60)
    ax[1,1].set(ylabel='Offers',title='Offer creation by month (UTC)')
    fig.suptitle('BPI17 source audit — observed data, no trained models',fontsize=15)
    for ext in ('png','svg','pdf'): fig.savefig(output/f'bpi17_audit.{ext}',dpi=160)
    plt.close(fig)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cutoff',default=CUTOFF)
    p.add_argument('--plots',action='store_true')
    args=p.parse_args(); out=args.output_dir
    if out.exists() and any(out.iterdir()):
        p.error('Output directory must be empty or new; existing evidence is not overwritten.')
    out.mkdir(parents=True,exist_ok=True)
    report=audit(args.input,args.cutoff)
    plots=render(report,out) if args.plots else False
    report['plots_generated']=plots
    report['audit_script_sha256']=sha256(Path(__file__))
    (out/'bpi17_source_audit_v0.1.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    text=html.escape(json.dumps(report,indent=2,sort_keys=True))
    body='<img src="bpi17_audit.svg" alt="Source audit plots" style="width:100%;max-width:1200px">' if plots else '<p>Plots were not generated.</p>'
    (out/'bpi17_audit.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BPI17 source audit</title><style>body{font:16px system-ui;margin:2rem;max-width:1200px}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}</style><h1>BPI17 source audit</h1><p>Measured source data. Target semantics and splitting require review. No model performance results.</p>'+body+'<details><summary>Inspect complete audit evidence</summary><pre>'+text+'</pre></details></html>',encoding='utf-8')
    print(json.dumps({'integrity_status':report['integrity_status'],'readiness_status':report['readiness_status'],
                      'counts':report['counts'],'reference_boundary':report['reference_boundary'],
                      'labels':report['candidate_labels'],'label_vs_later_event':report['label_vs_later_event'],
                      'output':str(out)},indent=2))


if __name__=='__main__': main()
