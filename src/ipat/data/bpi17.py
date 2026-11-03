"""BPI17 v0.1 data adapter: raw input values, train-only fitting, packed prefixes.

No model fitting. IDs and labels remain metadata, never feature channels.
The pinned manifest, not the class distribution, determines admitted examples.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

NUM = ('RequestedAmount','FirstWithdrawalAmount','NumberOfTerms','MonthlyCost','CreditScore','OfferedAmount')
CAT = ('Action','org:resource','concept:name','EventOrigin','lifecycle:transition','LoanGoal','ApplicationType')
STATIC = {'RequestedAmount','LoanGoal','ApplicationType'}
SPLITS = ('train','validation','calibration','test')
CONTRACT_SHA = '72d7a993b68e01fbb49bcf13b25a99e23411ffd2db6caa9ba8ab21386a61cf9c'
MANIFEST_SHA = '4934e54868e4674331b7df34afda09d57a3ee58646d39605ac3fa414cf51f044'


def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)


def parse_time(s):
    t=datetime.fromisoformat(s.replace('Z','+00:00'))
    if t.tzinfo is None:raise ValueError('Timezone required')
    return t.astimezone(timezone.utc)


def attrs(node):return {c.get('key'):c.get('value') for c in node if c.get('key') is not None}


def iter_traces(path):
    with gzip.open(path,'rb') as f:
        context=ET.iterparse(f,events=('start','end'));_,root=next(context)
        for action,node in context:
            if action=='end' and node.tag.split('}')[-1]=='trace':
                yield attrs(node),[attrs(e) for e in node if e.tag.split('}')[-1]=='event']
                node.clear();root.clear()


@dataclass
class Application:
    application_id: str
    split: str
    offers: list[dict]
    x: np.ndarray
    missing: np.ndarray
    categories: np.ndarray
    timestamp: np.ndarray
    delta: np.ndarray


def extract_application(trace,events,rows):
    """Sanitise a complete application only through its last admitted prefix."""
    if not rows:return None
    parts={r['split'] for r in rows}
    if len(parts)!=1 or next(iter(parts)) not in SPLITS:raise ValueError('Application must have one admitted split')
    n=max(int(r['prefix_length']) for r in rows)
    if n>len(events) or n<1:raise ValueError('Invalid prefix length')
    es=events[:n]  # Later source fields cannot influence the adapter.
    times=[parse_time(e['time:timestamp']) for e in es]
    if any(a>=b for a,b in zip(times,times[1:])):raise ValueError('Non-increasing prefix timestamps')
    for r in rows:
        e=es[int(r['prefix_length'])-1]
        if e.get('concept:name')!='O_Create Offer' or e.get('EventID')!=r['offer_id']:
            raise ValueError('Offer boundary does not match source')
        if e.get('Accepted') not in ('true','false') or int(e['Accepted']=='true')!=int(r['label']):
            raise ValueError('Target does not match manifest')
        if parse_time(r['creation_time'])!=parse_time(e['time:timestamp']):raise ValueError('Time provenance mismatch')
    x=np.zeros((n,len(NUM)),dtype=np.float64);missing=np.ones(x.shape,dtype=bool)
    cat=np.empty((n,len(CAT)),dtype=object)
    for i,e in enumerate(es):
        for j,k in enumerate(NUM):
            value=(trace if k in STATIC else e).get(k)
            if value is not None and value!='':
                v=float(value)
                if not np.isfinite(v):raise ValueError('Non-finite numerical source value')
                x[i,j]=v;missing[i,j]=False
        for j,k in enumerate(CAT):
            v=(trace if k in STATIC else e).get(k)
            cat[i,j]=None if v is None or v=='' else v
    elapsed=np.array([(t-times[0]).total_seconds() for t in times],dtype=np.float64)
    gaps=np.array([0.]+[(b-a).total_seconds() for a,b in zip(times,times[1:])],dtype=np.float64)
    return Application(trace['concept:name'],next(iter(parts)),rows,x,missing,cat,elapsed,gaps)


def load_source(source,manifest,contract_path):
    if sha256(contract_path)!=CONTRACT_SHA:raise ValueError('Contract changed: version separately before use')
    contract=json.loads(Path(contract_path).read_text())
    if sha256(source)!=contract['source_sha256']:raise ValueError('Source hash mismatch')
    if sha256(manifest)!=MANIFEST_SHA:raise ValueError('Split manifest hash mismatch')
    with Path(manifest).open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    byapp={}
    for r in rows:byapp.setdefault(r['application_id'],[]).append(r)
    result=[];seen=set();events_total=0;offers_total=0
    for trace,events in iter_traces(source):
        aid=trace.get('concept:name')
        if not aid or aid in seen:raise ValueError('Missing/duplicate application ID')
        seen.add(aid);events_total+=len(events)
        offers_total+=sum(e.get('concept:name')=='O_Create Offer' for e in events)
        if aid not in byapp:raise ValueError('Source application absent from manifest')
        rr=byapp[aid]
        if {r['split'] for r in rr}=={'excluded'}:continue
        result.append(extract_application(trace,events,rr))
    actual=dict(applications=len(seen),events=events_total,offers=offers_total)
    if actual!=contract['expected_source_counts'] or seen!=set(byapp):raise ValueError('Source/manifest count mismatch')
    return result


def moments(vals,floor):
    a=np.asarray(vals,dtype=np.float64)
    if a.size==0 or not np.isfinite(a).all():raise ValueError('Need finite observed training values')
    std=float(a.std(ddof=0));constant=std==0.
    return dict(count=int(a.size),mean=float(a.mean()),std=std,
                scale=1. if constant else max(std,floor),constant=constant)


def fit_preprocessing(apps,scale_floor=1e-8):
    """Only training applications; arrays hold each covered source event once."""
    if not np.isfinite(scale_floor) or scale_floor<=0:raise ValueError('Positive finite scale floor required')
    train=[a for a in apps if a.split=='train']
    if not train:raise ValueError('No training applications')
    if len({a.application_id for a in apps})!=len(apps):raise ValueError('Duplicate application object')
    numeric=[]
    for j,k in enumerate(NUM):
        vals=[]
        for a in train:
            x=a.x[:1,j] if k in STATIC else a.x[:,j]
            mask=a.missing[:1,j] if k in STATIC else a.missing[:,j]
            vals.extend(x[~mask])
        numeric.append(dict(name=k,**moments(vals,scale_floor)))
    vocabs=[]
    for j,k in enumerate(CAT):
        observed={v for a in train for v in a.categories[:,j] if v is not None}
        vocabs.append({v:i+3 for i,v in enumerate(sorted(observed))})
    return dict(version='IPAT-BPI17-adapter-v0.1',numeric=numeric,
        temporal={k:moments(np.concatenate([getattr(a,k) for a in train]),scale_floor) for k in ('timestamp','delta')},
        vocabularies=vocabs,categorical_fields=list(CAT),numeric_fields=list(NUM),
        train_applications=len(train),train_unique_events=sum(len(a.x) for a in train),
        scale_floor=scale_floor,raw_numeric_inputs=True,
        temporal_convention='elapsed seconds; preceding-event gap seconds; zero-based position; existing TemporalEncoder standardises and divides position by max(length-1,1)',
        fit_weighting='unique training source events; static numerical fields once per application')


def encode_application(a,state):
    ids=np.zeros(a.categories.shape,dtype=np.int64)
    for j,vocab in enumerate(state['vocabularies']):
        ids[:,j]=[1 if v is None else vocab.get(v,2) for v in a.categories[:,j]]
    raw=a.x.astype(np.float32)
    time=a.timestamp.astype(np.float32);delta=a.delta.astype(np.float32)
    if not np.isfinite(raw).all() or not np.isfinite(time).all() or not np.isfinite(delta).all():raise ValueError('Float32 overflow')
    return dict(x_obs=raw,missing_mask=a.missing.copy(),categorical=ids,
                timestamp=time,delta=delta,position=np.arange(len(raw),dtype=np.int64))


def logical_hash(arrays):
    h=hashlib.sha256()
    for k in sorted(arrays):
        a=np.ascontiguousarray(arrays[k]);h.update(canonical([k,a.dtype.str,list(a.shape)]).encode());h.update(a.tobytes())
    return h.hexdigest()


def pack_split(apps,state,split):
    chunks={k:[] for k in ('x_obs','missing_mask','categorical','timestamp','delta','position')}
    offsets=[0];labels=[];offer_ids=[];app_ids=[]
    for a in apps:
        if a.split!=split:continue
        encoded=encode_application(a,state)
        for r in a.offers:
            n=int(r['prefix_length'])
            for k in chunks:chunks[k].append(encoded[k][:n])
            offsets.append(offsets[-1]+n);labels.append(int(r['label']))
            offer_ids.append(r['offer_id']);app_ids.append(a.application_id)
    if not labels:raise ValueError('Empty admitted split')
    arrays={k:np.concatenate(v,axis=0) for k,v in chunks.items()}
    arrays.update(offsets=np.asarray(offsets,dtype=np.int64),y=np.asarray(labels,dtype=np.float32),
                  offer_id=np.asarray(offer_ids,dtype='U80'),application_id=np.asarray(app_ids,dtype='U80'))
    if any(len(s)>80 for s in offer_ids+app_ids):raise ValueError('Identifier storage width exceeded')
    return arrays


class PackedOffers:
    """Indexable dataset; metadata and model inputs are returned separately."""
    def __init__(self,arrays):self.arrays=arrays
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:return cls({k:f[k] for k in f.files})
    def __len__(self):return len(self.arrays['y'])
    def __getitem__(self,i):
        if not 0<=i<len(self):raise IndexError(i)
        a=self.arrays;start,end=a['offsets'][i:i+2]
        inputs={k:a[k][start:end].copy() for k in ('x_obs','missing_mask','categorical','timestamp','delta','position')}
        return inputs,float(a['y'][i]),{'offer_id':str(a['offer_id'][i]),'application_id':str(a['application_id'][i])}


def collate_numpy(examples,extra_padding=0):
    if not examples or extra_padding<0:raise ValueError('Nonempty batch and nonnegative extra padding required')
    sizes=[len(x[0]['timestamp']) for x in examples]
    if min(sizes)<1:raise ValueError('Empty sequence')
    b=len(examples);n=max(sizes)+extra_padding
    inputs={}
    for k in examples[0][0]:
        v=examples[0][0][k];fill=True if k=='missing_mask' else -1 if k=='position' else 0
        inputs[k]=np.full((b,n)+v.shape[1:],fill,dtype=v.dtype)
    inputs['valid_mask']=np.zeros((b,n),dtype=bool)
    for i,(x,_,_) in enumerate(examples):
        for k,v in x.items():inputs[k][i,:sizes[i]]=v
        inputs['valid_mask'][i,:sizes[i]]=True
    return inputs,np.asarray([x[1] for x in examples],dtype=np.float32),[x[2] for x in examples]


def collate_bpi17(examples):
    import torch
    inputs,y,metadata=collate_numpy(examples)
    return {k:torch.from_numpy(v) for k,v in inputs.items()},torch.from_numpy(y),metadata


def save_cache(apps,state,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    (directory/'preprocessing.json').write_text(canonical(state)+'\n',encoding='utf-8')
    reports={}
    for s in SPLITS:
        arrays=pack_split(apps,state,s);path=directory/f'{s}.npz'
        np.savez_compressed(path,**arrays)
        missing=arrays['missing_mask'];cats=arrays['categorical']
        reports[s]=dict(offers=len(arrays['y']),events=int(arrays['offsets'][-1]),
            applications=len(set(arrays['application_id'])),labels={str(k):int((arrays['y']==k).sum()) for k in (0,1)},
            logical_sha256=logical_hash(arrays),file_sha256=sha256(path),
            numeric_missing_fraction=missing.mean(axis=0).tolist(),
            categorical_unknown_fraction=(cats==2).mean(axis=0).tolist(),
            max_length=int(np.diff(arrays['offsets']).max()))
    return reports
