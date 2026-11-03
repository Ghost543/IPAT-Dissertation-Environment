"""Paired corruption of reconstructed, unique source events. No fitting."""
import hashlib
import json
import numpy as np
from ipat.data.bpi17 import PackedOffers, logical_hash, NUM, CAT
from .exp06_core import require, metadata

FIELDS = ('x_obs', 'missing_mask', 'categorical', 'timestamp', 'delta', 'position')
FAMILIES = {'missingness': (.10, .20, .40), 'deletion': (.10, .25, .40)}
REPLICATES = (1, 2, 3)
MASTER_SEED = 70101


def conditions():
    return [(f, s, r) for f, levels in FAMILIES.items() for s in levels for r in REPLICATES]


def condition_id(family, severity, replicate):
    return f'{family}_{round(100*severity):02d}_r{replicate}'


def uniform(family, replicate, application, position, channel):
    # Stable key: independent of offer order, model seed, severity and Python hash salt.
    key = json.dumps(['ROB01-v0.1', MASTER_SEED, family, replicate, str(application),
                      int(position), int(channel)], separators=(',', ':'), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') / 2**64


class SourceIndex:
    """Audit that repeated prefixes really are prefixes of the same application."""
    def __init__(self, data, state):
        metadata(data)
        self.data, self.state = data, state
        a = data.arrays
        self.original_hash = logical_hash(a)
        require(state.get('numeric_fields') == list(NUM), 'Unexpected numerical field order')
        require(state.get('categorical_fields') == list(CAT), 'Unexpected categorical field order')
        self.create_id = state['vocabularies'][2].get('O_Create Offer')
        require(self.create_id is not None and self.create_id >= 3, 'Offer-creation category absent')
        offsets = a['offsets']
        require(len(offsets) == len(data)+1 and offsets[0] == 0 and (np.diff(offsets)>0).all(), 'Invalid offsets')
        require(all(len(a[k]) == offsets[-1] for k in FIELDS), 'Packed array size mismatch')
        require(a['x_obs'].shape == a['missing_mask'].shape == (offsets[-1],6), 'Numerical shape mismatch')
        require(a['categorical'].shape == (offsets[-1],7), 'Categorical shape mismatch')
        require(a['missing_mask'].dtype == bool, 'Missingness must be boolean')
        for k in ('x_obs','timestamp','delta'):
            require(np.isfinite(a[k]).all(), 'Nonfinite source: '+k)
        require((a['delta'] >= 0).all(), 'Negative source gap')
        require((a['x_obs'][a['missing_mask']] == 0).all(), 'Missing placeholders must be zero')
        self.groups = {}
        for i, app in enumerate(a['application_id']): self.groups.setdefault(str(app), []).append(i)
        self.sources = {}
        for app, rows in sorted(self.groups.items()):
            longest = max(rows, key=lambda i: offsets[i+1]-offsets[i])
            lo, hi = offsets[longest:longest+2]
            src = {k:a[k][lo:hi].copy() for k in FIELDS}
            n = hi-lo
            require(np.array_equal(src['position'], np.arange(n)), 'Source position mismatch')
            require(src['timestamp'][0] == 0 and src['delta'][0] == 0, 'First-event origin mismatch')
            require((np.diff(src['timestamp']) >= 0).all(), 'Non-monotone represented timestamps')
            require((src['missing_mask'][:,0] == src['missing_mask'][0,0]).all() and
                    (src['x_obs'][:,0] == src['x_obs'][0,0]).all(), 'Static RequestedAmount inconsistent')
            for i in rows:
                b,e = offsets[i:i+2]; size=e-b
                for k in FIELDS:
                    require(np.array_equal(a[k][b:e], src[k][:size]), f'Overlapping prefix mismatch: {app}/{k}')
                require(src['categorical'][size-1,2] == self.create_id, 'Prefix lacks offer-creation boundary')
            self.sources[app] = src

    def transform(self, family, severity, replicate):
        require(family in FAMILIES and (severity == 0 or severity in FAMILIES[family]), 'Unregistered condition')
        require(replicate in REPLICATES, 'Unregistered replicate')
        chunks = {k:[] for k in FIELDS}
        transformed = {}; source_positions = {}; stats=[]
        plan_missing=[]; plan_keep=[]; plan_offsets=[0]; app_ids=[]
        for app, src in self.sources.items():
            n=len(src['timestamp']); out={k:v.copy() for k,v in src.items()}
            keep=np.ones(n,dtype=bool); masked=np.zeros((n,6),dtype=bool)
            protected=src['categorical'][:,2] == self.create_id; protected[0]=True
            # Missingness denominator: source observations; static RequestedAmount once per app.
            observed=~src['missing_mask']; eligible_numeric=int(observed[:,1:].sum()+observed[0,0])
            eligible_events=int((~protected).sum()); selected=0
            if family == 'missingness':
                for j in range(6):
                    positions=[0] if j==0 else range(n)
                    for pos in positions:
                        if observed[pos,j] and uniform(family,replicate,app,-1 if j==0 else pos,j)<severity:
                            if j==0: masked[:,j]=True
                            else: masked[pos,j]=True
                            selected+=1
                out['missing_mask'] |= masked; out['x_obs'][out['missing_mask']]=0
            else:
                for pos in np.flatnonzero(~protected):
                    if uniform(family,replicate,app,pos,-1)<severity: keep[pos]=False
                selected=int((~keep).sum())
                if severity != 0:
                    ix=np.flatnonzero(keep)
                    out={k:v[ix].copy() for k,v in out.items()}
                    # Sum original gaps between surviving events, in float64, then cast once.
                    # This avoids cancellation from differencing rounded float32 elapsed times.
                    gaps=[0.] + [np.sum(src['delta'][left+1:right+1],dtype=np.float64)
                        for left,right in zip(ix[:-1],ix[1:])]
                    out['delta']=np.asarray(gaps,dtype=src['delta'].dtype)
                    out['position']=np.arange(len(ix),dtype=src['position'].dtype)
            source_positions[app]=np.flatnonzero(keep)
            transformed[app]=out
            eligible=eligible_numeric if family=='missingness' else eligible_events
            stats.append(dict(application_id=app,source_events=n,eligible=eligible,selected=selected,
                realised_rate=selected/eligible if eligible else None,protected_events=int(protected.sum()),
                numerical_masked_event_cells=int(masked.sum())))
            app_ids.append(app);plan_offsets.append(plan_offsets[-1]+n)
            plan_missing.append(masked);plan_keep.append(keep)
        offsets=[0];orig=self.data.arrays
        for i,app in enumerate(orig['application_id']):
            app=str(app); n=int(orig['offsets'][i+1]-orig['offsets'][i])
            size=int(np.searchsorted(source_positions[app],n))
            require(size>0 and source_positions[app][size-1]==n-1, 'Offer boundary removed')
            for k in FIELDS: chunks[k].append(transformed[app][k][:size])
            offsets.append(offsets[-1]+size)
        a={k:np.concatenate(v) for k,v in chunks.items()}
        a.update(offsets=np.array(offsets,dtype=np.int64), **{k:orig[k].copy() for k in ('y','offer_id','application_id')})
        require(logical_hash(orig)==self.original_hash, 'Clean inputs mutated')
        if severity==0: require(logical_hash(a)==self.original_hash, 'Zero corruption is not exact identity')
        eligible=sum(r['eligible'] for r in stats); selected=sum(r['selected'] for r in stats)
        audit=dict(family=family,severity=severity,replicate=replicate,eligible=eligible,selected=selected,
            realised_rate=selected/eligible if eligible else None,
            applications_without_eligible=sum(r['eligible']==0 for r in stats),by_application=stats,
            input_logical_sha256=self.original_hash,output_logical_sha256=logical_hash(a),
            original_offer_lengths=np.diff(orig['offsets']).tolist(),corrupted_offer_lengths=np.diff(a['offsets']).tolist())
        plan=dict(application_id=np.array(app_ids),offsets=np.array(plan_offsets,dtype=np.int64),
                  masked=np.concatenate(plan_missing),keep=np.concatenate(plan_keep))
        return PackedOffers(a),audit,plan
