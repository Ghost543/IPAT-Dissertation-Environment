"""Synthetic source-consistency, corruption and metric tests; no real data."""
import ast
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ipat.data.bpi17 import PackedOffers,NUM,CAT,logical_hash
from ipat.experiments.rob01_corruptions import *
from ipat.experiments.rob01_core import hierarchical,save_record,load_record
from ipat.experiments.exp06_core import cluster_interval,metadata


def fixture(apps=8,seed=801,short=False):
    rng=np.random.default_rng(seed);chunks={k:[] for k in FIELDS};offsets=[0];ys=[];offers=[];ids=[]
    for i in range(apps):
        n=2 if short else (6,15,25,45)[i%4]
        x=rng.uniform(0,5,(n,6)).astype('float32');m=rng.random((n,6))<.15
        x[:,0]=i+1;m[:,0]=False;x[0,1]=0;m[0,1]=False;x[m]=0
        cat=np.full((n,7),3,dtype=np.int64);cat[-1,2]=4
        if not short:cat[2,2]=4
        gaps=np.r_[0,np.arange(1,n)+1].astype('float32')
        src=dict(x_obs=x,missing_mask=m,categorical=cat,delta=gaps,timestamp=np.cumsum(gaps),position=np.arange(n,dtype=np.int64))
        for j,size in enumerate([n] if short else [3,n]):
            for k in FIELDS:chunks[k].append(src[k][:size].copy())
            offsets.append(offsets[-1]+size);ys.append((i+j)%2);offers.append(f'o{seed}_{i}_{j}');ids.append(f'a{seed}_{i}')
    a={k:np.concatenate(v) for k,v in chunks.items()};a.update(offsets=np.array(offsets,dtype=np.int64),y=np.array(ys,dtype='float32'),
        offer_id=np.array(offers),application_id=np.array(ids))
    vocabs=[{'a':3} for _ in CAT];vocabs[2]['O_Create Offer']=4
    state=dict(numeric_fields=list(NUM),categorical_fields=list(CAT),numeric=[dict(mean=0.,scale=1.) for _ in NUM],
        temporal={k:dict(mean=0.,scale=1.) for k in ('timestamp','delta')},vocabularies=vocabs)
    config=dict(microbatch=8,model=dict(n_layers=2,n_heads=4,dropout=.1,identity_floor=.25,asinh_scales=[.25,1.,4.],gamma=1.))
    return state,PackedOffers(a),config


def reorder(data,order):
    a=data.arrays;ch={k:[] for k in FIELDS};offsets=[0]
    for i in order:
        b,e=a['offsets'][i:i+2]
        for k in FIELDS:ch[k].append(a[k][b:e])
        offsets.append(offsets[-1]+e-b)
    ar={k:np.concatenate(v) for k,v in ch.items()};ar.update(offsets=np.array(offsets),**{k:a[k][order] for k in ('y','offer_id','application_id')})
    return PackedOffers(ar)


class Rob01Tests(unittest.TestCase):
    def setUp(self):self.state,self.data,self.config=fixture();self.index=SourceIndex(self.data,self.state)
    def test_grid_exactly_18_conditions(self):self.assertEqual(len(set(conditions())),18)
    def test_zero_is_exact_for_both_families_all_replicates(self):
        for f in FAMILIES:
            for r in REPLICATES:
                out,a,p=self.index.transform(f,0,r);self.assertEqual(logical_hash(out.arrays),logical_hash(self.data.arrays))
    def test_clean_not_mutated(self):
        before=logical_hash(self.data.arrays)
        for f,s,r in conditions():self.index.transform(f,s,r)
        self.assertEqual(before,logical_hash(self.data.arrays))
    def test_reproducible_arrays_and_plans(self):
        for f in FAMILIES:
            a,_,p=self.index.transform(f,.4,1);b,_,q=self.index.transform(f,.4,1)
            self.assertEqual(logical_hash(a.arrays),logical_hash(b.arrays));self.assertEqual(logical_hash(p),logical_hash(q))
    def test_nested_missingness(self):
        _,_,a=self.index.transform('missingness',.1,1);_,_,b=self.index.transform('missingness',.2,1);_,_,c=self.index.transform('missingness',.4,1)
        self.assertTrue((~a['masked']|b['masked']).all());self.assertTrue((~b['masked']|c['masked']).all())
    def test_nested_deletion(self):
        _,_,a=self.index.transform('deletion',.1,1);_,_,b=self.index.transform('deletion',.25,1);_,_,c=self.index.transform('deletion',.4,1)
        self.assertTrue((a['keep']|~b['keep']).all());self.assertTrue((b['keep']|~c['keep']).all())
    def test_replicates_differ(self):
        _,_,a=self.index.transform('missingness',.4,1);_,_,b=self.index.transform('missingness',.4,2)
        self.assertFalse(np.array_equal(a['masked'],b['masked']))
    def test_source_overlap_inconsistency_rejected(self):
        a=copy.deepcopy(self.data.arrays);a['x_obs'][0,2]+=1
        with self.assertRaises(ValueError):SourceIndex(PackedOffers(a),self.state)
    def test_static_inconsistency_rejected(self):
        a=copy.deepcopy(self.data.arrays);a['x_obs'][3,0]+=1
        with self.assertRaises(ValueError):SourceIndex(PackedOffers(a),self.state)
    def test_field_order_rejected(self):
        s=copy.deepcopy(self.state);s['numeric_fields']=list(reversed(NUM))
        with self.assertRaises(ValueError):SourceIndex(self.data,s)
    def test_missing_offer_category_rejected(self):
        s=copy.deepcopy(self.state);del s['vocabularies'][2]['O_Create Offer']
        with self.assertRaises(ValueError):SourceIndex(self.data,s)
    def test_nonfinite_rejected(self):
        a=copy.deepcopy(self.data.arrays);a['x_obs'][0,2]=np.nan
        with self.assertRaises(ValueError):SourceIndex(PackedOffers(a),self.state)
    def test_unregistered_conditions_rejected(self):
        for args in [('noise',.1,1),('deletion',.2,1),('missingness',.1,4)]:
            with self.assertRaises(ValueError):self.index.transform(*args)
    def test_original_labels_and_ids_retained(self):
        for f in FAMILIES:
            out,_,_=self.index.transform(f,.4,1)
            for k in ('y','offer_id','application_id'):np.testing.assert_array_equal(out.arrays[k],self.data.arrays[k])
    def test_requested_amount_broadcast_and_unique_denominator(self):
        out,a,p=self.index.transform('missingness',.4,1)
        total=0
        for app,src in self.index.sources.items():total+=int((~src['missing_mask'][:,1:]).sum()+1)
        self.assertEqual(a['eligible'],total)
        SourceIndex(out,self.state)  # Also validates prefix/static consistency after masking.
    def test_original_missingness_retained_and_finite_zero_placeholder(self):
        out,_,_=self.index.transform('missingness',.4,1);a=out.arrays
        self.assertTrue((a['missing_mask']|~self.data.arrays['missing_mask']).all())
        self.assertTrue((a['x_obs'][a['missing_mask']]==0).all())
        obs=~a['missing_mask'];np.testing.assert_array_equal(a['x_obs'][obs],self.data.arrays['x_obs'][obs])
    def test_missingness_leaves_other_channels_unchanged(self):
        out,_,_=self.index.transform('missingness',.4,1)
        for k in ('categorical','timestamp','delta','position','offsets'):np.testing.assert_array_equal(out.arrays[k],self.data.arrays[k])
    def test_protected_events_never_deleted(self):
        _,_,p=self.index.transform('deletion',.4,1)
        for i,src in enumerate(self.index.sources.values()):
            keep=p['keep'][p['offsets'][i]:p['offsets'][i+1]]
            self.assertTrue(keep[0]);self.assertTrue(keep[src['categorical'][:,2]==4].all())
    def test_deletion_temporal_reconstruction_and_original_values(self):
        out,_,p=self.index.transform('deletion',.4,1)
        a=out.arrays
        for i,app in enumerate(a['application_id']):
            src=self.index.sources[str(app)];n=int(self.data.arrays['offsets'][i+1]-self.data.arrays['offsets'][i])
            j=list(self.index.sources).index(str(app));keep=p['keep'][p['offsets'][j]:p['offsets'][j+1]][:n]
            b,e=a['offsets'][i:i+2]
            np.testing.assert_array_equal(a['timestamp'][b:e],src['timestamp'][:n][keep])
            np.testing.assert_array_equal(a['position'][b:e],np.arange(keep.sum()))
            np.testing.assert_array_equal(a['delta'][b:e],np.r_[0,np.diff(np.cumsum(src['delta'].astype(float))[:n][keep])])
            np.testing.assert_array_equal(a['x_obs'][b:e],src['x_obs'][:n][keep])
    def test_offer_order_does_not_change_source_plan(self):
        data=reorder(self.data,np.arange(len(self.data))[::-1]);other=SourceIndex(data,self.state)
        for f in FAMILIES:
            _,_,p=self.index.transform(f,.4,1);_,_,q=other.transform(f,.4,1)
            self.assertEqual(logical_hash(p),logical_hash(q))
    def test_no_eligible_deletion_retains_cases_and_reports_null(self):
        s,d,_=fixture(short=True);out,a,p=SourceIndex(d,s).transform('deletion',.4,1)
        self.assertEqual(a['eligible'],0);self.assertIsNone(a['realised_rate']);self.assertEqual(len(out),len(d))
    def test_original_length_strata_after_deletion(self):
        out,_,_=self.index.transform('deletion',.4,1)
        with tempfile.TemporaryDirectory() as d:
            row=dict(arm='T00',seed=1001,temperature=1.,checkpoint_sha256='fixture')
            r=save_record(Path(d)/'record',out,np.zeros(len(out)),row,metadata(self.data)['lengths'])
            rr,a=load_record(Path(d)/'record')
            np.testing.assert_array_equal(a['lengths'],metadata(self.data)['lengths'])
            self.assertEqual(r['raw_length']['41+']['n'],2)
    def test_tampered_saved_metrics_rejected(self):
        from ipat.experiments.exp06_core import write_json
        with tempfile.TemporaryDirectory() as d:
            r=save_record(Path(d)/'r',self.data,np.zeros(len(self.data)),dict(temperature=1.),metadata(self.data)['lengths'])
            r['raw']['nll']=99;write_json(Path(d)/'r/result.json',r)
            with self.assertRaises(ValueError):load_record(Path(d)/'r')
    def test_hierarchical_equal_family_weight(self):
        with patch('ipat.experiments.rob01_core.FAMILIES',{'a':(.1,.2),'b':(.1,)}):
            vals={(f,s,r):v for f,ss,v in [('a',(.1,.2),2),('b',(.1,),8)] for s in ss for r in REPLICATES}
            self.assertEqual(hierarchical(vals),5)
    def test_positive_advantage_means_less_deterioration(self):
        t00=np.array([.5,.6])-np.array([.8,.9]);t11=np.array([.5,.6])-np.array([.6,.7])
        r=cluster_interval(np.stack([t11-t00]*3),np.array([0,1]),np.array([[0,1],[1,0]]))
        self.assertAlmostEqual(r['mean_delta'],.2)
    def test_no_fitting_calls_in_real_modules(self):
        root=Path(__file__).resolve().parents[1]/'src/ipat/experiments'
        forbidden={'fit','fit_temperature','train_run','AdamW','backward','step','fit_preprocessing'}
        for name in ('rob01.py','rob01_core.py','rob01_corruptions.py','rob01_plots.py'):
            for node in ast.walk(ast.parse((root/name).read_text())):
                if isinstance(node,ast.Call):
                    fun=node.func;called=fun.attr if isinstance(fun,ast.Attribute) else fun.id if isinstance(fun,ast.Name) else ''
                    self.assertNotIn(called,forbidden)

if __name__=='__main__':unittest.main()
