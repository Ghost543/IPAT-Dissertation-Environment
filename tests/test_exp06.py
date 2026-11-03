import ast
import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ipat.experiments.exp06_core import *


def fixture(n=24, seed=800):
    """Only synthetic arrays; IDs are disjoint between fixture seeds."""
    from ipat.data.bpi17 import PackedOffers
    rng=np.random.default_rng(seed)
    lengths=np.array([6,10,15,25,45]*(n//5+1))[:n];size=int(lengths.sum())
    raw=rng.normal(size=(size,6)).astype('float32');missing=rng.random((size,6))<.1;raw[missing]=0
    arrays=dict(x_obs=raw,missing_mask=missing,categorical=rng.integers(1,4,(size,7),dtype=np.int64),
        timestamp=np.concatenate([np.arange(k,dtype='float32') for k in lengths]),
        delta=np.concatenate([np.r_[0,np.ones(k-1)].astype('float32') for k in lengths]),
        position=np.concatenate([np.arange(k,dtype=np.int64) for k in lengths]),offsets=np.r_[0,np.cumsum(lengths)],
        y=(np.arange(n)%2).astype('float32'),offer_id=np.array([f'o{seed}_{i}' for i in range(n)]),
        application_id=np.array([f'a{seed}_{i//2}' for i in range(n)]))
    state=dict(numeric=[dict(mean=0.,scale=1.) for _ in range(6)],
        temporal={k:dict(mean=0.,scale=1.) for k in ('timestamp','delta')},vocabularies=[{'a':3} for _ in range(7)])
    config=dict(microbatch=8,model=dict(n_layers=2,n_heads=4,dropout=.1,identity_floor=.25,asinh_scales=[.25,1.,4.],gamma=1.))
    return state,PackedOffers(arrays),config


class Exp06Tests(unittest.TestCase):
    def setUp(self):
        self.meta=dict(y=np.array([0,1,0,1]),offer_id=np.array(['o1','o2','o3','o4']),
            application_id=np.array(['a','a','b','c']),lengths=np.array([6,15,25,45]))

    def test_hand_computed_scores_and_tie(self):
        a=prediction_record(self.meta,np.zeros(4),1.2);r=evaluate_record(a)
        self.assertAlmostEqual(r['raw']['nll'],math.log(2));self.assertEqual(r['raw']['brier'],.25)
        self.assertEqual(r['raw']['confusion_matrix'],[[0,2],[0,2]])

    def test_explicit_scaling_formula(self):
        a=prediction_record(self.meta,np.full(4,math.log(9)),2.)
        np.testing.assert_allclose(a['scaled_probability'],.75,rtol=0,atol=1e-15)

    def test_no_requirement_for_test_nll_improvement(self):
        x=np.array([-2,2,-2,2]);r=evaluate_record(prediction_record(self.meta,x,2.))
        self.assertGreater(r['scaled']['nll'],r['raw']['nll'])

    def test_class_metrics_unchanged(self):
        r=evaluate_record(prediction_record(self.meta,[-1,2,3,-4],.8))
        for key in ('confusion_matrix','classes','accuracy'):self.assertEqual(r['raw'][key],r['scaled'][key])

    def test_nonfinite_rejected(self):
        for x in ([1,2,3,np.nan],[1,2,3,np.inf]):
            with self.assertRaises(ValueError):prediction_record(self.meta,x,1.)

    def test_bad_temperature_rejected(self):
        for t in (0,-1,np.inf,np.nan):
            with self.assertRaises(ValueError):prediction_record(self.meta,[0,1,2,3],t)

    def test_sigmoid_rounding_detected(self):
        with self.assertRaises(ValueError):prediction_record(self.meta,[-1e-20,1,2,3],1.)

    def test_duplicate_offers_rejected(self):
        m=copy.deepcopy(self.meta);m['offer_id'][1]=m['offer_id'][0]
        with self.assertRaises(ValueError):validate_metadata(m)

    def test_labels_must_be_binary(self):
        m=copy.deepcopy(self.meta);m['y'][0]=2
        with self.assertRaises(ValueError):validate_metadata(m)

    def test_alignment_checks_ids_labels_lengths(self):
        for key in META:
            b=copy.deepcopy(self.meta);b[key]=b[key][::-1]
            with self.assertRaises(ValueError):aligned(self.meta,b)

    def test_cross_split_applications_rejected(self):
        m=copy.deepcopy(self.meta);m['offer_id']=np.array(['p1','p2','p3','p4'])
        with self.assertRaises(ValueError):disjoint(dict(train=self.meta,test=m))

    def test_exact_bootstrap_multiplicities(self):
        inverse=np.array([0,0,1]);indices=np.array([[0,0],[0,1],[1,1]])
        delta=np.array([[1.,3.,10.],[1.,3.,10.]])
        r=cluster_interval(delta,inverse,indices)
        self.assertAlmostEqual(r['mean_delta'],14/3)
        np.testing.assert_allclose(r['percentile_95'],np.quantile([2,14/3,10],[.025,.975]))

    def test_paired_equal_predictions_zero_interval(self):
        _,inv,ix=cluster_draws(self.meta['application_id'],30)
        r=cluster_interval(np.zeros((3,4)),inv,ix);self.assertEqual(r['percentile_95'],[0.,0.])

    def test_bootstrap_reproducible_sorted_apps(self):
        a,inv,x=cluster_draws(self.meta['application_id'],30);b,j,y=cluster_draws(self.meta['application_id'],30)
        np.testing.assert_equal(x,y);self.assertEqual(a.tolist(),['a','b','c'])

    def test_macro_and_offer_weighting_differ(self):
        r=evaluate_record(prediction_record(self.meta,[0,0,-10,10],1.))
        self.assertNotEqual(r['raw']['nll'],r['raw']['application_macro_nll'])

    def test_absent_class_explicit_null(self):
        m=copy.deepcopy(self.meta);m['y'][:]=1
        r=evaluate_record(prediction_record(m,[1,2,3,4],1.))
        self.assertIsNone(r['raw']['classes']['0']['recall']);self.assertIsNone(r['raw']['auroc'])

    def test_empty_stratum_explicit(self):
        m=copy.deepcopy(self.meta);m['lengths'][:]=6
        r=evaluate_record(prediction_record(m,[1,2,3,4],1.))
        self.assertEqual(r['raw_length']['41+']['n'],0);self.assertIsNone(r['raw_length']['41+']['nll'])

    def test_replay_threshold_and_changed_decisions(self):
        self.assertLessEqual(replay([1+1e-7],[1.],'fixture'),1e-6)
        with self.assertRaises(ValueError):replay([1.1],[1.],'fixture')
        with self.assertRaises(ValueError):replay([1e-7],[-1e-7],'fixture')

    def test_cache_hash_and_forbidden_training(self):
        from ipat.data.bpi17 import logical_hash
        _,data,_=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            save_npz(Path(tmp)/'test.npz',**data.arrays)
            self.assertEqual(len(load_partition(tmp,'test',logical_hash(data.arrays))),len(data))
            with self.assertRaises(ValueError):load_partition(tmp,'test','wrong')
            with self.assertRaises(ValueError):load_partition('/missing','train','wrong')

    def test_wrong_temperature_or_checkpoint_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint';p.write_bytes(b'fixture')
            r=dict(fit=dict(temperature=1.,status='INTERIOR'),checks=dict(checkpoint_sha256=sha(p)))
            self.assertEqual(validate_temperature(r,r,p),1.)
            other=copy.deepcopy(r);other['fit']['temperature']=2.
            with self.assertRaises(ValueError):validate_temperature(other,r,p)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_temperature(r,r,p)

    def test_saved_probability_mismatch_rejected(self):
        a=prediction_record(self.meta,[1,2,3,4],1.);a['raw_probability'][0]=.1
        with self.assertRaises(ValueError):evaluate_record(a)

    def test_numerical_probability_ties_disclosed(self):
        a=prediction_record(self.meta,[-1000,1000,-900,900],1.)
        r=evaluate_record(a)
        self.assertLess(r['ranking_diagnostics']['unique_raw_probabilities'],r['ranking_diagnostics']['unique_raw_logits'])

    def test_no_fitting_calls_in_evaluator(self):
        directory=Path(__file__).resolve().parents[1]/'src/ipat/experiments'
        forbidden={'fit','fit_temperature','train_run','optimiser','AdamW','backward','step','fit_preprocessing'}
        for name in ('exp06.py','exp06_core.py'):
            tree=ast.parse((directory/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    fn=node.func;called=fn.attr if isinstance(fn,ast.Attribute) else fn.id if isinstance(fn,ast.Name) else ''
                    self.assertNotIn(called,forbidden)

    def test_reliability_empty_bins_and_endpoint(self):
        from ipat.experiments.exp05_metrics import reliability
        value,bins=reliability(np.array([1,0,0]),np.array([1.,.5,0.]))
        self.assertAlmostEqual(value,1/6);self.assertEqual(sum(r['count'] for r in bins),3)


if __name__=='__main__':unittest.main()
