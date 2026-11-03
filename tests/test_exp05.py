import json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ipat.data.bpi17 import PackedOffers,logical_hash
from ipat.experiments.exp05 import *
from ipat.experiments.exp05_metrics import *
from ipat.experiments.exp05_references import prefix_summaries,tree


def fixture(n=9,seed=8):
    rng=np.random.default_rng(seed);lengths=np.array([1+i%5 for i in range(n)]);size=int(lengths.sum());offsets=np.r_[0,np.cumsum(lengths)]
    raw=rng.normal(size=(size,6)).astype('float32');mask=rng.random((size,6))<.15;raw[mask]=0
    a=dict(x_obs=raw,missing_mask=mask,categorical=rng.integers(1,4,(size,7),dtype=np.int64),timestamp=np.concatenate([np.arange(k,dtype='float32') for k in lengths]),
        delta=np.concatenate([np.r_[0,np.ones(k-1)].astype('float32') for k in lengths]),position=np.concatenate([np.arange(k,dtype=np.int64) for k in lengths]),
        offsets=offsets,y=(np.arange(n)%2).astype('float32'),offer_id=np.array([f'o{seed}_{i}' for i in range(n)]),application_id=np.array([f'a{seed}_{i//2}' for i in range(n)]))
    state=dict(numeric=[dict(mean=0.,scale=1.) for _ in range(6)],temporal={k:dict(mean=0.,scale=1.) for k in ('timestamp','delta')},vocabularies=[{'a':3} for _ in range(7)])
    c=dict(epochs=2,batch_size=4,microbatch=2,checkpoint_every=1,learning_rate=.001,weight_decay=.01,clip_norm=1.,model=dict(n_layers=2,n_heads=4,dropout=.1,identity_floor=.25,asinh_scales=[.25,1.,4.],gamma=1.))
    return state,PackedOffers(a),c


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_nll_brier_threshold_hand_computed(self):
        m=metrics([0,1],[0,0]);self.assertAlmostEqual(m['nll'],math.log(2));self.assertEqual(m['brier'],.25);self.assertEqual(m['confusion_matrix'],[[0,1],[0,1]])
    def test_extreme_logits_stable_and_invalid_rejected(self):
        self.assertTrue(np.isfinite(losses([1,0],[10000,-10000])).all())
        with self.assertRaises(ValueError):losses([1],[float('nan')])
    def test_undefined_one_class_explicit(self):
        m=metrics([1,1],[1,2]);self.assertIsNone(m['auroc']);self.assertIsNone(m['balanced_accuracy']);self.assertIsNone(m['classes']['0']['recall'])
    def test_reliability_endpoints(self):
        e,b=reliability(np.array([1,0,0]),np.array([1.,.5,0.]));self.assertAlmostEqual(e,1/6);self.assertEqual(sum(x['count'] for x in b),3);self.assertEqual(b[-1]['count'],2)
    def test_application_macro_differs_offer_mean(self):
        m=metrics([0,0,1],[0,0,2],['a','a','b']);self.assertNotEqual(m['nll'],m['application_macro_nll'])
    def test_cluster_pairing_zero_and_determinism(self):
        y=np.array([0,1,0]);a=np.array([[0.,1.,2.],[0.,2.,1.]]);apps=np.array(['a','a','b'])
        r=paired_cluster_interval(y,a,a,apps,draws=30);self.assertEqual(r['percentile_95'],[0.,0.]);self.assertEqual(r,paired_cluster_interval(y,a,a,apps,draws=30))
    def test_empty_length_stratum(self):
        r=stratified(np.array([0,1]),np.array([0.,0.]),np.array(['a','b']),np.array([1,2]));self.assertEqual(r['41+']['n'],0);self.assertIsNone(r['41+']['nll'])
    def test_heldout_loading_rejected_before_io(self):
        with patch('ipat.experiments.exp05.PackedOffers.load') as f:
            for split in ('calibration','test'):
                with self.assertRaises(ValueError):load_split('irrelevant',split,'x')
            f.assert_not_called()
    def test_cache_hash_rejects_modification(self):
        s,d,c=fixture()
        with tempfile.TemporaryDirectory() as t:
            np.savez_compressed(Path(t)/'train.npz',**d.arrays)
            self.assertEqual(len(load_split(t,'train',logical_hash(d.arrays))),len(d))
            with self.assertRaises(ValueError):load_split(t,'train','invalid')
    def test_preflight_load_never_requests_validation(self):
        s,d,c=fixture();c.update(preprocessing_logical_sha256=digest(s),logical_hashes={'train':'t','validation':'v'})
        with tempfile.TemporaryDirectory() as t:
            write_json(Path(t)/'preprocessing.json',s)
            with patch('ipat.experiments.exp05.load_split',return_value=d) as f:load_inputs(t,c,False);self.assertEqual(f.call_args_list[0].args[1],'train');self.assertEqual(f.call_count,1)
    def test_summary_missingness_and_metadata_exclusion(self):
        s,d,c=fixture();d.arrays['missing_mask'][:,0]=True;a=prefix_summaries(d,s);self.assertTrue(np.isnan(a[:,0:5]).all());self.assertTrue((a[:,5]==0).all())
        d.arrays['offer_id'][:]='secret';b=prefix_summaries(d,s);np.testing.assert_equal(a,b)
    def test_tree_budget_and_no_internal_split(self):
        m=tree(1001);self.assertFalse(m.early_stopping);self.assertEqual(m.max_iter,200);self.assertIsNone(m.class_weight)
    def test_paired_initialisation(self):
        s,d,c=fixture();h=assert_shared(s,1001,c['model']);self.assertEqual(len(set(h.values())),1)
    def test_offer_order_covers_every_offer_once(self):
        p=permutation(11,1001,0);self.assertEqual(sorted(p.tolist()),list(range(11)));np.testing.assert_equal(p,permutation(11,1001,0));self.assertFalse(np.array_equal(p,permutation(11,1001,1)))
    def test_gru_padding_invariance(self):
        from ipat.data.bpi17 import collate_numpy
        s,d,c=fixture();m=make_model('GRU',s,8,c['model']).eval();xx=[d[0],d[1]]
        def call(extra):
            x,_,_=collate_numpy(xx,extra);return m({k:torch.from_numpy(v) for k,v in x.items()})['logits']
        torch.testing.assert_close(call(0),call(5),rtol=0,atol=1e-6)
    def test_prediction_does_not_advance_rng(self):
        s,d,c=fixture();m=make_model('T11',s,8,c['model']);before=torch.get_rng_state().clone();predict(m,d,'cpu',2,True);self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(m.training)
    def test_source_review_cannot_default_to_verified(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'r.json';write_json(p,{'status':'NOT_VERIFIED'})
            with self.assertRaises(ValueError):provenance_review(p,{'source_sha256':'x'})
    def test_resume_matches_uninterrupted_parameters_and_predictions(self):
        s,d,c=fixture();_,v,_=fixture(5,seed=7)
        with tempfile.TemporaryDirectory() as t:
            a=Path(t)/'a';b=Path(t)/'b';ra=train_run('T11',19,s,d,v,c,a,'fixture');train_run('T11',19,s,d,v,c,b,'fixture',stop_after=2)
            with self.assertRaises(ValueError):train_run('T11',19,s,d,v,c,b,'changed',resume=True)
            rb=train_run('T11',19,s,d,v,c,b,'fixture',resume=True)
            self.assertEqual(ra['selected'],rb['selected']);self.assertEqual(ra['updates'],6)
            aa=load_checkpoint(a/'final.pt')['model'];bb=load_checkpoint(b/'final.pt')['model']
            for k in aa:self.assertTrue(torch.equal(aa[k],bb[k]),k)
    def test_identity_gamma_zero_forward_and_shared_gradients(self):
        s,d,c=fixture();settings={**c['model'],'asinh_scales':[],'gamma':0.,'dropout':0.}
        a=make_model('T00',s,12,settings);b=make_model('T11',s,12,settings);x,y=batch(d,[0,1,2],'cpu')
        aa=a(x)['logits'];bb=b(x)['logits'];torch.testing.assert_close(aa,bb,rtol=0,atol=0)
        aa.sum().backward();bb.sum().backward();other=dict(b.named_parameters())
        for k,p in a.named_parameters():
            if p.grad is not None:torch.testing.assert_close(p.grad,other[k].grad,rtol=0,atol=0)
    def test_fractional_labels_rejected(self):
        with self.assertRaises(ValueError):metrics([0.5,1],[0,0])
    def test_selected_checkpoint_tie_keeps_first_epoch(self):
        s,d,c=fixture();_,v,_=fixture(5,7);c['learning_rate']=0.
        with tempfile.TemporaryDirectory() as t:
            r=train_run('T00',22,s,d,v,c,t,'fixture');self.assertEqual(r['selected_epoch'],1)
    def test_microbatch_weighting_with_unequal_last_chunk(self):
        s,d,c=fixture(5);settings={**c['model'],'dropout':0.};m=make_model('T00',s,1,settings);n=make_model('T00',s,1,settings)
        x,y=batch(d,range(5),'cpu');F.binary_cross_entropy_with_logits(m(x)['logits'],y).backward()
        for ii in ([0,1],[2,3],[4]):
            x,y=batch(d,ii,'cpu');(F.binary_cross_entropy_with_logits(n(x)['logits'],y,reduction='sum')/5).backward()
        for a,b in zip(m.parameters(),n.parameters()):
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-6)

if __name__=='__main__':unittest.main()
