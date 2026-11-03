import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from ipat.data.bpi17 import (NUM,CAT,extract_application,fit_preprocessing,encode_application,
    pack_split,PackedOffers,collate_numpy,logical_hash,save_cache,moments)


def fixture(aid='a',split='train',amount=10.,unknown=False):
    trace={'concept:name':aid,'RequestedAmount':str(amount),'LoanGoal':'Goal','ApplicationType':'New'}
    events=[]
    for i in range(4):
        e={k:'usual' for k in CAT if k not in ('LoanGoal','ApplicationType')}
        e.update({'concept:name':'O_Create Offer' if i in (1,2) else 'Work','time:timestamp':f'2016-01-01T00:00:0{i}Z','EventID':f'{aid}-{i}','Accepted':'true' if i==1 else 'false'})
        if i in (1,2):e.update({k:str(0. if k=='CreditScore' else amount+i) for k in NUM if k!='RequestedAmount'})
        if unknown:e['Action']='held-out-only'
        events.append(e)
    rows=[dict(application_id=aid,offer_id=f'{aid}-{i}',label=int(i==1),creation_time=events[i]['time:timestamp'],prefix_length=i+1,split=split) for i in (1,2)]
    return trace,events,rows


def app(*args,**kwargs):return extract_application(*fixture(*args,**kwargs))


def state_and_examples():
    apps=[app('a'),app('b',amount=30.)];state=fit_preprocessing(apps)
    data=PackedOffers(pack_split(apps,state,'train'))
    return state,[data[0],data[1]]


class AdapterDataTests(unittest.TestCase):
    def test_unique_training_events_and_static_stats(self):
        s=fit_preprocessing([app('a'),app('b',amount=30.)])
        self.assertEqual(s['train_unique_events'],6)
        self.assertEqual(s['numeric'][0]['count'],2)
        self.assertEqual(s['numeric'][0]['mean'],20.)
        self.assertEqual(s['numeric'][1]['count'],4)
        self.assertEqual(s['numeric'][1]['mean'],21.5)
    def test_heldout_mutations_do_not_change_fitting(self):
        a=app();held=app('h','test',999.,True)
        before=fit_preprocessing([a,held]);held.x[:]=123456;held.categories[:]='new';held.timestamp[:]=99;held.offers[0]['label']=0
        self.assertEqual(before,fit_preprocessing([a,held]))
    def test_future_event_mutation_ignored(self):
        t,e,r=fixture();a=extract_application(t,e,r)
        e[-1]={'time:timestamp':'bad','Accepted':'unknown','RequestedAmount':'nan'}
        b=extract_application(t,e,r)
        np.testing.assert_array_equal(a.x,b.x);np.testing.assert_array_equal(a.categories,b.categories)
    def test_forbidden_fields_do_not_enter_features(self):
        t,e,r=fixture();a=extract_application(t,e,r)
        for x in e:x.update(Selected='leak',OfferID='leak',arbitrary_future_value='leak')
        b=extract_application(t,e,r);np.testing.assert_array_equal(a.categories,b.categories)
        self.assertEqual(a.x.shape[-1],6);self.assertEqual(a.categories.shape[-1],7)
    def test_target_mismatch_rejected(self):
        t,e,r=fixture();r[0]['label']=0
        with self.assertRaises(ValueError):extract_application(t,e,r)
    def test_zero_missing_padding_are_distinct(self):
        state,examples=state_and_examples();x,_,_=collate_numpy(examples)
        self.assertTrue(x['missing_mask'][0,0,4]);self.assertEqual(x['x_obs'][0,0,4],0)
        self.assertFalse(x['missing_mask'][0,1,4]);self.assertEqual(x['x_obs'][0,1,4],0)
        self.assertTrue(x['valid_mask'][0,0]);self.assertFalse(x['valid_mask'][0,2])
        self.assertEqual(x['categorical'][0,2].tolist(),[0]*7)
    def test_missing_unknown_and_known_categories(self):
        a=app();s=fit_preprocessing([a]);b=app('b','test',unknown=True);b.categories[0,1]=None
        enc=encode_application(b,s)
        self.assertEqual(enc['categorical'][0,0],2);self.assertEqual(enc['categorical'][0,1],1)
        self.assertGreaterEqual(enc['categorical'][1,1],3)
    def test_no_double_standardisation(self):
        a=app(amount=123.);s=fit_preprocessing([a]);x=encode_application(a,s)
        self.assertEqual(x['x_obs'][0,0],123.)
    def test_time_gap_and_position(self):
        a=app();s=fit_preprocessing([a]);x=encode_application(a,s)
        np.testing.assert_array_equal(x['timestamp'],[0,1,2]);np.testing.assert_array_equal(x['delta'],[0,1,1]);np.testing.assert_array_equal(x['position'],[0,1,2])
    def test_constant_and_empty_numeric_channel(self):
        s=fit_preprocessing([app()]);self.assertTrue(s['numeric'][4]['constant']);self.assertEqual(s['numeric'][4]['scale'],1.)
        a=app();a.missing[:,4]=True
        with self.assertRaises(ValueError):fit_preprocessing([a])
    def test_invalid_scale_and_nonfinite_source_rejected(self):
        with self.assertRaises(ValueError):fit_preprocessing([app()],scale_floor=0)
        t,e,r=fixture();e[1]['CreditScore']='nan'
        with self.assertRaises(ValueError):extract_application(t,e,r)
    def test_prefix_and_metadata_roundtrip(self):
        apps=[app()];s=fit_preprocessing(apps);packed=pack_split(apps,s,'train');ds=PackedOffers(packed)
        self.assertEqual(len(ds),2);self.assertEqual(len(ds[0][0]['x_obs']),2);self.assertEqual(len(ds[1][0]['x_obs']),3)
        self.assertEqual(ds[0][1],1.);self.assertEqual(ds[1][1],0.)
        self.assertEqual(ds[0][2]['offer_id'],'a-1');self.assertNotIn('offer_id',ds[0][0])
    def test_deterministic_cache_and_reload(self):
        apps=[app(s,s,amount=i+10.) for i,s in enumerate(('train','validation','calibration','test'))];s=fit_preprocessing(apps)
        with tempfile.TemporaryDirectory() as d:
            a=save_cache(apps,s,Path(d)/'a');b=save_cache(apps,s,Path(d)/'b')
            for split in a:
                self.assertEqual(a[split]['logical_sha256'],b[split]['logical_sha256'])
                self.assertEqual(a[split]['logical_sha256'],logical_hash(PackedOffers.load(Path(d)/'a'/f'{split}.npz').arrays))
    def test_collate_does_not_modify_examples(self):
        _,ex=state_and_examples();original=ex[0][0]['x_obs'].copy();batch,_,_=collate_numpy(ex);batch['x_obs'][0,0,0]=999
        np.testing.assert_array_equal(ex[0][0]['x_obs'],original)


class AdapterModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch;torch.set_num_threads(1)
    def make(self,identity=False):
        from ipat.model.bpi17 import config_from_state,model_stats,MODELS
        s,ex=state_and_examples();kw=dict(dropout=0.,n_layers=1,gamma=0.)
        if identity:kw['asinh_scales']=()
        cfg=config_from_state(s,**kw);models=[]
        for model in MODELS:
            self.torch.manual_seed(37);models.append(model(cfg,model_stats(s)))
        b,y,_=collate_numpy(ex);b={k:self.torch.from_numpy(v) for k,v in b.items()}
        return cfg,models,b,self.torch.from_numpy(y),ex
    def test_dimensions_and_shared_initialisation(self):
        cfg,ms,b,y,_=self.make();self.assertEqual((cfg.d_in,cfg.d_model),(59,60))
        base=dict(ms[0].named_parameters())
        for m in ms[1:]:
            p=dict(m.named_parameters())
            for name,v in base.items():self.assertTrue(self.torch.equal(v,p[name]),name)
    def test_limiting_forward_attention_and_shared_gradients(self):
        _,ms,b,y,_=self.make(identity=True);outs=[];grads=[]
        for m in ms:
            out=m(b,return_diagnostics=True);outs.append(out)
            self.torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],y).backward()
            grads.append({n:p.grad for n,p in m.named_parameters()})
        for out,grad in zip(outs[1:],grads[1:]):
            self.assertTrue(self.torch.equal(outs[0]['logits'],out['logits']))
            for a,c in zip(outs[0]['attention'],out['attention']):self.assertTrue(self.torch.equal(a,c))
            for name,g in grads[0].items():
                if g is not None:self.assertTrue(self.torch.equal(g,grad[name]),name)
    def test_padding_invariance(self):
        _,ms,b,y,ex=self.make();p,_,_=collate_numpy(ex,extra_padding=5);p={k:self.torch.from_numpy(v) for k,v in p.items()}
        for m in ms:
            m.eval();self.torch.testing.assert_close(m(b)['logits'],m(p)['logits'],atol=1e-6,rtol=1e-6)
    def test_temporal_encoder_exact_convention(self):
        _,ms,b,_,_=self.make();comp=ms[0](b,return_diagnostics=True)['components'];t=ms[0].assembler.temporal
        expected=(b['timestamp']-t.timestamp_mean)/t.timestamp_scale
        self.torch.testing.assert_close(comp['time'][...,0][b['valid_mask']],expected[b['valid_mask']])
        self.assertEqual(comp['time'][0,1,2].item(),1.)
        self.assertEqual(comp['time'][1,1,2].item(),0.5)
    def test_no_double_scaling_and_missing_expansion_zero(self):
        _,ms,b,_,_=self.make();out=ms[0](b,return_diagnostics=True);n=ms[0].assembler.numeric
        expected=(b['x_obs']-n.mean)/n.scale;observed=b['valid_mask'].unsqueeze(-1)&~b['missing_mask']
        self.torch.testing.assert_close(out['components']['x_std'][observed],expected[observed])
        self.assertEqual(out['components']['x_std'][0,0,4].item(),0.)
    def test_projection_and_finite_training_step(self):
        _,ms,b,y,_=self.make();m=ms[-1];m.set_gamma(1.)
        w=m.projector.orthogonal_weight();self.torch.testing.assert_close(w.T@w,self.torch.eye(w.shape[1]),atol=1e-6,rtol=1e-6)
        before=m.classifier.weight.detach().clone();opt=self.torch.optim.AdamW(m.parameters(),lr=1e-3)
        loss=self.torch.nn.functional.binary_cross_entropy_with_logits(m(b)['logits'],y);loss.backward()
        self.assertTrue(all(self.torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None));opt.step()
        self.assertFalse(self.torch.equal(before,m.classifier.weight))

if __name__=='__main__':unittest.main()
