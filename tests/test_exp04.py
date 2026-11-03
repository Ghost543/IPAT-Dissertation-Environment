"""EXP-04 engineering gates; generated fixtures are not dissertation results."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ipat.experiments import exp04_contract as C
from ipat.experiments import exp04 as E


class MetricTests(unittest.TestCase):
    def test_config(self):
        C.Exp04Config().validate()
        for changes in ({'ece_bins':10},{'batch_size':0},{'temperature_min':1},
                        {'temperature_max':1},{'replay_atol':-1},{'temperature_xatol':float('nan')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(C.Exp04Config(),**changes).validate()

    def test_stable_nll_extremes(self):
        self.assertEqual(C.nll([0,1],[1000,-1000]),1000)
        self.assertAlmostEqual(C.nll([0,1],[0,0]),np.log(2))
        self.assertEqual(C.nll([0,1],[-1000,1000]),0)

    def test_invalid_metric_inputs(self):
        for y,z in (([0,1],[np.nan,0]),([0,1],[[1],[2]]),([0,2],[0,1]),([0,0],[1,2])):
            with self.assertRaises(ValueError): C.metrics(y,z)

    def test_ece_hand_calculation(self):
        r=C.calibration_bins([0,1,1,0],[-np.log(3),np.log(3),-np.log(3),np.log(3)])
        self.assertAlmostEqual(r['ece'],.25)
        self.assertEqual(sum(b['count'] for b in r['bins']),4)
        self.assertEqual(sum(b['count']>0 for b in r['bins']),1)

    def test_ece_upper_boundary_empty_bins(self):
        r=C.calibration_bins([0,1],[-1000,1000])
        self.assertEqual(r['ece'],0)
        self.assertEqual(r['bins'][-1]['count'],2)
        self.assertIsNone(r['bins'][0]['confidence'])

    def test_ece_tie_decision(self):
        self.assertEqual(C.calibration_bins([0,1],[0,0])['ece'],0)
        self.assertEqual(C.metrics([0,1],[0,0])['accuracy'],.5)

    def test_equal_mass_stable_ties(self):
        y=np.array([0,1]*16)
        a=C.calibration_bins(y,np.ones(32),equal_mass=True)
        self.assertEqual(a,C.calibration_bins(y,np.ones(32),equal_mass=True))
        counts=[b['count'] for b in a['bins']]
        self.assertEqual(sum(counts),32)
        self.assertLessEqual(max(counts)-min(counts),1)

    def test_temperature_identity_rank_invariance(self):
        y,z=[0,1,0,1],[-3,1,2,4]
        self.assertEqual(C.metrics(y,z),C.metrics(y,z,1))
        for t in (.01,100):
            for key in ('accuracy','auroc'):
                self.assertEqual(C.metrics(y,z,t)[key],C.metrics(y,z)[key])

    def test_temperature_known_optimum(self):
        r=C.fit_temperature([0,1,1,1],np.full(4,np.log(9)),C.Exp04Config())
        self.assertAlmostEqual(r['temperature'],2,places=5)
        self.assertLess(r['clean_cal_nll_scaled'],r['clean_cal_nll_raw'])
        self.assertFalse(r['boundary_solution'])

    def test_temperature_flat_identity(self):
        r=C.fit_temperature([0,1],[0,0],C.Exp04Config())
        self.assertEqual(r['temperature'],1)
        self.assertFalse(r['boundary_solution'])

    def test_temperature_boundary(self):
        r=C.fit_temperature([0,1],[2,-2],C.Exp04Config())
        self.assertEqual(r['temperature'],100)
        self.assertTrue(r['boundary_solution'])

    @staticmethod
    def conditions():
        return {c:{p:{m:.5 for m in C.METRICS} for p in C.PANELS} for c in C.CONDITIONS}

    def test_signs_equal_family_weights(self):
        c=self.conditions()
        for condition in ('missing_10','missing_20','missing_40'):
            for p in C.PANELS:
                for m in C.METRICS: c[condition][p][m]=.7
        self.assertAlmostEqual(C.macro_change(c,'raw','nll')['mean'],-.05)
        self.assertAlmostEqual(C.macro_change(c,'raw','auroc')['mean'],.05)
        self.assertAlmostEqual(C.macro_change(c,'raw','ece')['mean'],.05)
        del c['delete_25']
        with self.assertRaises(ValueError): C.macro_change(c,'raw','nll')

    def test_aggregate_pairing_holm_direction(self):
        runs=[]
        for model in C.MODELS:
            for seed in (1,2,3):
                c=self.conditions()
                for condition in C.CONDITIONS[1:]:
                    c[condition]['raw']['nll'] += .1 if model=='baseline' else .05
                runs.append({'model':model,'seed':seed,'conditions':c})
        pair=C.aggregate(runs,(1,2,3))['paired_primary_contrasts']['full_ipat_minus_baseline']
        self.assertAlmostEqual(pair['mean'],.05)
        self.assertGreaterEqual(pair['secondary_signed_rank']['holm_pvalue'],pair['secondary_signed_rank']['pvalue'])
        for bad in (runs[:-1],runs+runs[:1]):
            with self.assertRaises(ValueError): C.aggregate(bad,(1,2,3))

    def test_undefined_ranking_not_zero(self):
        self.assertTrue(np.isnan(E.rank_auc(np.array([False,False]),np.array([.2,.3]))))
        self.assertTrue(np.isnan(E.rank_auc(np.array([True,True]),np.array([.2,.3]))))
        self.assertEqual(E.rank_auc(np.array([True,False]),np.array([.5,.5])),.5)
        self.assertEqual(E.rank_auc(np.array([True,False]),np.array([.9,.1])),1)

    def test_replay_eligibility_change(self):
        a={'x':np.array([np.nan,1.]),'id':np.array([1,2])}
        self.assertEqual(E.compare_arrays(a,a,0),0)
        with self.assertRaises(ValueError):
            E.compare_arrays(a,{'x':np.array([0.,1.]),'id':np.array([1,2])},0)

    def test_safe_paths(self):
        with self.assertRaises(ValueError): E.safe_relative(ROOT,'../outside.txt')
        self.assertEqual(E.safe_relative(ROOT,'src\\ipat\\data\\synthetic.py'),ROOT/'src/ipat/data/synthetic.py')


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ipat.data.synthetic import IPATSyntheticConfig,generate_split,compute_train_numeric_scale
        cls.cfg=IPATSyntheticConfig(n_train=48,n_val=32,n_cal=32,n_test=64,t_min=8,t_max=12)
        cls.clean=generate_split(cls.cfg,'test')
        cls.scale=compute_train_numeric_scale(generate_split(cls.cfg,'train'))

    def test_all_nine_corruptions(self):
        for condition in C.CONDITIONS[1:]:
            d=C.make_corruption(self.clean,condition,self.scale)
            C.exact_data(d,C.make_corruption(self.clean,condition,self.scale),condition)
            for k in ('y','sequence_id','p_true'): np.testing.assert_array_equal(d[k],self.clean[k])

    def test_tampering_rejected(self):
        d=C.make_corruption(self.clean,'noise_10',self.scale)
        d.arrays['y'][0]^=1
        with self.assertRaises(ValueError): C.exact_data(d,C.make_corruption(self.clean,'noise_10',self.scale),'noise_10')

    def test_deletion_times_positions(self):
        d=C.make_corruption(self.clean,'delete_25',self.scale)
        for row,length in enumerate(d['length']):
            keep=np.flatnonzero(self.clean['valid_mask'][row]&~d['deleted_event_mask_original'][row])
            for k in ('delta','timestamp'): np.testing.assert_array_equal(d[k][row,:length],self.clean[k][row,keep])
            np.testing.assert_array_equal(d['position'][row,:length],np.arange(length))

    def test_unknown_severity(self):
        with self.assertRaises(ValueError): C.make_corruption(self.clean,'missing_30',self.scale)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ipat.data.synthetic import IPATSyntheticConfig,generate_split,generate_position_sweep,save_hdf5,compute_train_numeric_scale
        from ipat.experiments.exp03 import run_exp03
        from ipat.experiments.exp03_contract import Exp03Config
        from ipat.model.baseline import BaselineConfig
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.data_dir=cls.root/'data'
        cls.data_dir.mkdir()
        cfg=IPATSyntheticConfig(n_train=32,n_val=32,n_cal=32,n_test=48,t_min=8,t_max=10)
        splits={s:generate_split(cfg,s) for s in ('train','val','cal','test')}
        for s,d in splits.items(): save_hdf5(d,cls.data_dir/f'{s}.h5')
        scale=compute_train_numeric_scale(splits['train'])
        for c in C.CONDITIONS[1:]: save_hdf5(C.make_corruption(splits['test'],c,scale),cls.data_dir/f'test_{c}.h5')
        save_hdf5(generate_position_sweep(cfg,12,seed=cfg.master_seed+77),cls.data_dir/'position_sweep.h5')
        cls.parent_dir=cls.root/'parent'
        run_exp03(cls.data_dir/'train.h5',cls.data_dir/'val.h5',cls.data_dir/'position_sweep.h5',
            cls.parent_dir,Exp03Config(seeds=(17,),epochs=1,patience=1,batch_size=16),
            BaselineConfig(n_layers=1,ff_multiplier=1),development=True)
        cls.cfg=C.Exp04Config(batch_size=16)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_formal_rejects_development_parent(self):
        with self.assertRaises(ValueError): E.parent_contract(self.parent_dir,ROOT,formal=True)

    def test_parent_and_changed_checkpoint(self):
        _,seeds,_,_=E.parent_contract(self.parent_dir,ROOT,formal=False)
        self.assertEqual(seeds,(17,))
        cp=self.parent_dir/'seed_17/baseline/checkpoint.pt'
        original=cp.read_bytes()
        try:
            cp.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): E.parent_contract(self.parent_dir,ROOT,formal=False)
        finally: cp.write_bytes(original)

    def test_preflight_config_blocks_resume(self):
        out=self.root/'preflight'
        r=E.run_exp04(self.parent_dir,self.data_dir,out,self.cfg,preflight_only=True,development=True)
        self.assertEqual(r['status'],'PREFLIGHT_PASS')
        self.assertFalse((out/'exp04_corruption_retention_v0.1.json').exists())
        with self.assertRaises(ValueError):
            E.run_exp04(self.parent_dir,self.data_dir,out,replace(self.cfg,batch_size=8),preflight_only=True,development=True,resume=True)

    def test_full_evaluation_report_resume_tamper(self):
        out=self.root/'experiment'
        r=E.run_exp04(self.parent_dir,self.data_dir,out,self.cfg,development=True)
        self.assertEqual(r['status'],'DEVELOPMENT_ONLY')
        self.assertEqual(r['primary_test_prediction_count'],3*10*48)
        self.assertEqual(r['calibration_prediction_count'],3*32)
        self.assertEqual(len(r['runs']),3)
        for item in r['runs']:
            self.assertTrue(item['model_state_unchanged'])
            self.assertEqual(item['replay']['status'],'PASS')
            self.assertEqual(set(item['replay']['maximum_differences']),set(C.CONDITIONS)|{'calibration'})
            self.assertEqual(max(item['replay']['maximum_differences'].values()),0)
        report=(out/'exp04_report.html').read_text()
        self.assertIn('DEVELOPMENT_ONLY',report)
        self.assertIn('application/json',report)
        self.assertNotIn('<script src=',report)
        self.assertNotIn('__CHARTS__',report)
        manifest=E.read_json(out/'report_manifest.json')
        self.assertEqual(manifest['figure_count'],9)
        for name,digest in manifest['files'].items(): self.assertEqual(E.file_hash(out/name),digest)
        import os,shutil
        if os.environ.get('IPAT_EXP04_QA_DIR'): shutil.copytree(out,os.environ['IPAT_EXP04_QA_DIR'],dirs_exist_ok=True)
        with self.assertRaises(ValueError): E.run_exp04(self.parent_dir,self.data_dir,out,self.cfg,development=True)
        again=E.run_exp04(self.parent_dir,self.data_dir,out,self.cfg,development=True,resume=True)
        self.assertEqual(r['aggregate'],again['aggregate'])
        (out/'seed_17/baseline/clean_predictions.npz').write_bytes(b'altered')
        with self.assertRaises(ValueError): E.run_exp04(self.parent_dir,self.data_dir,out,self.cfg,development=True,resume=True)

    def test_signal_absent_inference(self):
        from ipat.data.synthetic import load_hdf5
        from ipat.training.preprocessing import fit_baseline_preprocessing_stats
        from ipat.experiments.exp03_contract import Exp03Config
        from ipat.model.baseline import BaselineConfig
        import torch
        data=load_hdf5(self.data_dir/'test.h5')
        data.arrays['signal_mask'][0]=False
        models,_=E.build_models(fit_baseline_preprocessing_stats(load_hdf5(self.data_dir/'train.h5')),
            Exp03Config(seeds=(17,)),BaselineConfig(n_layers=1,ff_multiplier=1),17)
        a=E.predict(models['retention'],data,self.cfg,torch.device('cpu'))
        self.assertTrue(np.isfinite(a['logits']).all())
        self.assertTrue(np.isnan(a['retention_signal_auroc'][0]))
        self.assertFalse(a['retention_ranking_eligible'][0])
        self.assertGreaterEqual(E.diagnostic_summary(a)['retention_signal_auroc']['undefined_count'],1)


if __name__=='__main__': unittest.main(verbosity=2)
