import math
import unittest
import numpy as np
from ipat.experiments.cal01_core import fit_temperature, scale_logits, nll, load_partition


class TemperatureTests(unittest.TestCase):
    def test_known_interior_optimum(self):
        # All raw logits are log(9), but the observed positive fraction is 3/4.
        # sigmoid(log(9)/T)=3/4 gives T=2 exactly.
        y=np.array([1,1,1,0])
        r=fit_temperature(y,np.full(4,math.log(9)))
        self.assertAlmostEqual(r['temperature'],2.,places=9)
        self.assertEqual(r['status'],'INTERIOR')

    def test_identity_optimum(self):
        r=fit_temperature([1,1,1,0],np.full(4,math.log(3)))
        self.assertAlmostEqual(r['temperature'],1.,places=7)

    def test_underconfidence_sharpens(self):
        r=fit_temperature([1,1,1,0],np.full(4,math.log(3)/2))
        self.assertAlmostEqual(r['temperature'],.5,places=8)

    def test_class_signs_and_ties_preserved(self):
        x=np.array([-1000.,-1.,0.,1.,1000.])
        for t in [.05,1.,20.]:
            np.testing.assert_array_equal(x>=0,scale_logits(x,t)>=0)

    def test_invalid_temperatures(self):
        for t in [0.,-1.,np.inf,np.nan]:
            with self.assertRaises(ValueError):scale_logits([1.],t)

    def test_invalid_inputs(self):
        for y,x in [([],[]),([1],[1,2]),([.1,1],[1,2]),([0,1],[np.nan,0]),([1,1],[1,2])]:
            with self.assertRaises(ValueError):fit_temperature(y,x)

    def test_zero_logits_unidentifiable(self):
        r=fit_temperature([0,1],[0,0])
        self.assertEqual(r['status'],'UNIDENTIFIABLE')
        self.assertEqual(r['temperature'],1.)

    def test_separable_boundary_reported(self):
        r=fit_temperature([0,1],[-1,1])
        self.assertEqual(r['temperature'],.05)
        self.assertEqual(r['status'],'BOUNDARY')

    def test_reversed_predictions_boundary_reported(self):
        r=fit_temperature([0,1],[1,-1])
        self.assertEqual(r['temperature'],20.)
        self.assertEqual(r['status'],'BOUNDARY')

    def test_hand_nll_and_large_logits(self):
        self.assertAlmostEqual(nll([0,1],[0,0]),math.log(2))
        self.assertEqual(nll([0,1],[10000,-10000]),10000.)
        self.assertEqual(nll([0,1],[-10000,10000]),0.)

    def test_fit_reproducible_and_inputs_unchanged(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=100);y=rng.integers(0,2,size=100)
        xx,yy=x.copy(),y.copy()
        a=fit_temperature(y,x);b=fit_temperature(y,x)
        self.assertEqual(a,b)
        self.assertLessEqual(a['fitted_nll'],a['raw_nll']+1e-10)
        np.testing.assert_equal(x,xx);np.testing.assert_equal(y,yy)

    def test_test_and_train_rejected_before_io(self):
        for split in ['test','train','../test']:
            with self.assertRaises(ValueError):load_partition('does-not-exist',split,'x')

    def test_objective_derivative_matches_finite_difference(self):
        x=np.array([-2.,-.5,.7,1.8]);y=np.array([0,1,1,0])
        beta=.7;h=1e-6
        d=(nll(y,x*(beta+h))-nll(y,x*(beta-h)))/(2*h)
        p=np.exp(-np.logaddexp(0.,-beta*x))
        self.assertAlmostEqual(d,float(np.mean(x*(p-y))),places=8)

if __name__=='__main__':unittest.main()
