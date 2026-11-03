"""EXP-03 acceptance tests. unittest-compatible, also collected by pytest.

Dependency-free here means no torch/h5py: NumPy, SciPy and scikit-learn remain required.
Missing model/data dependencies produce explicit skips, never evidence of full acceptance.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Load the pure contract without the legacy experiments/__init__.py (which imports torch).
spec = importlib.util.spec_from_file_location("exp03_pure_contract", ROOT / "src/ipat/experiments/exp03_contract.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)
HAS_DATA = importlib.util.find_spec("h5py") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class MiniSweep:
    def __init__(self):
        self.arrays = {"y": np.repeat([0, 1], 3), "triplet_id": np.repeat([10, 20], 3),
                       "variant": np.tile(np.array(C.VARIANTS, dtype="S8"), 2)}

    def __getitem__(self, key):
        return self.arrays[key]


class MetricContractTests(unittest.TestCase):
    def test_approved_ten_seeds(self):
        C.Exp03Config().validate()
        self.assertEqual(len(C.SEEDS), 10)

    def test_wrong_formal_seed_count_rejected(self):
        with self.assertRaises(ValueError):
            replace(C.Exp03Config(), seeds=(1, 2)).validate()

    def test_duplicate_seeds_rejected(self):
        with self.assertRaises(ValueError):
            replace(C.Exp03Config(), seeds=(1, 1)).validate(False)

    def test_nonfinite_config_rejected(self):
        for field in ("learning_rate", "weight_decay", "gamma", "min_delta", "replay_atol"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(C.Exp03Config(), **{field: float("nan")}).validate()

    def test_invalid_integer_and_scales_rejected(self):
        for changes in ({"epochs": 0}, {"batch_size": 0}, {"patience": 0},
                        {"asinh_scales": ()}, {"asinh_scales": (1, 1)}, {"identity_floor": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(C.Exp03Config(), **changes).validate()

    def test_probability_range_matches_hand_calculation(self):
        p = np.array([.1, .2, .4, .8, .9, .7])
        r = C.summarise_predictions(MiniSweep(), np.log(p/(1-p)))
        self.assertAlmostEqual(r["primary_probability_range"], .25)
        self.assertEqual(r["prediction_count"], 6)

    def test_constant_predictor_stable_but_uninformative(self):
        r = C.summarise_predictions(MiniSweep(), np.zeros(6))
        self.assertEqual(r["primary_probability_range"], 0)
        self.assertEqual(r["macro"]["auroc"], .5)
        self.assertEqual(r["macro"]["accuracy"], .5)
        self.assertEqual(r["macro"]["brier"], .25)

    def test_row_order_does_not_change_metrics(self):
        a, b = MiniSweep(), MiniSweep()
        order = np.array([4, 1, 5, 0, 3, 2])
        b.arrays = {k: v[order] for k, v in b.arrays.items()}
        logits = np.array([-2., -1., -.5, 1., 2., 3.])
        self.assertEqual(C.summarise_predictions(a, logits), C.summarise_predictions(b, logits[order]))

    def test_duplicate_variant_rejected(self):
        sweep = MiniSweep()
        sweep.arrays["variant"][0] = b"late"
        with self.assertRaises(ValueError):
            C.triplet_rows(sweep)

    def test_nonfinite_predictions_rejected(self):
        with self.assertRaises(ValueError):
            C.summarise_predictions(MiniSweep(), np.full(6, np.nan))

    def test_wrong_prediction_shape_rejected(self):
        with self.assertRaises(ValueError):
            C.summarise_predictions(MiniSweep(), np.zeros((6, 1)))

    def test_extreme_logit_nll_is_finite(self):
        result = C.binary_metrics([0, 1], [1000., -1000.])
        self.assertEqual(result["nll"], 1000.)
        self.assertEqual(result["brier"], 1.)

    def test_worst_position_direction(self):
        r = C.summarise_predictions(MiniSweep(), np.array([-1., 1., -1., 1., -1., 1.]))
        self.assertEqual(r["worst_position"]["accuracy"], 0.)
        self.assertEqual(r["worst_position"]["auroc"], 0.)
        self.assertEqual(r["worst_position"]["nll"], r["per_position"]["middle"]["nll"])

    def test_five_seed_exact_minimum(self):
        self.assertEqual(C.signed_rank_exact([1, 2, 3, 4, 5])["pvalue"], .0625)

    def test_signed_rank_ties_zeros(self):
        self.assertEqual(C.signed_rank_exact([1, 1, 1, 1])["pvalue"], .125)
        self.assertEqual(C.signed_rank_exact([0, 0, 0])["pvalue"], 1.)
        self.assertEqual(C.signed_rank_exact([1, -1])["pvalue"], 1.)

    def test_ci_states_fixed_dataset_scope(self):
        r = C.describe([1., 2., 3.])
        self.assertEqual(r["mean"], 2.)
        self.assertEqual(r["sd"], 1.)
        self.assertIn("fixed datasets", r["ci_scope"])
        self.assertIsNone(C.describe([1.])["mean_95ci_t"])

    def test_paired_contrasts_and_holm(self):
        runs = []
        for i, model in enumerate(C.MODELS):
            for seed in (1, 2):
                m = C.summarise_predictions(MiniSweep(), np.zeros(6))
                m["primary_probability_range"] = .3 - .1*i + seed*.001
                runs.append({"model": model, "seed": seed, "metrics": m})
        r = C.aggregate_runs(runs, (1, 2))
        contrast = r["paired_primary_contrasts"]["retention_minus_baseline"]
        self.assertAlmostEqual(contrast["mean"], -.1)
        self.assertGreaterEqual(contrast["secondary_signed_rank"]["holm_pvalue"],
                                contrast["secondary_signed_rank"]["pvalue"])
        with self.assertRaises(ValueError):
            C.aggregate_runs(runs[:-1], (1, 2))
        with self.assertRaises(ValueError):
            C.aggregate_runs(runs + runs[:1], (1, 2))

    def test_hashes_include_shape_and_dtype(self):
        a = np.array([1., 2.])
        self.assertNotEqual(C.array_hash(a), C.array_hash(a.reshape(1, 2)))
        self.assertNotEqual(C.array_hash(a), C.array_hash(a.astype(np.float32)))

    def test_diagnostic_shape_rejected(self):
        with self.assertRaises(ValueError):
            C.summarise_predictions(MiniSweep(), np.zeros(6), {"probe_retention": np.zeros(5)})


@unittest.skipUnless(HAS_DATA, "h5py is required for the frozen DATA-01 generator")
class DataContractTests(unittest.TestCase):
    def setUp(self):
        from ipat.data.synthetic import IPATSyntheticConfig, generate_position_sweep, generate_split
        self.cfg = IPATSyntheticConfig(n_train=32, n_val=24, t_min=8, t_max=12)
        self.train = generate_split(self.cfg, "train")
        self.val = generate_split(self.cfg, "val")
        self.sweep = generate_position_sweep(self.cfg, 16, seed=self.cfg.master_seed+77)

    def test_valid_inputs(self):
        self.assertEqual(C.validate_inputs(self.train, self.val, self.sweep, False)["status"], "PASS")

    def test_debug_data_cannot_be_formal(self):
        with self.assertRaises(ValueError):
            C.validate_inputs(self.train, self.val, self.sweep, True)

    def test_label_corruption_rejected(self):
        self.sweep.arrays["y"][0] ^= 1
        with self.assertRaises(ValueError):
            C.validate_sweep(self.sweep, 16)

    def test_probe_target_corruption_rejected(self):
        self.sweep.arrays["probe_target_position"][0] += 1
        with self.assertRaises(ValueError):
            C.validate_sweep(self.sweep, 16)

    def test_event_content_corruption_rejected(self):
        self.sweep.arrays["nuisance_cat"][1, 0] = (self.sweep["nuisance_cat"][1, 0]+1) % 8
        with self.assertRaises(ValueError):
            C.validate_sweep(self.sweep, 16)

    def test_bad_timestamp_rejected(self):
        self.sweep.arrays["timestamp"][0, 0] += 1
        with self.assertRaises(ValueError):
            C.validate_sweep(self.sweep, 16)

    def test_padding_corruption_rejected(self):
        row = int(np.argmin(self.sweep["length"]))
        self.sweep.arrays["valid_mask"][row, -1] = True
        with self.assertRaises(ValueError):
            C.validate_sweep(self.sweep, 16)

    def test_nonzero_temporal_sweep_rejected(self):
        self.sweep.metadata["config"]["temporal_signal_weight"] = .4
        with self.assertRaises(ValueError):
            C.validate_sweep(self.sweep, 16)

    def test_one_class_rejected(self):
        self.sweep.arrays["y"][:] = 0
        with self.assertRaises(ValueError):
            C.validate_sweep(self.sweep, 16)


@unittest.skipUnless(HAS_DATA and HAS_TORCH, "PyTorch and h5py required for integration acceptance")
class TorchIntegrationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from ipat.data.synthetic import IPATSyntheticConfig, generate_position_sweep, generate_split
        from ipat.training.preprocessing import fit_baseline_preprocessing_stats
        from ipat.model.baseline import BaselineConfig
        from ipat.experiments import exp03 as E
        self.E = E
        torch.set_num_threads(1)
        data_cfg = IPATSyntheticConfig(n_train=24, n_val=20, t_min=8, t_max=10)
        self.train, self.val = generate_split(data_cfg, "train"), generate_split(data_cfg, "val")
        self.sweep = generate_position_sweep(data_cfg, 12, seed=data_cfg.master_seed+77)
        self.stats = fit_baseline_preprocessing_stats(self.train)
        self.cfg = E.Exp03Config(seeds=(17,), epochs=2, batch_size=8, patience=2)
        self.base = BaselineConfig(n_layers=1, ff_multiplier=1, dropout=.1)
        self.device = torch.device("cpu")

    def test_shared_initial_states(self):
        models, evidence = self.E.build_models(self.stats, self.cfg, self.base, 17)
        self.assertEqual(evidence["baseline_to_retention"]["max_difference"], 0.)
        self.assertEqual(evidence["retention_to_full"]["max_difference"], 0.)
        self.assertGreater(evidence["full_ipat"]["parameter_count"], evidence["retention"]["parameter_count"])

    def test_initial_baseline_retention_parity(self):
        import torch
        models, _ = self.E.build_models(self.stats, self.cfg, self.base, 17)
        batch = next(iter(self.E.loader(self.val, range(4), 4)))
        inputs = self.E.model_inputs(batch, self.device)
        self.assertNotIn("signal_mask", inputs)
        self.assertNotIn("p_true", inputs)
        for m in models.values():
            m.eval()
        with torch.no_grad():
            a, b = (models[n](inputs)["logits"] for n in ("baseline", "retention"))
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)

    def test_epoch_order_reproducible_independent(self):
        self.assertTrue(np.array_equal(self.E.epoch_order(24, 17, 1), self.E.epoch_order(24, 17, 1)))
        self.assertFalse(np.array_equal(self.E.epoch_order(24, 17, 1), self.E.epoch_order(24, 17, 2)))

    def test_three_models_train_and_evaluate(self):
        models, _ = self.E.build_models(self.stats, self.cfg, self.base, 17)
        for name, model in models.items():
            with self.subTest(model=name):
                training = self.E.train_model(model, self.train, self.val, self.cfg, 17, self.device, name)
                m, arrays = self.E.evaluate(model, self.sweep, self.cfg, self.device)
                self.assertEqual(m["prediction_count"], 36)
                self.assertIn(training["best_epoch"], (1, 2))
                self.assertEqual(training["best_val_nll"], min(r["val_nll"] for r in training["trajectory"]))
                self.assertTrue(np.isfinite(arrays["logits"]).all())

    def test_full_runner_resume_and_replay(self):
        from ipat.data.synthetic import save_hdf5
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            paths = [save_hdf5(d, p / f"{name}.h5") for name, d in
                     (("train", self.train), ("val", self.val), ("sweep", self.sweep))]
            args = (*paths, p / "output", self.cfg, self.base)
            r = self.E.run_exp03(*args, development=True)
            self.assertEqual(r["status"], "DEVELOPMENT_ONLY")
            self.assertEqual(r["primary_prediction_count"], 108)
            self.assertTrue(all(v["status"] == "PASS" for v in r["replay"].values()))
            with self.assertRaises(ValueError):
                self.E.run_exp03(*args, development=True)
            again = self.E.run_exp03(*args, development=True, resume=True)
            self.assertEqual(r["aggregate"], again["aggregate"])
            checkpoint = p / "output" / "seed_17" / "baseline" / "checkpoint.pt"
            checkpoint.write_bytes(b"corrupted test fixture")
            with self.assertRaises(ValueError):
                self.E.run_exp03(*args, development=True, resume=True)

    def test_resume_rejects_changed_config(self):
        from ipat.data.synthetic import save_hdf5
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            paths = [save_hdf5(d, p / f"{name}.h5") for name, d in
                     (("train", self.train), ("val", self.val), ("sweep", self.sweep))]
            self.E.run_exp03(*paths, p / "out", self.cfg, self.base, development=True, preflight_only=True)
            with self.assertRaises(ValueError):
                self.E.run_exp03(*paths, p / "out", replace(self.cfg, learning_rate=.01), self.base,
                                development=True, preflight_only=True, resume=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
