"""Artificial reporting fixtures only; never research results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("plots", Path(__file__).resolve().parents[1] / "scripts/plot_exp03.py")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")

def fixture(folder, models=()):
    m = {"experiment": "EXP-03", "development": True, "protocol": {"seeds": [17]}}
    fp = plots.json_digest(m)
    m["fingerprint"] = fp
    write(folder / "manifest.json", m)
    write(folder / "preflight.json", {"fingerprint": fp, "status": "PREFLIGHT_PASS"})
    for model in models:
        d = folder / "seed_17" / model
        d.mkdir(parents=True)
        (d / "checkpoint.pt").write_bytes(b"ARTIFICIAL CHECKPOINT; NEVER LOADED")
        np.savez(d / "predictions.npz", triplet_id=[0, 0, 0, 1, 1, 1],
                 variant=["late", "early", "middle"] * 2, probability=[.6, .4, .5, .2, .3, .4],
                 label=[1, 1, 1, 0, 0, 0], p_true=[.7, .7, .7, .2, .2, .2])
        quality = {"auroc": .75, "accuracy": .5, "nll": .7, "brier": .25}
        metrics = {"primary_probability_range": .2, "macro": quality,
                   "per_position": {v: quality for v in plots.POSITIONS},
                   "diagnostics_by_position": {v: {"probe_attention_mass": .2, "uniform_probe_mass": .1,
                       **({"signal_retention_mean": .6, "nuisance_retention_mean": .4} if model != "baseline" else {})}
                       for v in plots.POSITIONS}}
        r = {"fingerprint": fp, "status": "RUN_COMPLETE", "seed": 17, "model": model,
             "metrics": metrics, "training": {"best_epoch": 1, "trajectory": [
                 {"epoch": 1, "train_nll": .7, "val_nll": .8}, {"epoch": 2, "train_nll": .6, "val_nll": .9}]},
             "artefact_sha256": {p: plots.digest(d / p) for p in ("checkpoint.pt", "predictions.npz")}}
        r["content_sha256"] = plots.json_digest(r)
        write(d / "run.json", r)

class VisualTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "input"
        self.folder.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_preflight_no_predictions_and_input_unchanged(self):
        fixture(self.folder)
        before = {p.name: plots.digest(p) for p in self.folder.iterdir()}
        out = plots.render(self.folder, Path(self.tmp.name) / "report")
        self.assertEqual(before, {p.name: plots.digest(p) for p in self.folder.iterdir()})
        self.assertEqual(len(list(out.glob("*.png"))), 1)
        self.assertIn("PREFLIGHT ONLY", (out / "report.html").read_text())

    def test_partial_pair_excluded_and_replay_ignored(self):
        fixture(self.folder, ["baseline"])
        write(self.folder / "replay/seed_17/retention/run.json", {"invalid": "ignored"})
        m, runs, _, status = plots.collect(self.folder)
        self.assertEqual(len(runs), 1)
        self.assertEqual(plots.paired_seeds(m, runs), [])
        self.assertIn("PARTIAL", status)

    def test_tampered_predictions_rejected(self):
        fixture(self.folder, ["baseline"])
        with (self.folder / "seed_17/baseline/predictions.npz").open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Artefact hash"):
            plots.collect(self.folder)

    def test_tampered_manifest_rejected(self):
        fixture(self.folder)
        m = plots.read_json(self.folder / "manifest.json")
        m["protocol"]["seeds"] = [18]
        write(self.folder / "manifest.json", m)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            plots.collect(self.folder)

    def test_align_triplets_by_id_and_position(self):
        a = {"triplet_id": np.array([9, 9, 9]), "variant": np.array(["late", "early", "middle"]),
             "probability": np.array([.8, .2, .4]), "label": np.ones(3), "p_true": np.full(3, .7)}
        ids, matrix, _ = plots.triplet_matrix(a)
        np.testing.assert_array_equal(ids, [9])
        np.testing.assert_allclose(matrix, [[.2, .4, .8]])
        a["variant"][0] = "early"
        with self.assertRaisesRegex(ValueError, "Incomplete triplet"):
            plots.triplet_matrix(a)

    def test_all_figures_and_no_overwrite(self):
        fixture(self.folder, plots.MODELS)
        out = Path(self.tmp.name) / "report"
        plots.render(self.folder, out)
        for ext in ("png", "pdf", "svg"):
            self.assertEqual(len(list(out.glob("*." + ext))), 7)
        p = plots.read_json(out / "visual_report_manifest.json")
        self.assertEqual(p["paired_seeds"], [17])
        self.assertIn("DEVELOPMENT ONLY", p["status"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            plots.render(self.folder, out)

    def test_final_status_requires_consistent_summary_and_replay(self):
        fixture(self.folder, plots.MODELS)
        m, runs, _, _ = plots.collect(self.folder)
        summary = {"fingerprint": m["fingerprint"], "runs": list(runs.values()),
                   "status": "DEVELOPMENT_ONLY", "replay": {k: {"status": "PASS"} for k in plots.MODELS}}
        path = self.folder / "exp03_position_sweep_v0.1.json"
        write(path, summary)
        write(self.folder / "progress.json", {"fingerprint": m["fingerprint"], "summary_sha256": plots.digest(path)})
        self.assertIn("EXECUTION COMPLETE", plots.collect(self.folder)[3])
        summary["replay"]["baseline"]["status"] = "FAIL"
        write(path, summary)
        write(self.folder / "progress.json", {"fingerprint": m["fingerprint"], "summary_sha256": plots.digest(path)})
        with self.assertRaisesRegex(ValueError, "Replay not passed"):
            plots.collect(self.folder)

if __name__ == "__main__":
    unittest.main(verbosity=2)
