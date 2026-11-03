"""Read-only EXP-03 visual report; no IPAT/PyTorch imports or experiment mutations."""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

MODELS = ("baseline", "retention", "full_ipat")
LABELS = ("Baseline", "Retention", "Full IPAT")
COLOURS = ("#0072B2", "#D55E00", "#009E73")
POSITIONS = ("early", "middle", "late")
LIMIT = ("Stability must be read alongside predictive quality. Canonical training temporal weight 0.4; "
         "sweep weight 0.0: evaluation-profile shift. Attention/retention are non-causal diagnostics.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect(folder):
    """Read only committed primary runs. Replay directories never enter comparisons."""
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    fingerprint = manifest["fingerprint"]
    require(json_digest({k: v for k, v in manifest.items() if k != "fingerprint"}) == fingerprint,
            "Manifest fingerprint mismatch")
    require(manifest["experiment"] == "EXP-03", "Expected EXP-03 manifest")
    seeds = manifest["protocol"]["seeds"]
    require(seeds and len(set(seeds)) == len(seeds), "Missing/duplicate seeds")
    sources = {"manifest.json": digest(folder / "manifest.json")}
    preflight = read_json(folder / "preflight.json")
    require(preflight["fingerprint"] == fingerprint and preflight["status"] == "PREFLIGHT_PASS",
            "Missing or mismatched preflight")
    sources["preflight.json"] = digest(folder / "preflight.json")
    runs = {}
    for seed in seeds:
        for model in MODELS:
            path = folder / f"seed_{seed}" / model / "run.json"
            if not path.exists():
                continue
            r = read_json(path)
            require(r["content_sha256"] == json_digest({k: v for k, v in r.items()
                                                      if k != "content_sha256"}), "Run JSON hash mismatch")
            require(r["fingerprint"] == fingerprint and r["status"] == "RUN_COMPLETE"
                    and r["seed"] == seed and r["model"] == model, "Run identity mismatch")
            # Plotting never unpickles model checkpoints; verify their bytes only.
            for name in ("checkpoint.pt", "predictions.npz"):
                item = path.parent / name
                actual = digest(item)
                require(actual == r["artefact_sha256"][name], f"Artefact hash mismatch: {item}")
                sources[str(item.relative_to(folder))] = actual
            sources[str(path.relative_to(folder))] = digest(path)
            require(np.isfinite(r["metrics"]["primary_probability_range"]), "Non-finite endpoint")
            runs[seed, model] = r
    summary_path = folder / "exp03_position_sweep_v0.1.json"
    status = "PREFLIGHT ONLY" if not runs else "PARTIAL SNAPSHOT"
    if summary_path.exists():
        summary = read_json(summary_path)
        progress = read_json(folder / "progress.json")
        require(progress.get("summary_sha256") == digest(summary_path), "Final summary hash mismatch")
        require(summary["fingerprint"] == fingerprint and progress["fingerprint"] == fingerprint,
                "Final summary fingerprint mismatch")
        require(len(runs) == 3 * len(seeds), "Final summary has missing primary runs")
        listed = {(r["seed"], r["model"]): r for r in summary["runs"]}
        require(len(summary["runs"]) == len(runs) and listed == runs, "Final summary/run disagreement")
        require(set(summary["replay"]) == set(MODELS)
                and all(r["status"] == "PASS" for r in summary["replay"].values()), "Replay not passed")
        require(summary["status"] in ("DEVELOPMENT_ONLY", "EXECUTION_COMPLETE_REVIEW_REQUIRED"),
                "Unexpected final status")
        sources[summary_path.name] = digest(summary_path)
        sources["progress.json"] = digest(folder / "progress.json")
        status = "EXECUTION COMPLETE — REVIEW REQUIRED"
    if manifest.get("development"):
        status = "DEVELOPMENT ONLY / " + status
    return manifest, runs, sources, status


def paired_seeds(manifest, runs):
    return [s for s in manifest["protocol"]["seeds"] if all((s, m) in runs for m in MODELS)]


def triplet_matrix(arrays):
    """Align by identifiers, never assume storage order means early/middle/late."""
    ids = np.asarray(arrays["triplet_id"])
    variants = np.asarray(arrays["variant"]).astype(str)
    probs = np.asarray(arrays["probability"], dtype=float)
    require(ids.ndim == 1 and variants.shape == probs.shape == ids.shape, "Prediction shape mismatch")
    require(np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all(), "Invalid probabilities")
    ordered, rows = np.unique(ids), []
    for tid in ordered:
        group = np.flatnonzero(ids == tid)
        require(len(group) == 3 and set(variants[group]) == set(POSITIONS), "Incomplete triplet")
        idx = [int(group[np.flatnonzero(variants[group] == v)[0]]) for v in POSITIONS]
        require(len(set(np.asarray(arrays["label"])[idx].tolist())) == 1, "Labels differ in triplet")
        truth = np.asarray(arrays["p_true"])[idx]
        require(np.isfinite(truth).all() and ((truth >= 0) & (truth <= 1)).all()
                and np.all(truth == truth[0]), "True probabilities differ in triplet")
        rows.append(idx)
    return ordered, probs[np.asarray(rows)], np.asarray(arrays["p_true"])[np.asarray(rows)[:, 0]]


def render(folder, output=None):
    folder = Path(folder).resolve()
    manifest, runs, sources, status = collect(folder)
    # Fresh snapshots avoid silently overwriting any previous report or evidence.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path(output).resolve() if output else folder / "visuals" / stamp
    require(not output.exists(), "Report output already exists; choose a new directory")
    output.mkdir(parents=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white", "savefig.facecolor": "white"})
    seeds = manifest["protocol"]["seeds"]
    paired = paired_seeds(manifest, runs)
    figures = []

    def save(fig, name, title, caption):
        fig.suptitle(title + "\n" + status, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0.02, 1, 0.91))
        for ext in ("png", "svg", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        figures.append((name, title, caption))

    fig, ax = plt.subplots(figsize=(10, max(3, 0.38 * len(seeds) + 1.8)))
    grid = np.array([[(s, m) in runs for m in MODELS] for s in seeds], dtype=int)
    ax.imshow(grid, cmap=ListedColormap(["#e8edf2", "#9cd8c1"]), vmin=0, vmax=1, aspect="auto")
    for i in range(len(seeds)):
        for j in range(3):
            ax.text(j, i, "Saved" if grid[i, j] else "Pending", ha="center", va="center")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(len(seeds)), [str(s) for s in seeds])
    ax.set_ylabel("Training seed")
    save(fig, "01_readiness", f"Saved primary runs: {len(runs)} / {3 * len(seeds)}",
         "Saved means a completed, hash-checked primary run. Pending can mean unstarted or currently training. "
         "Replay runs are excluded. The runner saves trajectories at run completion, so this is not a live epoch monitor.")

    for seed in seeds:
        available = [m for m in MODELS if (seed, m) in runs]
        if not available:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for j, model in enumerate(MODELS):
            ax = axes[j]
            ax.set_title(LABELS[j])
            ax.set_xlabel("Epoch")
            if model not in available:
                ax.text(0.5, 0.5, "Run pending", transform=ax.transAxes, ha="center")
                continue
            tr = runs[seed, model]["training"]
            epochs = [x["epoch"] for x in tr["trajectory"]]
            ax.plot(epochs, [x["train_nll"] for x in tr["trajectory"]], "--", color=COLOURS[j], label="Training")
            ax.plot(epochs, [x["val_nll"] for x in tr["trajectory"]], "-o", markersize=3,
                    color=COLOURS[j], label="Validation")
            ax.axvline(tr["best_epoch"], color="#525b65", linestyle=":", label="Selected epoch")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.15)
        axes[0].set_ylabel("Negative log-likelihood (lower is better)")
        save(fig, f"02_training_seed_{seed}", f"Learning curves | seed {seed}",
             "Dashed: training-mode minibatch loss; solid: validation loss in evaluation mode. "
             "These are different measurement conditions. The selected checkpoint uses validation only, never sweep scores.")

    if paired:
        values = np.array([[runs[s, m]["metrics"]["primary_probability_range"] for m in MODELS] for s in paired])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for row in values:
            axes[0].plot(range(3), row, "o-", color="#88939f", alpha=0.6, lw=1)
        axes[0].plot(range(3), values.mean(0), "D-", color="#152b40", lw=2, label="Mean over paired seeds")
        axes[0].set_xticks(range(3), LABELS)
        axes[0].set_ylabel("Mean triplet probability range (lower = more stable)")
        axes[0].set_ylim(bottom=0)
        axes[0].legend(fontsize=8)
        for j, model in enumerate(MODELS):
            quality = [runs[s, model]["metrics"]["macro"]["auroc"] for s in paired]
            axes[1].scatter(values[:, j], quality, color=COLOURS[j], label=LABELS[j], alpha=0.8)
        axes[1].axhline(0.5, color="#88939f", linestyle=":", label="Chance AUROC")
        axes[1].set(xlabel="Probability range (lower = more stable)", ylabel="Macro AUROC (higher is better)", ylim=(0, 1))
        axes[1].legend(fontsize=8)
        save(fig, "03_stability_and_quality", f"Stability with predictive quality | {len(paired)} paired seeds",
             "Each thin line connects the same seed across models. Each scatter point is one model/seed. "
             "A constant predictor has zero range but AUROC 0.5: stability alone cannot establish improvement. "
             "Only seeds with all three completed models appear in comparative plots; no missing run is imputed.")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (a, b) in zip(axes, ((1, 0), (2, 1), (2, 0))):
            d = values[:, a] - values[:, b]
            ax.axvline(0, color="#88939f", linestyle=":")
            ax.scatter(d, range(len(paired)), color=COLOURS[a])
            ax.set_yticks(range(len(paired)), [str(s) for s in paired], fontsize=8)
            ax.set_title(f"{LABELS[a]} minus {LABELS[b]}")
            ax.set_xlabel("Difference in primary probability range")
        save(fig, "04_paired_differences", "Within-seed primary contrasts",
             "Negative values favour the first model for stability only. Zero means no observed difference. "
             "No significance stars, interim hypothesis decisions or new inferential tests are added.")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, metric, label in zip(axes.flat, ("auroc", "accuracy", "nll", "brier"),
                                    ("AUROC (higher is better)", "Accuracy (higher is better)",
                                     "NLL (lower is better)", "Brier score (lower is better)")):
            for j, model in enumerate(MODELS):
                a = np.array([[runs[s, model]["metrics"]["per_position"][v][metric] for v in POSITIONS] for s in paired])
                for row in a:
                    ax.plot(range(3), row, color=COLOURS[j], alpha=0.16, lw=0.8)
                ax.plot(range(3), a.mean(0), "o-", color=COLOURS[j], label=LABELS[j])
            ax.set_xticks(range(3), ["Early (10%)", "Middle (50%)", "Late (90%)"])
            ax.set_title(label)
            ax.grid(alpha=0.15)
            ax.legend(fontsize=8)
            if metric in ("auroc", "accuracy", "brier"):
                ax.set_ylim(0, 1)
        save(fig, "05_position_quality", "Predictive quality as the probe moves",
             "Faint lines show individual paired seeds; bold lines show their arithmetic mean, not confidence bounds. "
             "Position percentages are rounded to discrete token indices. Triplet variants are dependent observations.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for j, model in enumerate(MODELS):
            diag = [runs[s, model]["metrics"].get("diagnostics_by_position", {}) for s in paired]
            if not all(all(v in d for v in POSITIONS) for d in diag):
                continue
            for ax, key, subtract in ((axes[0], "probe_attention_mass", "uniform_probe_mass"),
                                     (axes[1], "signal_retention_mean", "nuisance_retention_mean")):
                if not all(key in d[v] and subtract in d[v] for d in diag for v in POSITIONS):
                    continue
                a = np.array([[d[v][key] - d[v][subtract] for v in POSITIONS] for d in diag])
                for row in a:
                    ax.plot(range(3), row, color=COLOURS[j], alpha=0.16, lw=0.8)
                ax.plot(range(3), a.mean(0), "o-", color=COLOURS[j], label=LABELS[j])
        for ax, title in zip(axes, ("Probe attention minus uniform reference", "Mean signal retention minus nuisance retention")):
            ax.axhline(0, linestyle=":", color="#88939f")
            ax.set_xticks(range(3), POSITIONS)
            ax.set_title(title, fontsize=10)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        save(fig, "06_diagnostics", "Mechanism diagnostics | descriptive, not causal",
             "Attention is averaged over layers, heads and valid queries before reading probe key mass. "
             "Zero attention excess is the uniform-attention reference; zero retention contrast means equal group means. "
             "Baseline has no retention gate. Signal membership is associated with observable event type. "
             "These plots do not prove causal importance or end-to-end information preservation.")

        seed = paired[0]
        predictions = {}
        for model in MODELS:
            with np.load(folder / f"seed_{seed}" / model / "predictions.npz", allow_pickle=False) as a:
                predictions[model] = triplet_matrix(a)
        ids, _, truth = predictions[MODELS[0]]
        for other_ids, _, other_truth in predictions.values():
            require(np.array_equal(ids, other_ids) and np.array_equal(truth, other_truth), "Unmatched model triplets")
        count = min(3, len(ids))
        fig, axes = plt.subplots(1, count, figsize=(4.2 * count, 4), squeeze=False)
        for i, ax in enumerate(axes[0]):
            for j, model in enumerate(MODELS):
                ax.plot(range(3), predictions[model][1][i], "o-", color=COLOURS[j], label=LABELS[j])
            ax.axhline(truth[i], color="#333333", linestyle=":", label="Generator p_true")
            ax.set(ylim=(0, 1), title=f"Triplet {ids[i]}", ylabel="Predicted positive-class probability")
            ax.set_xticks(range(3), POSITIONS)
            ax.legend(fontsize=8)
        save(fig, "07_triplet_examples", f"Same content, moved probe | seed {seed}",
             "First three sorted triplet IDs from the first available complete paired seed, chosen without looking at scores. "
             "Examples illustrate behaviour; all triplets contribute to the primary endpoint. "
             "The dotted line is the generator probability, not an additional model prediction.")

    with (output / "per_seed_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "model", "primary_probability_range", "macro_auroc", "macro_accuracy", "macro_nll", "macro_brier", "best_epoch"])
        for (seed, model), r in runs.items():
            writer.writerow([seed, model, r["metrics"]["primary_probability_range"],
                             *[r["metrics"]["macro"][k] for k in ("auroc", "accuracy", "nll", "brier")], r["training"]["best_epoch"]])
    cards = []
    for name, title, caption in figures:
        encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode()
        cards.append(f'<details open><summary>{html.escape(title)}</summary><p>{html.escape(caption)}</p>'
                     f'<img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}"></details>')
    note = ("No completed paired seeds yet; comparative plots will appear after all three models finish for a seed."
            if not paired else f"Comparative plots use {len(paired)} complete paired seeds out of {len(seeds)} planned.")
    report = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
              '<title>IPAT EXP-03 visual report</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#183044;background:#f4f7fa}'
              'details{background:white;padding:20px;margin:20px 0;border-radius:12px}summary{font-size:20px;font-weight:650;cursor:pointer}'
              'img{width:100%;height:auto}p{line-height:1.6}.banner{padding:18px;background:#fff0cc;border-radius:8px}code{overflow-wrap:anywhere}</style>'
              f'<h1>IPAT · EXP-03</h1><div class="banner"><b>{html.escape(status)}</b><p>{html.escape(note)}</p></div>'
              f'<p>{html.escape(LIMIT)}</p><p>Every chart is descriptive. No experiment is frozen by this report. '
              'Exported PNG, SVG and PDF figures and per_seed_metrics.csv are alongside this self-contained page.</p>'
              + ''.join(cards) + f'<p>Experiment fingerprint: <code>{html.escape(manifest["fingerprint"])}</code></p></html>')
    (output / "report.html").write_text(report, encoding="utf-8")
    provenance = {"report_version": "v0.1", "status": status, "created_utc": stamp,
                  "experiment_fingerprint": manifest["fingerprint"], "input_sha256": sources,
                  "plot_script_sha256": digest(__file__), "completed_primary_runs": len(runs),
                  "paired_seeds": paired, "replays_excluded": True,
                  "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
                  "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
                  "limitations": LIMIT}
    (output / "visual_report_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/exp03/v0.1"))
    parser.add_argument("--output-dir", type=Path, help="New directory; default: results-dir/visuals/UTC-timestamp")
    args = parser.parse_args()
    output = render(args.results_dir, args.output_dir)
    print(f"VISUAL_REPORT_READY: {output / 'report.html'}")
    print("Saved PNG, SVG, PDF, CSV and provenance. Experimental evidence was not modified.")


if __name__ == "__main__":
    main()
