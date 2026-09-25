"""How similar to the un-evicted run does each method stay, budget by budget?

Reads the summaries ``scripts/origin_drift.py`` writes -- one directory per
dataset under ``results/drift`` -- and plots agreement with origin against the
prompt-KV retention budget, averaged over the datasets present.

The mean is taken over datasets, not over documents: each dataset contributes
one point per (method, keep) so a long task does not outvote a short one. The
datasets sit at very different absolute levels -- a 128-token dialogue summary
diverges far more readily than a 32-token span answer -- so every dataset is
also drawn as a thin line behind the mean, and the band is their spread. A
mean read without them says less than it looks like it does.

Origin is the reference rather than an arm, so its point is agreement 1.0 at
keep 1.0 by construction; it is drawn to anchor the axis, not as a measurement.

    python visualization/drift_curve.py
    python visualization/drift_curve.py --metric exact_match --out artifacts/drift
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

METHOD_STYLE = {
    "ours":   {"label": "ours (student)", "color": "#1B9E77", "marker": "o"},
    "sparse": {"label": "sparse-dLLM",    "color": "#4C78A8", "marker": "s"},
}
# What the chart is titled with: the similarity being measured, nothing else.
# Everything about the run -- model, checkpoint, datasets, sample count -- is
# caption material in a paper, and repeating it on the axes only shrinks the plot.
METRIC_TITLE = {
    "token_agreement": "Token agreement",
    "first_divergence": "First divergence",
    "token_f1": "Token F1",
    "token_jaccard": "Token Jaccard",
    "exact_match": "Exact match",
}
METRIC_LABEL = {
    "token_agreement": "Token agreement with origin (position-wise)",
    "first_divergence": "First divergence (fraction of origin's answer)",
    "token_f1": "Token F1 with origin (order-blind)",
    "token_jaccard": "Token Jaccard with origin (order-blind)",
    "exact_match": "Exact match with origin",
}


def load(drift_root: Path):
    """dataset -> method -> keep -> metrics, from every summary.json present."""
    out = {}
    for summary in sorted(drift_root.glob("*/summary.json")):
        payload = json.loads(summary.read_text())
        rows = {}
        for row in payload["rows"]:
            rows.setdefault(row["method"], {})[float(row["keep_ratio"])] = row
        out[payload.get("dataset", summary.parent.name)] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drift-root", default=str(REPO / "results/drift"))
    ap.add_argument("--metric", default="token_agreement", choices=tuple(METRIC_LABEL))
    ap.add_argument("--out", default=str(REPO / "artifacts/drift"))
    args = ap.parse_args()

    data = load(Path(args.drift_root))
    if not data:
        raise SystemExit(f"no summary.json under {args.drift_root}; run scripts/origin_drift.py first")

    # Only budgets every dataset covers: a keep ratio one dataset happens to
    # have would otherwise move the mean for reasons that are not the method's.
    keeps = sorted(set.intersection(*[
        {k for m in rows.values() for k in m} for rows in data.values()]))
    methods = [m for m in METHOD_STYLE if all(m in rows for rows in data.values())]
    print(f"datasets: {', '.join(data)}")
    print(f"keeps: {keeps}   methods: {methods}")

    series = {m: {"mean": [], "lo": [], "hi": []} for m in methods}
    per_dataset = {m: {ds: [] for ds in data} for m in methods}
    for m in methods:
        for k in keeps:
            vals = [data[ds][m][k][args.metric] for ds in data]
            for ds, v in zip(data, vals):
                per_dataset[m][ds].append(v)
            series[m]["mean"].append(statistics.fmean(vals))
            series[m]["lo"].append(min(vals))
            series[m]["hi"].append(max(vals))

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    import matplotlib.pyplot as plt

    # Sized for a single column at print scale: the type has to stay legible
    # after the figure is scaled down, so it is set large relative to the axes.
    fig, ax = plt.subplots(figsize=(6.0, 4.4), constrained_layout=True)

    # One line per method and nothing else: the per-dataset detail lives in the
    # csv beside the figure, and the origin anchor is 1.0 by construction rather
    # than a measurement, so neither earns ink here.
    for m in methods:
        style = METHOD_STYLE[m]
        ax.plot(keeps, series[m]["mean"], color=style["color"], marker=style["marker"],
                markersize=8, linewidth=2.8, label=style["label"], zorder=3)

    # Read left to right as the budget tightens, which is the direction the
    # experiment moves in.
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(keeps)
    ax.set_xticklabels([f"{k:g}" for k in keeps])
    ax.minorticks_off()
    ax.set_xlabel("Prompt-KV retention (keep ratio)", fontsize=16)
    ax.set_ylabel("Similarity to origin", fontsize=16)
    ax.tick_params(axis="both", labelsize=15, length=4, color="#7E8490")
    # The top of the axis is the data's own ceiling rounded up to a tick, not 1.0:
    # no metric here comes near 1, and the empty band above the lines only makes
    # the curves shallower than they are.
    top = max(v for m in methods for v in series[m]["mean"])
    ax.set_ylim(0, min(1.0, math.ceil((top + 0.02) / 0.05) * 0.05))
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#7E8490")
    ax.set_title(METRIC_TITLE[args.metric], fontsize=18, color="#30343B", pad=10)
    ax.legend(frameon=False, fontsize=15, loc="lower left", handlelength=1.8)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"drift_{args.metric}"
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor="white")

    with open(out / f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "keep_ratio", "mean", "min", "max"] + list(data))
        for m in methods:
            for i, k in enumerate(keeps):
                w.writerow([m, k, f"{series[m]['mean'][i]:.4f}",
                            f"{series[m]['lo'][i]:.4f}", f"{series[m]['hi'][i]:.4f}"]
                           + [f"{per_dataset[m][ds][i]:.4f}" for ds in data])

    print(f"\n{METRIC_LABEL[args.metric]} -- mean over datasets")
    print("| Method | " + " | ".join(f"keep {k:g}" for k in keeps) + " |")
    print("|---|" + "---:|" * len(keeps))
    for m in methods:
        print(f"| {METHOD_STYLE[m]['label']} | "
              + " | ".join(f"{v:.4f}" for v in series[m]["mean"]) + " |")
    print(f"\nwrote {out}/{stem}.png, .pdf and .csv")


if __name__ == "__main__":
    main()
