"""How close is each scorer's ranking to the teacher's future-attention label?

The task tables compare generations; this compares the decision that produced
them -- which cache entries each scorer would keep -- so nothing here depends on
what the model went on to write, and nothing saturates once the budget is loose.

Reads the json ``scripts/baseline_recall.py --out`` writes.

    python visualization/ranking_bars.py --metric recall
    python visualization/ranking_bars.py --metric spearman
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The random ranker is the floor the diagnostic prints (recall 0.099 at keep
# 0.1); it is left out of the bars because plotting it compresses the range the
# three real scorers live in. The caption carries it instead.
ARMS = [
    ("student",  "student (future)",      "#2a78d6"),
    ("current",  "current (this block)",  "#eb6834"),
    ("baseline", "sparse-dLLM (pooled)",  "#1baf7a"),
]
TITLE = {
    "recall": "Recall of the future-attention Top-K",
    "jaccard": "Jaccard of the kept set",
    "mass": "Future-attention mass retained",
    "spearman": "Spearman with the future-attention label",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="",
                    help="baseline_recall output; the newest under "
                         "results/measurements is used when empty")
    ap.add_argument("--metric", default="recall",
                    choices=("recall", "jaccard", "mass", "spearman"))
    ap.add_argument("--keeps", default="0.1 0.5",
                    help="ignored for spearman, which has no budget")
    ap.add_argument("--out", default=str(REPO / "artifacts/ranking"))
    args = ap.parse_args()

    path = args.json or sorted(glob.glob(
        str(REPO / "results/measurements/dream_scorer_ranking_*.json")))[-1]
    payload = json.loads(Path(path).read_text())
    per, ratios = payload["per_domain"], payload["ratios"]
    domains = sorted(per)

    def value(dom, arm, keep=None):
        b = per[dom]
        n = max(1, b["n"])
        if args.metric == "spearman":
            return b[f"{arm}_spearman"] / n
        return b[f"{arm}_{args.metric}"][ratios.index(keep)] / n

    keeps = [None] if args.metric == "spearman" else [float(k) for k in args.keeps.split()]
    for k in keeps:
        if k is not None and k not in ratios:
            raise SystemExit(f"keep {k} is not in the measured ratios {ratios}")

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    import matplotlib.pyplot as plt
    import numpy as np

    top = max(value(d, a, k) for d in domains for a, _, _ in ARMS for k in keeps)
    ylim = min(1.0, math.ceil((top + 0.06) / 0.05) * 0.05)

    fig, axes = plt.subplots(1, len(keeps), figsize=(7.2 * len(keeps), 5.0),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(domains))
    width = 0.26

    for ax, keep in zip(axes, keeps):
        for i, (arm, label, color) in enumerate(ARMS):
            vals = [value(d, arm, keep) for d in domains]
            bars = ax.bar(x + (i - 1) * (width + 0.015), vals, width,
                          label=label, color=color, zorder=3)
            # The aqua sits below 3:1 on white, so every bar carries its value:
            # the number, not the fill, is what has to be legible.
            ax.bar_label(bars, fmt="%.2f", fontsize=10, padding=2, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=14)
        if keep is not None:
            ax.set_title(f"keep {keep:g}", fontsize=17, color="#30343B", pad=8)
        ax.set_ylim(0, ylim)
        ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=15)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#7E8490")

    axes[0].set_ylabel("Agreement with the label", fontsize=16)
    fig.suptitle(TITLE[args.metric], fontsize=19, color="#30343B")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=15, ncol=3,
               loc="outside lower center")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"ranking_{args.metric}_bars"
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    print(f"wrote {out}/{stem}.png and .pdf   (source: {Path(path).name})")


if __name__ == "__main__":
    main()
