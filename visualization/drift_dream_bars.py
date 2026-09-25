"""Current-block attention against the oracle's own-block label, per task.

Bars rather than lines: only two budgets were run, so there is no curve to
trace -- the comparison is between two arms within each task, and a grouped bar
puts that pair side by side. One panel per keep ratio, shared scale.

    python visualization/drift_dream_bars.py
    python visualization/drift_dream_bars.py --metric token_agreement
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Tasks in the order they were run, which also groups them by kind: reasoning,
# code, then the three LongBench tasks.
ORDER = ["gsm8k", "math", "humaneval", "mbpp", "qasper", "musique", "samsum"]
ARMS = [
    ("current", "current (this block)", "#2a78d6"),
    ("oracle",  "oracle (future label)", "#eb6834"),
]
METRIC_TITLE = {
    "token_agreement": "Token agreement",
    "first_divergence": "First divergence",
    "token_f1": "Token F1",
    "token_jaccard": "Token Jaccard",
    "exact_match": "Exact match",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drift-root", default=str(REPO / "results/drift_dream"))
    ap.add_argument("--metric", default="token_f1", choices=tuple(METRIC_TITLE))
    ap.add_argument("--keeps", default="0.5 0.1")
    ap.add_argument("--out", default=str(REPO / "artifacts/drift_dream"))
    args = ap.parse_args()

    keeps = [float(k) for k in args.keeps.split()]
    root = Path(args.drift_root)
    data = {}
    for ds in ORDER:
        summary = root / ds / "summary.json"
        if not summary.exists():
            continue
        rows = json.loads(summary.read_text())["rows"]
        data[ds] = {(r["method"], float(r["keep_ratio"])): r[args.metric] for r in rows}
    if not data:
        raise SystemExit(f"no summary.json under {root}")
    tasks = [ds for ds in ORDER if ds in data]

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    import matplotlib.pyplot as plt
    import numpy as np

    top = max(v for d in data.values() for v in d.values())
    ylim = min(1.0, math.ceil((top + 0.02) / 0.05) * 0.05)

    fig, axes = plt.subplots(1, len(keeps), figsize=(6.4 * len(keeps), 4.8),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(tasks))
    # A gap between the two bars of a pair, and a wider one between tasks, so
    # the pairing is read from the spacing before the colour is consulted.
    width = 0.38

    for ax, keep in zip(axes, keeps):
        for i, (arm, label, color) in enumerate(ARMS):
            vals = [data[ds][(arm, keep)] for ds in tasks]
            ax.bar(x + (i - 0.5) * (width + 0.02), vals, width,
                   label=label, color=color, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=14)
        ax.set_title(f"keep {keep:g}", fontsize=17, color="#30343B", pad=8)
        ax.set_ylim(0, ylim)
        ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=15)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#7E8490")

    axes[0].set_ylabel("Similarity to origin", fontsize=16)
    fig.suptitle(METRIC_TITLE[args.metric], fontsize=19, color="#30343B")
    # Below the panels: the bars fill them at every budget, so a legend inside
    # sits on a task's bar, and the space above is the title's.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=15, ncol=2,
               loc="outside lower center")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"drift_dream_{args.metric}_bars"
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    print(f"wrote {out}/{stem}.png and .pdf")


if __name__ == "__main__":
    main()
