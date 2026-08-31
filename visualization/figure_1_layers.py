"""Render layer-grid and layer-averaged Figure 1 heatmaps from analysis.pt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from visualization.figure_1 import (
    attach_budget_results,
    candidate_regions,
    mask_future_attention,
)


METHODS = (
    ("Sparse-dLLM", "current_keep", "current_recall_at_k", "#4C78A8"),
    ("Oracle", "oracle_keep", None, "#6B5CA5"),
    ("Ours", "ours_keep", "ours_recall_at_k", "#1B9E77"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="4,15,24",
                        help="comma-separated zero-based layer indices")
    parser.add_argument("--block-index", type=int, default=4)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--cache-budget", type=int)
    budget.add_argument("--keep-ratio", type=float, default=0.1)
    args = parser.parse_args()
    try:
        args.layers = [int(value.strip()) for value in args.layers.split(",")]
    except ValueError as exc:
        parser.error(f"--layers must contain integers: {exc}")
    if not args.layers or any(layer < 0 for layer in args.layers):
        parser.error("--layers must contain non-negative indices")
    if args.block_index < 0:
        parser.error("--block-index must be non-negative")
    if args.cache_budget is not None and args.cache_budget < 1:
        parser.error("--cache-budget must be positive")
    if args.cache_budget is None and not 0.0 < args.keep_ratio <= 1.0:
        parser.error("--keep-ratio must be in (0, 1]")
    return args


def average_retained_attention(
    future_attention_rows: torch.Tensor,
    keep_indices: torch.Tensor,
) -> torch.Tensor:
    """Average actual retained attention after applying each layer's own Top-K."""
    if future_attention_rows.ndim != 3:
        raise ValueError("future_attention_rows must be [layers, rows, candidates]")
    if keep_indices.ndim != 2 or keep_indices.shape[0] != future_attention_rows.shape[0]:
        raise ValueError("keep_indices must be [layers, K]")
    retained = [
        mask_future_attention(future_attention_rows[layer], keep_indices[layer])
        for layer in range(future_attention_rows.shape[0])
    ]
    return torch.stack(retained).mean(dim=0)


def selection_frequency(
    keep_indices: torch.Tensor,
    candidate_count: int,
) -> torch.Tensor:
    """Fraction of layers that retain each candidate."""
    if keep_indices.ndim != 2:
        raise ValueError("keep_indices must be [layers, K]")
    mask = torch.zeros(
        keep_indices.shape[0], candidate_count, dtype=torch.float32
    )
    mask.scatter_(1, keep_indices.to(torch.long), 1.0)
    return mask.mean(dim=0)


def _color_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if not positive.size:
        return 1.0
    return max(float(np.quantile(positive, 0.995)), float(positive.min()))


def _region_label(name: str) -> str:
    return {
        "Previously completed blocks": "Prev.",
        "Future masked blocks": "Future",
    }.get(name, name)


def _decorate_heatmap(
    axis: Any,
    values: np.ndarray,
    norm: Any,
    strip_values: np.ndarray,
    strip_cmap: Any,
    regions: list[dict[str, Any]],
    overlap: float,
    show_x: bool,
    show_y: bool,
    overlap_prefix: str = "GT Top-K overlap",
) -> Any:
    image = axis.imshow(
        values, aspect="auto", interpolation="nearest", cmap="viridis",
        norm=norm, origin="upper",
    )
    row_count, candidate_count = values.shape
    ticks = np.unique(np.linspace(0, row_count - 1, min(5, row_count)).round().astype(int))
    axis.set_yticks(ticks, [str(int(tick) + 1) for tick in ticks], fontsize=7)
    axis.tick_params(length=2.5, width=0.6)
    if not show_y:
        axis.tick_params(labelleft=False)
    if show_x:
        axis.set_xlabel("Cache candidate position", fontsize=8, labelpad=5)
        axis.set_xticks([0, candidate_count - 1], ["0", str(candidate_count - 1)],
                        fontsize=7)
    else:
        axis.tick_params(axis="x", labelbottom=False)
    axis.set_xlim(-0.5, candidate_count - 0.5)
    axis.text(
        0.5, 1.145, f"{overlap_prefix}: {overlap:.1%}",
        transform=axis.transAxes, ha="center", va="bottom", fontsize=7.7,
        color="#30343B", fontweight="semibold", clip_on=False,
    )
    for spine in axis.spines.values():
        spine.set_color("#7E8490")
        spine.set_linewidth(0.65)

    strip_axis = axis.inset_axes([0.0, 1.018, 1.0, 0.060])
    strip_axis.imshow(
        strip_values[None, :], aspect="auto", interpolation="nearest",
        cmap=strip_cmap, vmin=0.0, vmax=1.0,
    )
    strip_axis.set_axis_off()
    for region in regions:
        start, end = int(region["start"]), int(region["end"])
        axis.axvline(start - 0.5, color="white", linewidth=0.55,
                    linestyle=(0, (2, 2)), alpha=0.85)
        axis.text(
            (start + end - 1) / 2, 1.085, _region_label(region["name"]),
            transform=axis.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=6.2, color="#3F4650", clip_on=False,
        )
    return image


def _save(figure: Any, output_dir: Path, stem: str) -> dict[str, str]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    return {"png": str(png.resolve()), "pdf": str(pdf.resolve())}


def render_layer_grid(
    block: dict[str, Any],
    layers: list[int],
    regions: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    raw_rows = block["future_attention_rows"].float()
    candidate_count = raw_rows.shape[-1]
    color_max = _color_max(raw_rows[layers].cpu().numpy())
    norm = PowerNorm(gamma=0.40, vmin=0.0, vmax=color_max, clip=True)
    figure, axes = plt.subplots(
        len(layers), len(METHODS), figsize=(11.1, 3.25 * len(layers)),
        squeeze=False, gridspec_kw={"hspace": 0.34, "wspace": 0.18},
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.91)
    image = None
    for row, layer in enumerate(layers):
        for column, (method, keep_key, recall_key, color) in enumerate(METHODS):
            keep = block[keep_key][layer]
            values = mask_future_attention(raw_rows[layer], keep).cpu().numpy()
            strip = np.zeros(candidate_count, dtype=np.float32)
            strip[np.asarray(keep, dtype=np.int64)] = 1.0
            strip_cmap = LinearSegmentedColormap.from_list(
                f"{method}-{layer}", ["#E5E7EB", color]
            )
            overlap = 1.0 if recall_key is None else float(block[recall_key][layer])
            image = _decorate_heatmap(
                axes[row, column], values, norm, strip, strip_cmap, regions,
                overlap=overlap, show_x=row == len(layers) - 1,
                show_y=column == 0,
            )
            if row == 0:
                axes[row, column].text(
                    0.5, 1.255, method, transform=axes[row, column].transAxes,
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                    color="#242932", clip_on=False,
                )
        axes[row, 0].text(
            -0.20, 0.5, f"Layer index {layer}",
            transform=axes[row, 0].transAxes, rotation=90,
            ha="center", va="center", fontsize=9.0, fontweight="semibold",
            color="#30343B", clip_on=False,
        )
    if image is not None:
        colorbar = figure.colorbar(
            image, ax=axes.ravel().tolist(), orientation="horizontal",
            fraction=0.025, pad=0.075, aspect=55,
        )
        colorbar.set_ticks(np.linspace(0.0, color_max, 4))
        colorbar.ax.tick_params(labelsize=7, length=2)
        colorbar.set_label("Actual completed-answer attention retained after eviction",
                           fontsize=8)
    layer_suffix = "_".join(f"{layer:02d}" for layer in layers)
    return _save(figure, output_dir, f"figure_1_layers_{layer_suffix}")


def render_layer_average(
    block: dict[str, Any],
    regions: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    raw_rows = block["future_attention_rows"].float()
    candidate_count = raw_rows.shape[-1]
    averages = []
    frequencies = []
    overlaps = []
    for _, keep_key, recall_key, _ in METHODS:
        keep = block[keep_key]
        averages.append(average_retained_attention(raw_rows, keep).cpu().numpy())
        frequencies.append(selection_frequency(keep, candidate_count).cpu().numpy())
        overlaps.append(1.0 if recall_key is None else float(block[recall_key].mean()))
    unmasked_mean = raw_rows.mean(dim=0).cpu().numpy()
    color_max = _color_max(unmasked_mean)
    norm = PowerNorm(gamma=0.40, vmin=0.0, vmax=color_max, clip=True)
    figure, axes = plt.subplots(1, 3, figsize=(11.1, 4.05), squeeze=False)
    axes = axes[0]
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.83,
                           wspace=0.18)
    image = None
    for column, ((method, _, _, color), values, frequency, overlap) in enumerate(zip(
        METHODS, averages, frequencies, overlaps,
    )):
        strip_cmap = LinearSegmentedColormap.from_list(
            f"mean-{method}", ["#E5E7EB", color]
        )
        image = _decorate_heatmap(
            axes[column], values, norm, frequency, strip_cmap, regions,
            overlap=overlap, show_x=True, show_y=column == 0,
            overlap_prefix="Mean GT Top-K overlap",
        )
        axes[column].text(
            0.5, 1.255, method, transform=axes[column].transAxes,
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            color="#242932", clip_on=False,
        )
    axes[0].set_ylabel("Completed answer token", fontsize=8.5)
    if image is not None:
        colorbar = figure.colorbar(
            image, ax=axes.tolist(), orientation="horizontal",
            fraction=0.055, pad=0.16, aspect=45,
        )
        colorbar.set_ticks(np.linspace(0.0, color_max, 4))
        colorbar.ax.tick_params(labelsize=7, length=2)
        colorbar.set_label(
            "Actual retained attention averaged after layer-wise Top-K eviction",
            fontsize=8,
        )
    return _save(figure, output_dir, "figure_1_layer_average")


def main() -> int:
    args = parse_args()
    payload = torch.load(args.analysis_input, map_location="cpu", weights_only=False)
    if payload.get("format") != "future_dllm_figure_1_v1":
        raise SystemExit(f"unsupported analysis format: {payload.get('format')!r}")
    if args.block_index >= len(payload["blocks"]):
        raise SystemExit("--block-index is out of range")
    layer_count = int(payload["layer_count"])
    if any(layer >= layer_count for layer in args.layers):
        raise SystemExit(f"--layers must be below {layer_count}")
    block = payload["blocks"][args.block_index]
    candidate_count = int(block["candidate_indices"].numel())
    cache_budget = (
        args.cache_budget if args.cache_budget is not None
        else max(1, int(candidate_count * args.keep_ratio))
    )
    if cache_budget > candidate_count:
        raise SystemExit("cache budget exceeds candidate count")
    attach_budget_results(payload, cache_budget)
    regions = candidate_regions(block)

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_files = render_layer_grid(block, args.layers, regions, output_dir)
    plt.close("all")
    average_files = render_layer_average(block, regions, output_dir)
    plt.close("all")
    summary = {
        "dataset": payload["dataset"],
        "sample_id": str(payload.get("sample_id", payload.get("sample_index", "?"))),
        "block_index": int(block["block_index"]),
        "layers": args.layers,
        "layer_count": layer_count,
        "candidate_count": candidate_count,
        "cache_budget": cache_budget,
        "requested_keep_ratio": (
            float(args.keep_ratio) if args.cache_budget is None
            else cache_budget / candidate_count
        ),
        "actual_keep_ratio": cache_budget / candidate_count,
        "layer_grid": grid_files,
        "layer_average": average_files,
        "average_definition": (
            "apply each layer's own Top-K mask to that layer's future attention, "
            "then average retained attention across layers"
        ),
        "selected_layer_overlap": {
            str(layer): {
                "sparse": float(block["current_recall_at_k"][layer]),
                "oracle": 1.0,
                "ours": float(block["ours_recall_at_k"][layer]),
            }
            for layer in args.layers
        },
        "mean_layer_overlap": {
            "sparse": float(block["current_recall_at_k"].mean()),
            "oracle": 1.0,
            "ours": float(block["ours_recall_at_k"].mean()),
        },
    }
    summary_path = output_dir / "summary_layers.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
