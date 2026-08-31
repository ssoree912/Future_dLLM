"""Build Figure 1 from one real full-cache denoising trajectory.

The selected layer/block is used only for the qualitative heatmap.  The two
right-hand metrics average every layer and generation block in the sample.
Raw matrices and per-layer metrics are saved next to the PNG/PDF so the figure
can be replotted without running the 8B model again.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "model" / "LLaDA-8B-Instruct"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "figure_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--student", default="",
                        help="block-conditioned student checkpoint directory")
    parser.add_argument("--dataset", default="math5s")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prompt-shard", default="",
                        help="override artifacts/prompt_shards/<dataset>/<dataset>-N.pt")
    parser.add_argument("--shard-root",
                        default=str(REPO_ROOT / "artifacts" / "prompt_shards"))
    parser.add_argument("--gen-length", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-prompt-len", type=int, default=None)
    parser.add_argument(
        "--gpu-reserve-gib", type=float, default=2.0,
        help="leave this much free GPU memory for student/cache/activations",
    )
    parser.add_argument("--layer", type=int, default=24,
                        help="zero-based qualitative layer (default: 24)")
    parser.add_argument("--block-index", type=int, default=1,
                        help="zero-based qualitative generation block (default: 1)")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--cache-budget", type=int, default=None,
                        help="exact K; default is floor(candidates * keep ratio)")
    budget.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--pool-kernel", type=int, default=3,
                        help="Sparse-dLLM local max-pool width; 0 disables pooling")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--analysis-input", default="",
                        help="plot an existing analysis.pt without loading the model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.sample_index < 0:
        parser.error("--sample-index must be non-negative")
    if args.block_length < 1:
        parser.error("--block-length must be positive")
    if args.max_seq_len < 1:
        parser.error("--max-seq-len must be positive")
    if args.gpu_reserve_gib < 0:
        parser.error("--gpu-reserve-gib must be non-negative")
    if args.layer < 0 or args.block_index < 0:
        parser.error("--layer and --block-index must be non-negative")
    if args.cache_budget is not None and args.cache_budget < 1:
        parser.error("--cache-budget must be positive")
    if args.cache_budget is None and not 0.0 < args.keep_ratio <= 1.0:
        parser.error("--keep-ratio must be in (0, 1]")
    if args.pool_kernel < 0 or (args.pool_kernel and args.pool_kernel % 2 == 0):
        parser.error("--pool-kernel must be zero or a positive odd integer")
    if not args.analysis_input and not args.student:
        parser.error("--student is required unless --analysis-input is used")
    return args


def topk_metrics(
    predicted_score: torch.Tensor,
    future_score: torch.Tensor,
    cache_budget: int,
) -> tuple[torch.Tensor, float, float]:
    """Return kept indices, Future-Mass@K, and Future-Recall@K."""
    predicted_score = predicted_score.flatten().float()
    future_score = future_score.flatten().float()
    if predicted_score.shape != future_score.shape:
        raise ValueError("predicted and future scores must have the same shape")
    if not 1 <= cache_budget <= future_score.numel():
        raise ValueError("cache budget must be between 1 and candidate count")
    if not torch.isfinite(predicted_score).all() or not torch.isfinite(future_score).all():
        raise ValueError("scores must be finite")
    denominator = float(future_score.sum())
    if denominator <= 0.0:
        raise ValueError("future score must have positive mass")

    keep = torch.topk(predicted_score, cache_budget).indices
    oracle = torch.topk(future_score, cache_budget).indices
    mass = float(future_score.index_select(0, keep).sum()) / denominator
    recall = float(torch.isin(keep, oracle).sum()) / cache_budget
    return keep.sort().values.cpu(), mass, recall


def candidate_regions(block: dict[str, Any]) -> list[dict[str, Any]]:
    """Candidate-space spans after removing the current block itself."""
    prompt = int(block["prompt_length"])
    block_start = int(block["block_start"])
    block_length = int(block["block_length"])
    candidate_count = int(block["candidate_indices"].numel())
    previous = max(0, block_start - prompt)
    future = candidate_count - prompt - previous
    if future < 0:
        raise ValueError("candidate metadata produces a negative future region")

    regions = []
    cursor = 0
    for name, count, color in (
        ("Prompt", prompt, "#D9DEE7"),
        ("Previously completed blocks", previous, "#F2CF8D"),
        ("Future masked blocks", future, "#CFC4E8"),
    ):
        if count:
            regions.append({"name": name, "start": cursor,
                            "end": cursor + count, "count": count,
                            "color": color})
        cursor += count
    if cursor != candidate_count:
        raise ValueError("candidate regions do not cover the candidate axis")
    if block_start + block_length > prompt + previous + block_length + future:
        raise ValueError("current block lies outside the recorded sequence")
    return regions


def _resolve_budget(args: argparse.Namespace, candidate_count: int) -> int:
    if args.cache_budget is not None:
        if args.cache_budget > candidate_count:
            raise SystemExit(
                f"cache budget {args.cache_budget} exceeds {candidate_count} candidates"
            )
        return args.cache_budget
    return max(1, int(candidate_count * args.keep_ratio))


def _token_label(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    text = text.replace("\n", "\\n").replace("\t", "\\t").strip()
    if not text:
        text = f"id={token_id}"
    return text[:18]


@torch.no_grad()
def collect_analysis(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    from transformers import AutoConfig, AutoTokenizer
    from future_dllm import (CustomCache, LLaDAModelLM,
                             load_prompt_utility_student)
    from teacher.extract_teacher import collect
    from teacher.gen_length import resolve as resolve_gen_length

    gen_length = args.gen_length
    if gen_length is None:
        gen_length, source = resolve_gen_length(args.dataset)
        print(f"gen_length {gen_length} from {source}", flush=True)
    if gen_length % args.block_length:
        raise SystemExit("generation length must be a multiple of block length")
    prompt_limit = min(
        args.max_seq_len - gen_length,
        args.max_prompt_len or args.max_seq_len - gen_length,
    )
    if prompt_limit < 1:
        raise SystemExit("generation length leaves no prompt space")

    prompt_path = Path(args.prompt_shard) if args.prompt_shard else (
        Path(args.shard_root) / args.dataset /
        f"{args.dataset}-{args.sample_index}.pt"
    )
    if not prompt_path.is_file():
        raise SystemExit(f"prompt shard not found: {prompt_path}")
    source = torch.load(prompt_path, map_location="cpu", weights_only=False)
    prompt_ids = source["prompt_input_ids"].to(torch.long)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    native_limit = int(getattr(config, "max_sequence_length", args.max_seq_len))
    if args.max_seq_len > native_limit:
        print(f"warning: max_seq_len={args.max_seq_len} exceeds trained context "
              f"{native_limit}", flush=True)
    config.max_sequence_length = args.max_seq_len
    config.block_len, config.keep_ratio = args.block_length, 1.0
    print(f"loading model from {args.model}", flush=True)
    max_memory = None
    if torch.cuda.is_available() and args.gpu_reserve_gib:
        free_bytes, _ = torch.cuda.mem_get_info()
        reserve_bytes = int(args.gpu_reserve_gib * 1024 ** 3)
        usable_bytes = max(1024 ** 3, free_bytes - reserve_bytes)
        max_memory = {torch.cuda.current_device(): usable_bytes, "cpu": "80GiB"}
        print(
            f"device_map budget: {usable_bytes / 1024 ** 3:.1f} GiB GPU + CPU "
            f"({args.gpu_reserve_gib:.1f} GiB GPU reserve)",
            flush=True,
        )
    model = LLaDAModelLM.from_pretrained(
        args.model,
        config=config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    student = load_prompt_utility_student(args.student, model.device).float().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    collect_args = SimpleNamespace(
        prompt_limit=prompt_limit,
        gen_length=gen_length,
        block_length=args.block_length,
        save_attention_rows=True,
    )
    print(f"denoising sample {source.get('sample_id', args.sample_index)!r} with full cache",
          flush=True)
    teacher_records = collect(model, prompt_ids, collect_args)
    layer_count = int(model.config.n_layers)
    pool_kernel = args.pool_kernel or None
    blocks = []

    for index, record in enumerate(teacher_records):
        x_select = record["x_at_block_start"].unsqueeze(0).to(model.device)
        cache = CustomCache(
            n_layers=layer_count,
            device=model.device,
            keep_ratio=1.0,
            cache_scorer=student,
            capture_current_scores=True,
            current_score_pool_kernel=pool_kernel,
        )
        model(x_select, int(record["block_start"]), 1, cache)
        current_score = torch.stack(
            [cache.current_scores[layer].squeeze(0).float().cpu()
             for layer in range(layer_count)]
        )
        predicted_layers = []
        for layer in range(layer_count):
            student_device = next(student.layers[str(layer)].parameters()).device
            candidates = record["candidate_indices"].to(student_device)
            block_indices = torch.arange(
                int(record["block_start"]),
                int(record["block_start"]) + int(record["block_length"]),
                device=student_device,
            )
            predicted_layers.append(student.forward_layer(
                layer,
                cache.layer_hidden_states[layer].to(student_device).float(),
                candidates,
                head="score",
                block_indices=block_indices,
            ).squeeze(0).float().cpu())
        predicted_score = torch.stack(predicted_layers)
        future_rows = record["future_attention_rows"].cpu()
        future_score = record["label_final_rowmax"].float().cpu()
        if not torch.equal(future_rows.float().max(dim=1).values, future_score):
            raise RuntimeError("saved attention rows do not reproduce label_final_rowmax")

        completed_ids = record["completed_block_ids"].tolist()
        blocks.append({
            "block_index": int(record["block_index"]),
            "block_start": int(record["block_start"]),
            "block_length": int(record["block_length"]),
            "prompt_length": int(record["prompt_length"]),
            "gen_length": int(record["gen_length"]),
            "candidate_indices": record["candidate_indices"].cpu(),
            "completed_block_ids": record["completed_block_ids"].cpu(),
            "completed_token_labels": [
                _token_label(tokenizer, int(token_id)) for token_id in completed_ids
            ],
            "x_at_block_start": record["x_at_block_start"].cpu(),
            "current_score": current_score,
            "future_attention_rows": future_rows,
            "future_score": future_score,
            "predicted_score": predicted_score,
        })
        del cache, x_select
        print(f"replayed selection state {index + 1}/{len(teacher_records)}", flush=True)

    sample_id = source.get("sample_id", args.sample_index)
    if isinstance(sample_id, torch.Tensor) and sample_id.numel() == 1:
        sample_id = sample_id.item()
    payload = {
        "format": "future_dllm_figure_1_v1",
        "dataset": args.dataset,
        "sample_id": sample_id,
        "sample_index": args.sample_index,
        "prompt_shard": str(prompt_path.resolve()),
        "model": str(Path(args.model).resolve()),
        "student": str(Path(args.student).resolve()),
        "gen_length": gen_length,
        "block_length": args.block_length,
        "max_seq_len": args.max_seq_len,
        "prompt_limit": prompt_limit,
        "layer_count": layer_count,
        "sparse_dllm_pool_kernel": pool_kernel,
        "blocks": blocks,
    }
    return payload


def attach_budget_results(payload: dict[str, Any], cache_budget: int) -> list[dict[str, Any]]:
    """Attach Top-K sets and metrics to every layer/block; return flat CSV rows."""
    metric_rows = []
    for block in payload["blocks"]:
        current_keeps, ours_keeps, oracle_keeps = [], [], []
        current_masses, ours_masses = [], []
        current_recalls, ours_recalls = [], []
        for layer in range(int(payload["layer_count"])):
            future = block["future_score"][layer].float()
            current_keep, current_mass, current_recall = topk_metrics(
                block["current_score"][layer], future, cache_budget
            )
            ours_keep, ours_mass, ours_recall = topk_metrics(
                block["predicted_score"][layer], future, cache_budget
            )
            oracle_keep = torch.topk(future, cache_budget).indices.sort().values.cpu()
            current_keeps.append(current_keep)
            ours_keeps.append(ours_keep)
            oracle_keeps.append(oracle_keep)
            current_masses.append(current_mass)
            ours_masses.append(ours_mass)
            current_recalls.append(current_recall)
            ours_recalls.append(ours_recall)
            metric_rows.append({
                "block_index": int(block["block_index"]),
                "layer_index": layer,
                "candidate_count": int(future.numel()),
                "cache_budget": cache_budget,
                "current_mass_at_k": current_mass,
                "ours_mass_at_k": ours_mass,
                "current_recall_at_k": current_recall,
                "ours_recall_at_k": ours_recall,
            })
        block["current_keep"] = torch.stack(current_keeps)
        block["ours_keep"] = torch.stack(ours_keeps)
        block["oracle_keep"] = torch.stack(oracle_keeps)
        block["current_mass_at_k"] = torch.tensor(current_masses)
        block["ours_mass_at_k"] = torch.tensor(ours_masses)
        block["current_recall_at_k"] = torch.tensor(current_recalls)
        block["ours_recall_at_k"] = torch.tensor(ours_recalls)
    payload["cache_budget"] = cache_budget
    return metric_rows


def _score_display(score: torch.Tensor) -> np.ndarray:
    values = score.detach().float().cpu().numpy()
    low, high = np.quantile(values, [0.02, 0.98])
    if not math.isfinite(float(high)) or high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _future_display(rows: torch.Tensor) -> np.ndarray:
    values = rows.detach().float().cpu().numpy()
    maxima = np.maximum(values.max(axis=1, keepdims=True), 1e-12)
    return np.sqrt(np.clip(values / maxima, 0.0, 1.0))


def _add_regions(axis: Any, regions: list[dict[str, Any]], labels: bool = False) -> None:
    for region in regions:
        start, end = region["start"], region["end"]
        axis.axvspan(start - 0.5, end - 0.5, color=region["color"], alpha=0.10,
                    linewidth=0)
        if start:
            axis.axvline(start - 0.5, color="#525866", linewidth=0.75, alpha=0.8)
        if labels:
            midpoint = (start + end - 1) / 2
            axis.text(midpoint, 1.16,
                      f"{region['name']}\n(n={region['count']})",
                      transform=axis.get_xaxis_transform(), ha="center", va="bottom",
                      fontsize=8.5, color="#30343B", clip_on=False)


def render_figure(
    payload: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if args.block_index >= len(payload["blocks"]):
        raise SystemExit(
            f"block index {args.block_index} is out of range for "
            f"{len(payload['blocks'])} blocks"
        )
    if args.layer >= int(payload["layer_count"]):
        raise SystemExit(
            f"layer {args.layer} is out of range for {payload['layer_count']} layers"
        )
    block = payload["blocks"][args.block_index]
    candidate_count = int(block["candidate_indices"].numel())
    regions = candidate_regions(block)
    layer = args.layer

    current = _score_display(block["current_score"][layer])[None, :]
    future = _future_display(block["future_attention_rows"][layer])
    ours = _score_display(block["predicted_score"][layer])[None, :]
    current_keep = block["current_keep"][layer].numpy()
    ours_keep = block["ours_keep"][layer].numpy()

    blue = LinearSegmentedColormap.from_list("current", ["#F7FAFC", "#2B6CB0"])
    orange = LinearSegmentedColormap.from_list("future", ["#FFF9F2", "#D95F02"])
    green = LinearSegmentedColormap.from_list("ours", ["#F5FBF8", "#16866A"])

    figure = plt.figure(figsize=(15.2, 8.2), constrained_layout=False)
    grid = figure.add_gridspec(
        3, 2, width_ratios=(6.5, 1.45), height_ratios=(1.0, 4.7, 1.0),
        left=0.105, right=0.975, bottom=0.10, top=0.82, hspace=0.26, wspace=0.18,
    )
    current_axis = figure.add_subplot(grid[0, 0])
    future_axis = figure.add_subplot(grid[1, 0], sharex=current_axis)
    ours_axis = figure.add_subplot(grid[2, 0], sharex=current_axis)
    metric_axis = figure.add_subplot(grid[:, 1])

    for axis, values, cmap, label in (
        (current_axis, current, blue, "Current-State Score"),
        (ours_axis, ours, green, "Ours Prediction"),
    ):
        axis.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap,
                    vmin=0.0, vmax=1.0)
        axis.set_yticks([])
        axis.set_ylabel(label, rotation=0, ha="right", va="center",
                        labelpad=17, fontsize=10, fontweight="bold")
        axis.tick_params(axis="x", length=0)
        for spine in axis.spines.values():
            spine.set_color("#9AA0AA")
            spine.set_linewidth(0.65)
    current_axis.scatter(current_keep, np.full_like(current_keep, 0.72, dtype=float),
                         marker="|", s=45, linewidths=1.1, color="#181A1F",
                         clip_on=False, label="Sparse-dLLM Top-K")
    ours_axis.scatter(ours_keep, np.full_like(ours_keep, 0.72, dtype=float),
                      marker="|", s=45, linewidths=1.1, color="#181A1F",
                      clip_on=False, label="Ours Top-K")
    current_axis.set_ylim(0.86, -0.5)
    ours_axis.set_ylim(0.86, -0.5)

    future_axis.imshow(future, aspect="auto", interpolation="nearest", cmap=orange,
                       vmin=0.0, vmax=1.0)
    future_axis.set_ylabel("Future Answer Usage", rotation=0, ha="right", va="center",
                           labelpad=17, fontsize=10, fontweight="bold")
    token_count = int(block["block_length"])
    tick_count = min(8, token_count)
    ticks = np.unique(np.linspace(0, token_count - 1, tick_count).round().astype(int))
    labels = []
    for tick in ticks:
        token = block["completed_token_labels"][int(tick)].replace("$", "\\$")
        labels.append(f"{tick + 1:02d}  {token}")
    future_axis.set_yticks(ticks, labels, fontsize=7.2)
    future_axis.tick_params(axis="y", length=0, pad=4)
    for spine in future_axis.spines.values():
        spine.set_color("#9AA0AA")
        spine.set_linewidth(0.65)

    _add_regions(current_axis, regions, labels=True)
    _add_regions(future_axis, regions)
    _add_regions(ours_axis, regions)
    current_axis.tick_params(labelbottom=False)
    future_axis.tick_params(labelbottom=False, axis="x", length=0)
    ours_axis.set_xlim(-0.5, candidate_count - 0.5)
    ours_axis.set_xlabel("Cache candidate position  →   (current block excluded)",
                         fontsize=9.5, labelpad=7)
    # Label each region's first candidate and the final candidate.  Showing
    # both sides of a boundary (e.g. 58 and 59) is redundant and overlaps in
    # the paper-width rendering.
    tick_positions = sorted(set(
        [0, candidate_count - 1] + [region["start"] for region in regions]
    ))
    ours_axis.set_xticks(tick_positions, [str(position) for position in tick_positions],
                         fontsize=7.5)

    mass_current = float(np.mean([row["current_mass_at_k"] for row in metric_rows]))
    mass_ours = float(np.mean([row["ours_mass_at_k"] for row in metric_rows]))
    recall_current = float(np.mean([row["current_recall_at_k"] for row in metric_rows]))
    recall_ours = float(np.mean([row["ours_recall_at_k"] for row in metric_rows]))
    actual_keep_ratio = float(payload["actual_keep_ratio"])
    evicted_ratio = 1.0 - actual_keep_ratio
    x = np.arange(2)
    width = 0.34
    metric_axis.axhline(
        1.0, color="#545B66", linestyle=(0, (4, 3)), linewidth=1.1,
        label="Full cache (1.0)", zorder=1,
    )
    current_bars = metric_axis.bar(
        x - width / 2, [mass_current, recall_current], width,
        color="#4C78A8", label="Sparse-dLLM"
    )
    ours_bars = metric_axis.bar(
        x + width / 2, [mass_ours, recall_ours], width,
        color="#1B9E77", label="Ours"
    )
    metric_axis.set_xticks(x, ["Future-\nMass@K", "Future-\nRecall@K"])
    metric_axis.set_ylim(0.0, 1.05)
    metric_axis.set_ylabel("Score (higher is better)")
    metric_axis.set_title(
        f"Keep ratio {payload['requested_keep_ratio']:.1f} "
        f"({actual_keep_ratio:.1%} retained)\n"
        f"All {payload['layer_count']} layers × {len(payload['blocks'])} blocks",
        fontsize=9.5, pad=10,
    )
    metric_axis.grid(axis="y", color="#D8DCE3", linewidth=0.65, alpha=0.8)
    metric_axis.set_axisbelow(True)
    metric_axis.spines[["top", "right"]].set_visible(False)
    metric_axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.10),
                       fontsize=8.5)
    for bars in (current_bars, ours_bars):
        metric_axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    sample_text = str(payload.get("sample_id", payload.get("sample_index", "?")))
    figure.suptitle(
        "Predicting Which Cache Entries the Completed Answer Will Use",
        x=0.50, y=0.965, fontsize=15, fontweight="bold",
    )
    figure.text(
        0.50, 0.915,
        f"{payload['dataset']} sample {sample_text}  ·  block {args.block_index + 1}  ·  "
        f"layer {layer + 1} (index {layer})  ·  keep ratio "
        f"{payload['requested_keep_ratio']:.1f} → K={payload['cache_budget']} / "
        f"{candidate_count} ({actual_keep_ratio:.1%} retained, {evicted_ratio:.1%} evicted)",
        ha="center", va="center", fontsize=9.5, color="#4B515B",
    )
    figure.text(
        0.105, 0.035,
        "Heatmaps are normalized only for display (score strips per panel; future usage per answer row). "
        "Black ticks mark Top-K; the dashed line is full cache. Metrics use raw label_final_rowmax values.",
        ha="left", va="center", fontsize=7.6, color="#5B616B",
    )

    png_path = output_dir / "figure_1.png"
    pdf_path = output_dir / "figure_1.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    selected_row = next(
        row for row in metric_rows
        if row["block_index"] == int(block["block_index"])
        and row["layer_index"] == layer
    )
    return {
        "dataset": payload["dataset"],
        "sample_id": sample_text,
        "sample_index": int(payload.get("sample_index", 0)),
        "qualitative_block_index": int(block["block_index"]),
        "qualitative_layer_index": layer,
        "candidate_count": candidate_count,
        "cache_budget": int(payload["cache_budget"]),
        "requested_keep_ratio": float(payload["requested_keep_ratio"]),
        "actual_keep_ratio": actual_keep_ratio,
        "evicted_ratio": evicted_ratio,
        "averaging_units": len(metric_rows),
        "full_cache_reference": {
            "future_mass": 1.0,
            "future_recall": 1.0,
        },
        "average": {
            "current_mass_at_k": mass_current,
            "ours_mass_at_k": mass_ours,
            "current_recall_at_k": recall_current,
            "ours_recall_at_k": recall_ours,
        },
        "selected_layer_block": {
            key: value for key, value in selected_row.items()
            if key not in {"block_index", "layer_index", "candidate_count", "cache_budget"}
        },
        "regions": regions,
        "figure_png": str(png_path.resolve()),
        "figure_pdf": str(pdf_path.resolve()),
    }


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not payload.get("blocks"):
        raise SystemExit("analysis contains no generation blocks")
    candidate_counts = {
        int(block["candidate_indices"].numel()) for block in payload["blocks"]
    }
    if len(candidate_counts) != 1:
        raise SystemExit("all blocks must use the same candidate count")
    candidate_count = candidate_counts.pop()
    cache_budget = _resolve_budget(args, candidate_count)
    payload["requested_keep_ratio"] = (
        float(args.keep_ratio)
        if args.cache_budget is None
        else cache_budget / candidate_count
    )
    payload["actual_keep_ratio"] = cache_budget / candidate_count
    metric_rows = attach_budget_results(payload, cache_budget)

    analysis_path = output_dir / "analysis.pt"
    torch.save(payload, analysis_path)
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = render_figure(payload, metric_rows, args, output_dir)
    summary["analysis"] = str(analysis_path.resolve())
    summary["metrics_csv"] = str(metrics_path.resolve())
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print(json.dumps(summary["average"], indent=2), flush=True)
    print(f"saved Figure 1 outputs -> {output_dir.resolve()}", flush=True)
    return summary


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.analysis_input:
        payload = torch.load(args.analysis_input, map_location="cpu", weights_only=False)
        if payload.get("format") != "future_dllm_figure_1_v1":
            raise SystemExit(f"unsupported analysis format: {payload.get('format')!r}")
    else:
        payload = collect_analysis(args)
    write_outputs(payload, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
