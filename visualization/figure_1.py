"""Build Figure 1 from one real full-cache denoising trajectory.

The selected layer/block is used only for the qualitative heatmap.  The two
right-hand metrics average every layer and generation block in the sample.
Raw matrices and per-layer metrics are saved next to the PNG/PDF so the figure
can be replotted without running the 8B model again.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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
    parser.add_argument("--teacher-shard", default="",
                        help="optional stored row-max shard; auto-detected when present")
    parser.add_argument("--teacher-root",
                        default=str(REPO_ROOT / "artifacts" / "teacher"))
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
    parser.add_argument(
        "--summary-granularity", choices=("token", "sentence"), default="token",
        help="qualitative future-usage rows (sentence uses row-wise max)",
    )
    parser.add_argument(
        "--qualitative-scope", choices=("all", "prompt"), default="all",
        help="candidate columns shown; metrics always use the full candidate pool",
    )
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


def _resolve_prompt_shard(args: argparse.Namespace) -> Path:
    if args.prompt_shard:
        return Path(args.prompt_shard)
    shard_dir = Path(args.shard_root) / args.dataset
    direct = shard_dir / f"{args.dataset}-{args.sample_index}.pt"
    if direct.is_file():
        return direct
    # Some builders intentionally use compact sample IDs (multi_news ->
    # multinews-N). Resolve by numeric suffix without baking those aliases into
    # the visualization path.
    matches = sorted(shard_dir.glob(f"*-{args.sample_index}.pt"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return direct
    raise SystemExit(
        f"multiple prompt shards end in -{args.sample_index}.pt under {shard_dir}; "
        "pass --prompt-shard explicitly"
    )


def _resolve_teacher_shard(args: argparse.Namespace) -> Path | None:
    if args.teacher_shard:
        path = Path(args.teacher_shard)
        if not path.is_file():
            raise SystemExit(f"teacher shard not found: {path}")
        return path
    shard_dir = Path(args.teacher_root) / args.dataset
    matches = sorted(shard_dir.glob(f"*-{args.sample_index}.pt"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(
            f"multiple teacher shards end in -{args.sample_index}.pt under "
            f"{shard_dir}; pass --teacher-shard explicitly"
        )
    return None


def _token_span_for_chars(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    indices = [
        index for index, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end and end > start
    ]
    if not indices:
        raise ValueError(f"no tokens overlap character span [{char_start}, {char_end})")
    return indices[0], indices[-1] + 1


def _article_regions(tokenizer: Any, prompt_ids: torch.Tensor) -> list[dict[str, Any]]:
    """Map MultiNews ``|||||`` document separators to prompt-token spans."""
    ids = prompt_ids.tolist()
    text = tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    separator_matches = list(re.finditer(r"\|\|\|\|\|", text))
    if not separator_matches:
        return []
    news_marker = "\n\nNews:\n"
    summary_marker = "\n\nNow, write"
    if news_marker not in text or summary_marker not in text:
        return []
    document_start = text.index(news_marker) + len(news_marker)
    document_end = text.rindex(summary_marker)
    separators = [
        match for match in separator_matches
        if document_start <= match.start() < document_end
    ]
    char_spans = []
    cursor = document_start
    for match in separators:
        char_spans.append((cursor, match.start()))
        cursor = match.end()
    char_spans.append((cursor, document_end))

    encoding = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True
    )
    if encoding.input_ids != ids:
        raise RuntimeError("decoded prompt no longer maps to its original token IDs")
    offsets = [tuple(offset) for offset in encoding.offset_mapping]
    regions = []
    for index, (start, end) in enumerate(char_spans):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        token_start, token_end = _token_span_for_chars(offsets, start, end)
        regions.append({
            "name": f"Article {index + 1}",
            "start": token_start,
            "end": token_end,
            "count": token_end - token_start,
        })
    return regions


def _summary_sentence_groups(
    tokenizer: Any, completed_ids: torch.Tensor
) -> list[dict[str, Any]]:
    """Sentence-like contiguous token groups for one completed generation block."""
    ids = completed_ids.tolist()
    text = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    # Prefix decoding gives a stable token-row -> decoded-character boundary
    # even when this block begins halfway through a word or sentence.
    token_char_ends = [0] + [
        len(tokenizer.decode(
            ids[:end], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ))
        for end in range(1, len(ids) + 1)
    ]
    boundary_tokens = []
    for match in re.finditer(r"[.!?](?:[\"'”’\)]{0,2})(?=\s|$)", text):
        token_end = bisect.bisect_left(token_char_ends, match.end())
        previous = boundary_tokens[-1] if boundary_tokens else 0
        # Avoid turning short abbreviations into their own display row.
        if token_end - previous >= 3:
            boundary_tokens.append(min(token_end, len(ids)))
    if not boundary_tokens or boundary_tokens[-1] != len(ids):
        boundary_tokens.append(len(ids))

    groups = []
    start = 0
    for end in boundary_tokens:
        if end <= start:
            continue
        sentence = tokenizer.decode(
            ids[start:end], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        sentence = " ".join(sentence.replace("\n", " ").split())
        if len(sentence) > 38:
            sentence = sentence[:35].rstrip() + "…"
        groups.append({
            "start": start,
            "end": end,
            "label": f"S{len(groups) + 1}  {sentence or '(empty)'}",
        })
        start = end
    return groups


def aggregate_future_rows(
    future_rows: torch.Tensor, groups: list[dict[str, Any]]
) -> torch.Tensor:
    """Apply the teacher's row-wise max within each summary sentence."""
    if future_rows.ndim != 2:
        raise ValueError("future_rows must have shape [answer_token, candidate]")
    if not groups:
        return future_rows
    aggregated = []
    cursor = 0
    for group in groups:
        start, end = int(group["start"]), int(group["end"])
        if start != cursor or not start < end <= future_rows.size(0):
            raise ValueError("sentence groups must contiguously cover answer rows")
        aggregated.append(future_rows[start:end].max(dim=0).values)
        cursor = end
    if cursor != future_rows.size(0):
        raise ValueError("sentence groups do not cover every answer row")
    return torch.stack(aggregated)


def _reveal_from_logits(
    logits: torch.Tensor,
    model_input: torch.Tensor,
    target: torch.Tensor,
    reveal_count: int,
    block_end: int | None = None,
) -> None:
    mask = model_input == 126336
    prediction = torch.argmax(logits, dim=-1)
    confidence = torch.squeeze(torch.gather(
        F.softmax(logits, dim=-1), -1, prediction.unsqueeze(-1)
    ), -1)
    if block_end is not None:
        confidence[:, block_end:] = -float("inf")
    prediction = torch.where(mask, prediction, target)
    confidence = torch.where(
        mask, confidence, torch.full_like(confidence, -float("inf"))
    )
    if reveal_count:
        reveal = torch.topk(confidence[0], k=reveal_count).indices
        target[0, reveal] = prediction[0, reveal]


@torch.no_grad()
def _complete_last_teacher_block(
    model: Any,
    cache: Any,
    record: dict[str, Any],
    x_select: torch.Tensor,
    selection_logits: torch.Tensor,
) -> torch.Tensor:
    """Finish the only block whose completed tokens are absent from the next record."""
    from future_dllm import get_num_transfer_tokens

    x = x_select.clone()
    block_start = int(record["block_start"])
    block_length = int(record["block_length"])
    block_end = block_start + block_length
    steps = int(record["steps_per_block"])
    initial_mask = torch.ones(
        (1, block_length), dtype=torch.bool, device=x.device
    )
    transfer = get_num_transfer_tokens(initial_mask, steps)

    _reveal_from_logits(
        selection_logits, x, x, int(transfer[0, 1]), block_end=block_end
    )
    for step in range(2, steps):
        block = x[:, block_start:block_end]
        logits = model(block, block_start, 2, cache).logits
        _reveal_from_logits(logits, block, block, int(transfer[0, step]))
    if (x[:, block_start:block_end] == 126336).any():
        raise RuntimeError("last teacher block did not finish decoding")
    return x[0, block_start:block_end].clone()


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

    prompt_path = _resolve_prompt_shard(args)
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
    truncated_prompt_ids = prompt_ids[-prompt_limit:]
    article_regions = _article_regions(tokenizer, truncated_prompt_ids)
    if article_regions:
        print(
            "prompt articles: "
            + " | ".join(
                f"{region['name']}={region['count']} tokens"
                for region in article_regions
            ),
            flush=True,
        )

    collect_args = SimpleNamespace(
        prompt_limit=prompt_limit,
        gen_length=gen_length,
        block_length=args.block_length,
        save_attention_rows=True,
    )
    teacher_path = _resolve_teacher_shard(args)
    replay_stored_teacher = teacher_path is not None
    if teacher_path is not None:
        teacher_payload = torch.load(
            teacher_path, map_location="cpu", weights_only=False
        )
        teacher_records = teacher_payload.get("blocks") or []
        if not teacher_records:
            raise SystemExit(f"teacher shard contains no blocks: {teacher_path}")
        first_record = teacher_records[0]
        if (int(first_record.get("gen_length", -1)) != gen_length
                or int(first_record.get("block_length", -1)) != args.block_length
                or len(teacher_records) != gen_length // args.block_length
                or int(first_record["prompt_length"])
                != min(prompt_ids.numel(), prompt_limit)):
            raise SystemExit(
                f"teacher shard shape does not match requested prompt/generation: "
                f"{teacher_path}"
            )
        print(
            f"replaying stored row-max teacher shard {teacher_path} "
            "(only the last block is denoised)",
            flush=True,
        )
    else:
        print(
            f"denoising sample {source.get('sample_id', args.sample_index)!r} "
            "with full cache",
            flush=True,
        )
        teacher_records = collect(model, prompt_ids, collect_args)
    layer_count = int(model.config.n_layers)
    pool_kernel = args.pool_kernel or None
    blocks = []
    replay_rowmax_errors = []

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
        selection_logits = model(
            x_select, int(record["block_start"]), 1, cache
        ).logits
        if "future_attention_rows" in record:
            completed_block_ids = record["completed_block_ids"].cpu()
            future_rows = record["future_attention_rows"].cpu()
        else:
            if index + 1 < len(teacher_records):
                next_state = teacher_records[index + 1]["x_at_block_start"]
                block_start = int(record["block_start"])
                block_end = block_start + int(record["block_length"])
                completed_block_ids = next_state[block_start:block_end].clone().cpu()
            else:
                completed_block_ids = _complete_last_teacher_block(
                    model, cache, record, x_select, selection_logits
                ).cpu()
            cache.capture_rows = True
            model(
                completed_block_ids.unsqueeze(0).to(model.device),
                int(record["block_start"]), 2, cache,
            )
            future_rows = torch.stack([
                cache.pending_rows[layer].to(torch.float16).cpu()
                for layer in range(layer_count)
            ])
            cache.capture_rows = False
            cache.pending_rows.clear()
        del selection_logits
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
        future_score = record["label_final_rowmax"].float().cpu()
        replayed_rowmax = future_rows.float().max(dim=1).values
        rowmax_error = (replayed_rowmax - future_score).abs()
        rowmax_exact = torch.equal(replayed_rowmax, future_score)
        if not rowmax_exact and not replay_stored_teacher:
            raise RuntimeError(
                "fresh attention rows do not reproduce label_final_rowmax"
            )
        replay_rowmax_errors.append({
            "block_index": int(record["block_index"]),
            "exact": rowmax_exact,
            "mean_abs_error": float(rowmax_error.mean()),
            "max_abs_error": float(rowmax_error.max()),
        })

        completed_ids = completed_block_ids.tolist()
        sentence_groups = _summary_sentence_groups(
            tokenizer, completed_block_ids
        )
        blocks.append({
            "block_index": int(record["block_index"]),
            "block_start": int(record["block_start"]),
            "block_length": int(record["block_length"]),
            "prompt_length": int(record["prompt_length"]),
            "gen_length": int(record["gen_length"]),
            "candidate_indices": record["candidate_indices"].cpu(),
            "completed_block_ids": completed_block_ids,
            "completed_token_labels": [
                _token_label(tokenizer, int(token_id)) for token_id in completed_ids
            ],
            "completed_sentence_groups": sentence_groups,
            "x_at_block_start": record["x_at_block_start"].cpu(),
            "current_score": current_score,
            "future_attention_rows": future_rows,
            "future_attention_rowmax": replayed_rowmax,
            "future_score": future_score,
            "future_rows_match_teacher_exactly": rowmax_exact,
            "predicted_score": predicted_score,
        })
        del cache, x_select
        print(f"replayed selection state {index + 1}/{len(teacher_records)}", flush=True)

    if replay_stored_teacher:
        print(
            "teacher replay row-max difference: mean MAE "
            f"{np.mean([item['mean_abs_error'] for item in replay_rowmax_errors]):.3e}, "
            "max error "
            f"{max(item['max_abs_error'] for item in replay_rowmax_errors):.3e} "
            "(stored label_final_rowmax remains the metric oracle)",
            flush=True,
        )

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
        "article_regions": article_regions,
        "teacher_shard": str(teacher_path.resolve()) if teacher_path else None,
        "teacher_replay": replay_stored_teacher,
        "teacher_replay_rowmax_errors": replay_rowmax_errors,
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


def _add_regions(
    axis: Any,
    regions: list[dict[str, Any]],
    labels: bool = False,
    label_names: set[str] | None = None,
) -> None:
    for region in regions:
        start, end = region["start"], region["end"]
        axis.axvspan(start - 0.5, end - 0.5, color=region["color"], alpha=0.10,
                    linewidth=0)
        if start:
            axis.axvline(start - 0.5, color="#525866", linewidth=0.75, alpha=0.8)
        if labels and (label_names is None or region["name"] in label_names):
            midpoint = (start + end - 1) / 2
            display_name = {
                "Previously completed blocks": "Prev.",
                "Future masked blocks": "Future",
            }.get(region["name"], region["name"])
            axis.text(midpoint, 1.16,
                      f"{display_name}\n(n={region['count']})",
                      transform=axis.get_xaxis_transform(), ha="center", va="bottom",
                      fontsize=8.5, color="#30343B", clip_on=False)


def _add_article_regions(
    axis: Any, article_regions: list[dict[str, Any]], labels: bool = False
) -> None:
    colors = ("#D9EAF3", "#E2F0D9", "#FCE4D6", "#E4DFEC", "#FFF2CC")
    for index, region in enumerate(article_regions):
        start, end = int(region["start"]), int(region["end"])
        axis.axvspan(start - 0.5, end - 0.5, color=colors[index % len(colors)],
                    alpha=0.14, linewidth=0)
        axis.axvline(start - 0.5, color="#3F4650", linewidth=0.65,
                     linestyle=(0, (2, 2)), alpha=0.85)
        axis.axvline(end - 0.5, color="#3F4650", linewidth=0.65,
                     linestyle=(0, (2, 2)), alpha=0.85)
        if labels:
            midpoint = (start + end - 1) / 2
            axis.text(
                midpoint, 1.16,
                f"{region['name']}\n(n={region['count']})",
                transform=axis.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8.0, color="#30343B", clip_on=False,
            )


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
    if args.qualitative_scope == "prompt":
        display_candidate_count = int(block["prompt_length"])
        display_regions = [region for region in regions if region["name"] == "Prompt"]
    else:
        display_candidate_count = candidate_count
        display_regions = regions
    display_slice = slice(0, display_candidate_count)
    layer = args.layer

    current = _score_display(
        block["current_score"][layer][display_slice]
    )[None, :]
    raw_future_rows = block["future_attention_rows"][layer][:, display_slice]
    sentence_groups = block.get("completed_sentence_groups") or []
    if args.summary_granularity == "sentence":
        if not sentence_groups:
            raise SystemExit(
                "analysis has no completed_sentence_groups; regenerate it before "
                "using --summary-granularity sentence"
            )
        display_future_rows = aggregate_future_rows(raw_future_rows, sentence_groups)
        future_row_labels = [group["label"] for group in sentence_groups]
        future_ylabel = "Future Answer Usage\n(sentence-wise max)"
    else:
        display_future_rows = raw_future_rows
        future_row_labels = [
            f"{index + 1:02d}  {label}"
            for index, label in enumerate(block["completed_token_labels"])
        ]
        future_ylabel = "Future Answer Usage"
    future = _future_display(display_future_rows)
    ours = _score_display(
        block["predicted_score"][layer][display_slice]
    )[None, :]
    current_keep = block["current_keep"][layer].numpy()
    ours_keep = block["ours_keep"][layer].numpy()
    current_keep = current_keep[current_keep < display_candidate_count]
    ours_keep = ours_keep[ours_keep < display_candidate_count]

    blue = LinearSegmentedColormap.from_list("current", ["#F7FAFC", "#2B6CB0"])
    orange = LinearSegmentedColormap.from_list("future", ["#FFF9F2", "#D95F02"])
    green = LinearSegmentedColormap.from_list("ours", ["#F5FBF8", "#16866A"])

    figure = plt.figure(figsize=(15.2, 8.2), constrained_layout=False)
    left_margin = 0.155 if args.summary_granularity == "sentence" else 0.105
    grid = figure.add_gridspec(
        3, 2, width_ratios=(6.5, 1.45), height_ratios=(1.0, 4.7, 1.0),
        left=left_margin, right=0.975, bottom=0.10, top=0.82,
        hspace=0.26, wspace=0.18,
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
    future_axis.set_ylabel(future_ylabel, rotation=0, ha="right", va="center",
                           labelpad=17, fontsize=10, fontweight="bold",
                           multialignment="center")
    row_count = len(future_row_labels)
    tick_count = min(8, row_count)
    ticks = np.unique(np.linspace(0, row_count - 1, tick_count).round().astype(int))
    labels = []
    for tick in ticks:
        labels.append(future_row_labels[int(tick)].replace("$", "\\$"))
    future_axis.set_yticks(ticks, labels, fontsize=7.2)
    future_axis.tick_params(axis="y", length=0, pad=4)
    for spine in future_axis.spines.values():
        spine.set_color("#9AA0AA")
        spine.set_linewidth(0.65)

    article_regions = payload.get("article_regions") or []
    if article_regions:
        tail_names = {"Previously completed blocks", "Future masked blocks"}
        _add_regions(current_axis, display_regions, labels=True, label_names=tail_names)
        _add_article_regions(current_axis, article_regions, labels=True)
    else:
        _add_regions(current_axis, display_regions, labels=True)
    _add_regions(future_axis, display_regions)
    _add_regions(ours_axis, display_regions)
    if article_regions:
        _add_article_regions(future_axis, article_regions)
        _add_article_regions(ours_axis, article_regions)
    current_axis.tick_params(labelbottom=False)
    future_axis.tick_params(labelbottom=False, axis="x", length=0)
    ours_axis.set_xlim(-0.5, display_candidate_count - 0.5)
    if args.qualitative_scope == "prompt":
        x_label = "Prompt token position  →   (global Top-K also includes response suffix)"
    else:
        x_label = "Cache candidate position  →   (current block excluded)"
    ours_axis.set_xlabel(x_label, fontsize=9.5, labelpad=7)
    # Label each region's first candidate and the final candidate.  Showing
    # both sides of a boundary (e.g. 58 and 59) is redundant and overlaps in
    # the paper-width rendering.
    tick_candidates = sorted(set(
        [0, display_candidate_count - 1]
        + [region["start"] for region in display_regions]
    ))
    tick_positions = []
    minimum_tick_gap = max(1, int(display_candidate_count * 0.04))
    for position in tick_candidates:
        if (not tick_positions or position - tick_positions[-1] >= minimum_tick_gap
                or position == display_candidate_count - 1):
            tick_positions.append(position)
    if (len(tick_positions) >= 2
            and tick_positions[-1] - tick_positions[-2] < minimum_tick_gap):
        tick_positions.pop(-2)
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
        f"{candidate_count} ({actual_keep_ratio:.1%} retained, {evicted_ratio:.1%} evicted)  ·  "
        f"{args.summary_granularity} rows  ·  {args.qualitative_scope} scope",
        ha="center", va="center", fontsize=9.5, color="#4B515B",
    )
    figure.text(
        left_margin, 0.035,
        "Heatmaps are normalized only for display. Black ticks mark the global Top-K; the dashed line is full cache. "
        + ("Qualitative columns crop to prompt articles; metrics use all cache candidates."
           if args.qualitative_scope == "prompt"
           else "Metrics use raw label_final_rowmax values over all cache candidates."),
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
    replay_errors = payload.get("teacher_replay_rowmax_errors") or []
    replay_summary = None
    if replay_errors:
        replay_summary = {
            "exact_blocks": sum(bool(item["exact"]) for item in replay_errors),
            "total_blocks": len(replay_errors),
            "mean_abs_error": float(np.mean([
                item["mean_abs_error"] for item in replay_errors
            ])),
            "max_abs_error": max(item["max_abs_error"] for item in replay_errors),
            "metric_oracle": "stored label_final_rowmax",
        }
    return {
        "dataset": payload["dataset"],
        "sample_id": sample_text,
        "sample_index": int(payload.get("sample_index", 0)),
        "qualitative_block_index": int(block["block_index"]),
        "qualitative_layer_index": layer,
        "summary_granularity": args.summary_granularity,
        "qualitative_scope": args.qualitative_scope,
        "future_display_rows": row_count,
        "candidate_count": candidate_count,
        "display_candidate_count": display_candidate_count,
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
        "article_regions": article_regions,
        "teacher_replay_rowmax": replay_summary,
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
