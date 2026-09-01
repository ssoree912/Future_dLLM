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
    parser.add_argument(
        "--reference-label", choices=("oracle", "future"), default="oracle",
        help="terminology used for the completed-LLaDA-attention reference",
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
    """Return kept indices, Utility-Mass@K, and Oracle-Recall@K."""
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


def attention_mass_at_k(
    attention_rows: torch.Tensor,
    keep_indices: torch.Tensor | np.ndarray,
) -> float:
    """Mean per-query fraction of candidate attention retained by Top-K."""
    rows = attention_rows.float()
    if rows.ndim != 2:
        raise ValueError("attention_rows must be [query tokens, candidates]")
    if not torch.isfinite(rows).all() or bool((rows < 0).any()):
        raise ValueError("attention rows must be finite and non-negative")
    keep = torch.as_tensor(keep_indices, dtype=torch.long, device=rows.device)
    if keep.ndim != 1 or keep.numel() < 1:
        raise ValueError("keep_indices must be a non-empty vector")
    if int(keep.min()) < 0 or int(keep.max()) >= rows.shape[1]:
        raise ValueError("keep index is outside the candidate dimension")
    row_totals = rows.sum(dim=1)
    if bool((row_totals <= 0).any()):
        raise ValueError("every attention row must have positive candidate mass")
    retained = rows.index_select(1, keep).sum(dim=1)
    return float((retained / row_totals).mean())


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
        current_attention_masses, ours_attention_masses = [], []
        current_utility_masses, ours_utility_masses = [], []
        current_recalls, ours_recalls = [], []
        for layer in range(int(payload["layer_count"])):
            future = block["future_score"][layer].float()
            future_rows = block["future_attention_rows"][layer].float()
            current_keep, current_utility_mass, current_recall = topk_metrics(
                block["current_score"][layer], future, cache_budget
            )
            ours_keep, ours_utility_mass, ours_recall = topk_metrics(
                block["predicted_score"][layer], future, cache_budget
            )
            current_attention_mass = attention_mass_at_k(future_rows, current_keep)
            ours_attention_mass = attention_mass_at_k(future_rows, ours_keep)
            oracle_keep = torch.topk(future, cache_budget).indices.sort().values.cpu()
            current_keeps.append(current_keep)
            ours_keeps.append(ours_keep)
            oracle_keeps.append(oracle_keep)
            current_attention_masses.append(current_attention_mass)
            ours_attention_masses.append(ours_attention_mass)
            current_utility_masses.append(current_utility_mass)
            ours_utility_masses.append(ours_utility_mass)
            current_recalls.append(current_recall)
            ours_recalls.append(ours_recall)
            metric_rows.append({
                "block_index": int(block["block_index"]),
                "layer_index": layer,
                "candidate_count": int(future.numel()),
                "cache_budget": cache_budget,
                "current_attention_mass_at_k": current_attention_mass,
                "ours_attention_mass_at_k": ours_attention_mass,
                "current_recall_at_k": current_recall,
                "ours_recall_at_k": ours_recall,
                "current_utility_mass_at_k": current_utility_mass,
                "ours_utility_mass_at_k": ours_utility_mass,
            })
        block["current_keep"] = torch.stack(current_keeps)
        block["ours_keep"] = torch.stack(ours_keeps)
        block["oracle_keep"] = torch.stack(oracle_keeps)
        block["current_attention_mass_at_k"] = torch.tensor(current_attention_masses)
        block["ours_attention_mass_at_k"] = torch.tensor(ours_attention_masses)
        block["current_utility_mass_at_k"] = torch.tensor(current_utility_masses)
        block["ours_utility_mass_at_k"] = torch.tensor(ours_utility_masses)
        block["current_recall_at_k"] = torch.tensor(current_recalls)
        block["ours_recall_at_k"] = torch.tensor(ours_recalls)
    payload["cache_budget"] = cache_budget
    return metric_rows


def mask_future_attention(
    future_attention_rows: torch.Tensor,
    keep_indices: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Show the future attention that remains accessible after eviction."""
    keep = torch.as_tensor(keep_indices, dtype=torch.long)
    masked = torch.zeros_like(future_attention_rows)
    masked[..., keep] = future_attention_rows[..., keep]
    return masked


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
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, PowerNorm

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
    actual_keep_ratio = float(payload["actual_keep_ratio"])
    evicted_ratio = 1.0 - actual_keep_ratio
    regions = candidate_regions(block)
    if args.qualitative_scope == "prompt":
        display_candidate_count = int(block["prompt_length"])
        display_regions = [region for region in regions if region["name"] == "Prompt"]
    else:
        display_candidate_count = candidate_count
        display_regions = regions
    display_slice = slice(0, display_candidate_count)
    layer = args.layer

    raw_future_rows = block["future_attention_rows"][layer]
    sentence_groups = block.get("completed_sentence_groups") or []
    if args.summary_granularity == "sentence":
        if not sentence_groups:
            raise SystemExit(
                "analysis has no completed_sentence_groups; regenerate it before "
                "using --summary-granularity sentence"
            )
        comparison_rows = aggregate_future_rows(raw_future_rows, sentence_groups)
        future_row_labels = [group["label"] for group in sentence_groups]
        future_ylabel = "Completed summary sentence"
    else:
        comparison_rows = raw_future_rows
        future_row_labels = [
            f"{index + 1:02d}  {label}"
            for index, label in enumerate(block["completed_token_labels"])
        ]
        future_ylabel = "Completed block token"

    current_keep = block["current_keep"][layer]
    oracle_keep = block["oracle_keep"][layer]
    ours_keep = block["ours_keep"][layer]
    unmasked_future = comparison_rows[..., display_slice].detach().float().cpu().numpy()
    finite_future = unmasked_future[np.isfinite(unmasked_future)]
    positive_future = finite_future[finite_future > 0]
    if positive_future.size:
        color_max = float(np.quantile(positive_future, 0.995))
        color_max = max(color_max, float(positive_future.min()))
    else:
        color_max = 1.0
    power_gamma = 0.40
    heatmaps = [
        mask_future_attention(comparison_rows, keep)[..., display_slice]
        .detach().float().cpu().numpy()
        for keep in (current_keep, oracle_keep, ours_keep)
    ]
    attention_norm = PowerNorm(
        gamma=power_gamma, vmin=0.0, vmax=color_max, clip=True
    )

    selected_recall = (
        float(block["current_recall_at_k"][layer]),
        1.0,
        float(block["ours_recall_at_k"][layer]),
    )
    reference_label = {
        "oracle": "Oracle",
        "future": "Future",
    }[args.reference_label]
    retention_colors = ["#4C78A8", "#5B4B8A", "#1B9E77"]

    # Native two-column paper width: avoids shrinking 8--9 pt labels from a
    # presentation-sized canvas when the PDF is included at \textwidth.
    figure = plt.figure(figsize=(7.15, 2.85), constrained_layout=False)
    grid = figure.add_gridspec(
        1, 4, width_ratios=(1.0, 1.0, 1.0, 0.78),
        left=0.065, right=0.985, bottom=0.28, top=0.84, wspace=0.28,
    )
    heatmap_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    metric_axis = figure.add_subplot(grid[0, 3])

    image_artist = None
    article_regions = payload.get("article_regions") or []
    for panel_index, (axis, values, keep, recall, retain_color) in enumerate(zip(
        heatmap_axes, heatmaps, (current_keep, oracle_keep, ours_keep),
        selected_recall, retention_colors,
    )):
        image_artist = axis.imshow(
            values, aspect="auto", interpolation="nearest", cmap="viridis",
            norm=attention_norm, origin="upper",
        )
        if panel_index == 0:
            axis.set_ylabel(future_ylabel, fontsize=8.5, labelpad=5)
        else:
            axis.tick_params(labelleft=False)

        row_count = len(future_row_labels)
        tick_count = min(5, row_count)
        y_ticks = np.unique(
            np.linspace(0, row_count - 1, tick_count).round().astype(int)
        )
        if args.summary_granularity == "token":
            y_labels = [str(int(tick) + 1) for tick in y_ticks]
        else:
            y_labels = [f"S{int(tick) + 1}" for tick in y_ticks]
        axis.set_yticks(y_ticks, y_labels, fontsize=7.5)
        axis.tick_params(length=2.5, width=0.6)
        axis.set_xlim(-0.5, display_candidate_count - 0.5)
        axis.set_xticks(
            [0, display_candidate_count - 1],
            ["0", str(display_candidate_count - 1)], fontsize=7.5,
        )
        axis.text(
            0.5, 1.145, f"{reference_label} overlap: {recall:.1%}",
            transform=axis.transAxes, ha="center", va="bottom", fontsize=7.3,
            color="#30343B", fontweight="semibold", clip_on=False,
        )
        for spine in axis.spines.values():
            spine.set_color("#7E8490")
            spine.set_linewidth(0.65)

        keep_mask = np.zeros(candidate_count, dtype=np.uint8)
        keep_mask[np.asarray(keep, dtype=np.int64)] = 1
        mask_axis = axis.inset_axes([0.0, 1.018, 1.0, 0.060])
        mask_axis.imshow(
            keep_mask[display_slice][None, :], aspect="auto",
            interpolation="nearest",
            cmap=ListedColormap(["#E5E7EB", retain_color]), vmin=0, vmax=1,
        )
        mask_axis.set_axis_off()

        if article_regions:
            for article_index, region in enumerate(article_regions):
                start = max(0, int(region["start"]))
                end = min(display_candidate_count, int(region["end"]))
                if end <= start:
                    continue
                for boundary in (start, end):
                    axis.axvline(
                        boundary - 0.5, color="white", linewidth=0.55,
                        linestyle=(0, (2, 2)), alpha=0.80,
                    )
                midpoint = (start + end - 1) / 2
                axis.text(
                    midpoint, 1.085, f"A{article_index + 1}",
                    transform=axis.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7.0, color="#3F4650", clip_on=False,
                )
            if args.qualitative_scope == "all":
                tail_labels = {
                    "Previously completed blocks": "Decoded",
                    "Future masked blocks": "Masked",
                }
                for region in display_regions:
                    if region["name"] not in tail_labels:
                        continue
                    start, end = int(region["start"]), int(region["end"])
                    axis.axvline(
                        start - 0.5, color="white", linewidth=0.65,
                        linestyle=(0, (2, 2)), alpha=0.90,
                    )
                    axis.text(
                        (start + end - 1) / 2, 1.085,
                        tail_labels[region["name"]],
                        transform=axis.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=6.7, color="#3F4650",
                        clip_on=False,
                    )
        else:
            short_region_names = {
                "Previously completed blocks": "Decoded",
                "Future masked blocks": "Masked",
            }
            for region in display_regions:
                start, end = int(region["start"]), int(region["end"])
                axis.axvline(start - 0.5, color="white", linewidth=0.55,
                            linestyle=(0, (2, 2)), alpha=0.80)
                axis.text(
                    (start + end - 1) / 2, 1.085,
                    short_region_names.get(region["name"], region["name"]),
                    transform=axis.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6.9, color="#3F4650", clip_on=False,
                )

    if image_artist is not None:
        heatmap_left = heatmap_axes[0].get_position().x0
        heatmap_right = heatmap_axes[-1].get_position().x1
        figure.text(
            (heatmap_left + heatmap_right) / 2, 0.175,
            "Cache candidate token", ha="center", va="center", fontsize=8.2,
        )
        colorbar_axis = figure.add_axes([
            heatmap_left, 0.065, heatmap_right - heatmap_left, 0.035,
        ])
        colorbar = figure.colorbar(
            image_artist, cax=colorbar_axis, orientation="horizontal",
        )
        colorbar.set_label(
            "Retained post-completion attention", fontsize=7.5, labelpad=2,
        )
        colorbar.set_ticks(np.linspace(0.0, color_max, 4))
        colorbar.ax.tick_params(labelsize=6.5, length=2, pad=1)

    attention_mass_current = float(np.mean([
        row["current_attention_mass_at_k"] for row in metric_rows
    ]))
    attention_mass_ours = float(np.mean([
        row["ours_attention_mass_at_k"] for row in metric_rows
    ]))
    utility_mass_current = float(np.mean([
        row["current_utility_mass_at_k"] for row in metric_rows
    ]))
    utility_mass_ours = float(np.mean([
        row["ours_utility_mass_at_k"] for row in metric_rows
    ]))
    recall_current = float(np.mean([row["current_recall_at_k"] for row in metric_rows]))
    recall_ours = float(np.mean([row["ours_recall_at_k"] for row in metric_rows]))
    x = np.arange(2)
    width = 0.34
    metric_axis.axhline(
        1.0, color="#545B66", linestyle=(0, (4, 3)), linewidth=1.1,
        label="_nolegend_", zorder=1,
    )
    current_bars = metric_axis.bar(
        x - width / 2, [attention_mass_current, recall_current], width,
        color="#4C78A8", label="Sparse-dLLM"
    )
    ours_bars = metric_axis.bar(
        x + width / 2, [attention_mass_ours, recall_ours], width,
        color="#1B9E77", label="Preview-dLLM"
    )
    metric_axis.set_xticks(
        x, ["Attention\nMass@K", f"{reference_label}\nRecall@K"]
    )
    metric_axis.set_ylim(0.0, 1.05)
    metric_axis.set_ylabel("Score ↑", fontsize=8.0, labelpad=4)
    metric_axis.grid(axis="y", color="#D8DCE3", linewidth=0.65, alpha=0.8)
    metric_axis.set_axisbelow(True)
    metric_axis.spines[["top", "right"]].set_visible(False)
    metric_axis.legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.015),
        ncol=1, fontsize=6.8, borderaxespad=0.0, labelspacing=0.15,
        handlelength=1.2,
    )
    for bars in (current_bars, ours_bars):
        metric_axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.2)

    sample_text = str(payload.get("sample_id", payload.get("sample_index", "?")))

    png_path = output_dir / "figure_1.png"
    pdf_path = output_dir / "figure_1.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    panel_paths = {}
    for name, axis in zip(
        ("sparse", "oracle", "ours", "metrics"),
        (*heatmap_axes, metric_axis),
    ):
        visible_axes = {item: item.get_visible() for item in figure.axes}
        keep_visible = {axis, *axis.child_axes}
        for item in figure.axes:
            if item not in keep_visible:
                item.set_visible(False)
        original_xlabel = axis.get_xlabel()
        if axis in heatmap_axes:
            axis.set_xlabel("Cache candidate token", fontsize=8.2, labelpad=4)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        extent = axis.get_tightbbox(renderer).transformed(
            figure.dpi_scale_trans.inverted()
        ).expanded(1.03, 1.05)
        if axis in heatmap_axes:
            extent.y1 += 0.20
        panel_png = output_dir / f"panel_{name}.png"
        panel_pdf = output_dir / f"panel_{name}.pdf"
        figure.savefig(panel_png, dpi=300, bbox_inches=extent, facecolor="white")
        figure.savefig(panel_pdf, bbox_inches=extent, facecolor="white")
        axis.set_xlabel(original_xlabel)
        for item, visible in visible_axes.items():
            item.set_visible(visible)
        panel_paths[name] = {
            "png": str(panel_png.resolve()),
            "pdf": str(panel_pdf.resolve()),
        }
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
        "reference_label": args.reference_label,
        "future_display_rows": len(future_row_labels),
        "candidate_count": candidate_count,
        "display_candidate_count": display_candidate_count,
        "heatmap_definition": (
            "full-cache future_attention_rows multiplied by each method's global "
            "Top-K retention mask; the separate strip shows every selected entry"
        ),
        "attention_color_scale": {
            "shared_across_panels": True,
            "power_gamma": power_gamma,
            "vmax_quantile": 0.995,
            "vmax": color_max,
        },
        "cache_budget": int(payload["cache_budget"]),
        "requested_keep_ratio": float(payload["requested_keep_ratio"]),
        "actual_keep_ratio": actual_keep_ratio,
        "evicted_ratio": evicted_ratio,
        "averaging_units": len(metric_rows),
        "metric_definitions": {
            "attention_mass_at_k": (
                "mean over completed-block query tokens of retained candidate "
                "attention divided by total candidate attention"
            ),
            "oracle_recall_at_k": (
                "intersection of method Top-K and row-max utility Top-K, divided by K"
            ),
            "utility_mass_at_k_analysis_only": (
                "sum of row-max utility at method Top-K divided by total row-max utility"
            ),
        },
        "full_cache_reference": {
            "attention_mass": 1.0,
            "oracle_recall": 1.0,
        },
        "average": {
            "current_attention_mass_at_k": attention_mass_current,
            "ours_attention_mass_at_k": attention_mass_ours,
            "current_recall_at_k": recall_current,
            "ours_recall_at_k": recall_ours,
            "current_utility_mass_at_k": utility_mass_current,
            "ours_utility_mass_at_k": utility_mass_ours,
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
        "panel_files": panel_paths,
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
