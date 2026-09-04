"""Future-attention KV cache eviction, shared by every diffusion-LM backend.

The block-wise machinery here is model-agnostic: it only ever sees post-RoPE
K/V in ``[batch, kv_heads, positions, head_dim]`` layout and the residual-stream
hidden states the scorer reads. ``modeling_llada`` and ``modeling_dream`` both
call into this one implementation, so a change to the selection rule cannot
apply to one backend and not the other.

Note for GQA backends (Dream): the cache stores K/V *before* ``repeat_kv``, so
``keep_ratios`` budgets and head indexing are expressed over KV heads. The
scoring helpers below expand KV heads to query heads themselves.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn.functional as F


def sparse_dllm_current_score(
    q_block: torch.Tensor,
    candidate_k: torch.Tensor,
    pool_kernel_size: Optional[int] = 3,
) -> torch.Tensor:
    """Sparse-dLLM's selection-time score for each candidate cache entry.

    The reference baseline first averages the current-block queries over the
    token rows, takes their dot product with every candidate key, averages over
    attention heads, and optionally applies stride-1 local max pooling.  It does
    not softmax or scale the dot products before ranking.

    Args:
        q_block: ``[batch, query_heads, block_length, head_dim]``.
        candidate_k: ``[batch, kv_heads, candidates, head_dim]``.
        pool_kernel_size: Positive odd pooling width, or ``None`` to disable it.

    Returns:
        A ``[batch, candidates]`` tensor in candidate-cache order.
    """
    if q_block.ndim != 4 or candidate_k.ndim != 4:
        raise ValueError("q_block and candidate_k must both be rank-4 tensors")
    if q_block.size(0) != candidate_k.size(0):
        raise ValueError("q_block and candidate_k batch sizes do not match")
    if q_block.size(-1) != candidate_k.size(-1):
        raise ValueError("q_block and candidate_k head dimensions do not match")
    if q_block.size(1) != candidate_k.size(1):
        if q_block.size(1) % candidate_k.size(1):
            raise ValueError("query heads must be divisible by KV heads")
        candidate_k = candidate_k.repeat_interleave(
            q_block.size(1) // candidate_k.size(1), dim=1
        )

    average_query = q_block.mean(dim=-2)
    scores = torch.matmul(
        average_query.unsqueeze(-2), candidate_k.transpose(-2, -1)
    ).squeeze(-2)
    importance = scores.mean(dim=1)

    if pool_kernel_size is not None:
        if pool_kernel_size < 1 or pool_kernel_size % 2 == 0:
            raise ValueError("pool_kernel_size must be a positive odd integer or None")
        importance = F.max_pool1d(
            importance.unsqueeze(1),
            kernel_size=pool_kernel_size,
            stride=1,
            padding=pool_kernel_size // 2,
        ).squeeze(1)
    return importance


class CustomCache:
    """Block-wise KV cache with future-attention eviction.

    One instance per block. Step 0-1 run the full sequence and build the cache;
    ``filter_cache`` then prunes it to ``keep_ratio`` using the student scorer,
    and steps 2.. run against the pruned cache plus the block itself.

    Two modes:
      * deployment - ``cache_scorer`` is a trained student, one selection per
        block, prompt and suffix compete in a single top-k.
      * teacher collection - ``collect_pool`` keeps every candidate in candidate
        order so recorded attention columns line up with the cache entries, and
        ``capture_rows`` hands the per-row attention to the extractor.
    """

    def __init__(
        self,
        n_layers: int,
        device: torch.device,
        keep_ratio: float = 1.0,
        cache_scorer=None,
        prompt_length: int = 0,
        generation_length: int = 0,
        capture_current_scores: bool = False,
        current_score_pool_kernel: Optional[int] = 3,
    ) -> None:
        self.cache = {}
        self.keep_ratios = [keep_ratio for _ in range(n_layers)]
        self.cache_scorer = cache_scorer
        self.prompt_length = prompt_length
        self.generation_length = generation_length

        # Figure/analysis-only Sparse-dLLM baseline captured at the exact cache
        # selection forward. Disabled during ordinary teacher and deployment
        # runs, so it adds no matmul or retained tensors there.
        self.capture_current_scores = capture_current_scores
        self.current_score_pool_kernel = current_score_pool_kernel
        self.current_scores = {}

        # Hidden states the student scores from; the block writes them per layer
        # on the step-1 forward and filter_cache consumes them.
        self.layer_hidden_states = {}

        # Teacher collection.
        self.collect_pool = False
        self.capture_rows = False
        self.pending_rows = {}
        self.row_mask = None

        # Optional: append one jsonl line per selection recording *where* the
        # kept entries sit (prompt / confirmed blocks / suffix). Off unless the
        # env var names a file, so normal runs pay nothing.
        self.keep_log = os.environ.get("FUTURE_DLLM_KEEP_LOG") or None

    def set_row_mask(self, row_mask: Optional[torch.Tensor]) -> None:
        self.row_mask = row_mask

    def record_attention(self, layer_id: int, q: torch.Tensor, k: torch.Tensor) -> None:
        """Per-row attention over the candidate columns, head-averaged.

        Only used while collecting teacher labels. ``k`` holds the candidates
        followed by the block's own keys; the block columns are dropped so the
        rows score the cache alone.
        """
        if not self.capture_rows:
            return
        if q.size(1) != k.size(1):
            k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (q.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        n_cand = k.size(-2) - q.size(-2)
        self.pending_rows[layer_id] = weights[..., :n_cand].mean(dim=1).squeeze(0)

    def capture_layer_hidden_states(self, layer_id: int, hidden_states: torch.Tensor) -> None:
        if self.cache_scorer is not None:
            self.layer_hidden_states[layer_id] = hidden_states

    def get_cache(self, layer_id: int):
        return self.cache.get(layer_id, {"k": None, "v": None})

    def update_cache(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        self.cache[layer_id] = {"k": k.clone(), "v": v.clone()}

    def filter_cache(self, layer_id: int, q_block: torch.Tensor,
                     cur_filtered_len: int, block_len: int):
        """Drop the block's own columns, then keep the top ``keep_ratio`` share."""
        cached = self.get_cache(layer_id)
        cached_k, cached_v = cached["k"], cached["v"]

        # The block's own keys never belong in its cache - it attends to them
        # directly - so they come out before anything is scored.
        keep_k = torch.cat([cached_k[:, :, :cur_filtered_len, :],
                            cached_k[:, :, cur_filtered_len + block_len:, :]], dim=2)
        keep_v = torch.cat([cached_v[:, :, :cur_filtered_len, :],
                            cached_v[:, :, cur_filtered_len + block_len:, :]], dim=2)

        if self.capture_current_scores:
            self.current_scores[layer_id] = sparse_dllm_current_score(
                q_block, keep_k, self.current_score_pool_kernel
            ).detach()

        if self.collect_pool or self.keep_ratios[layer_id] >= 1.0:
            # Nothing is evicted: keep the pool in candidate order so recorded
            # attention columns line up with the entries they belong to.
            self.cache[layer_id] = {"k": keep_k, "v": keep_v}
            return

        if self.cache_scorer is None:
            raise RuntimeError(
                "future_dllm evicts with a trained scorer; pass a student "
                "checkpoint, or run with keep_ratio=1.0 to disable eviction"
            )

        hidden_states = self.layer_hidden_states.pop(layer_id, None)
        if hidden_states is None:
            raise RuntimeError(f"missing hidden states for scorer layer {layer_id}")
        if hidden_states.shape[0] != 1:
            raise RuntimeError("scorer selection requires batch_size=1")
        sequence_length = int(hidden_states.shape[1])

        # Candidates are everything outside the block: prompt and the suffix
        # blocks that follow. Both compete in one top-k, so the budget is
        # exactly candidates * keep_ratio.
        candidate_indices = torch.cat([
            torch.arange(cur_filtered_len, device=hidden_states.device),
            torch.arange(cur_filtered_len + block_len, sequence_length,
                         device=hidden_states.device),
        ])
        if candidate_indices.numel() != keep_k.size(-2):
            raise RuntimeError("scorer candidates do not match cached K/V")

        block_indices = torch.arange(cur_filtered_len, cur_filtered_len + block_len,
                                     device=hidden_states.device)

        scores = self.cache_scorer.forward_layer(
            layer_id, hidden_states.float(), candidate_indices,
            head="score", block_indices=block_indices).float()
        keep_num = int(candidate_indices.numel() * self.keep_ratios[layer_id])
        keep_indices = torch.topk(scores, k=keep_num, dim=-1).indices.squeeze(0).sort().values

        if self.keep_log:
            kept = candidate_indices[keep_indices]
            P, bs = int(self.prompt_length), int(cur_filtered_len)
            hist = torch.histc(kept.float(), bins=10, min=0,
                               max=float(sequence_length)).to(torch.long)
            with open(self.keep_log, "a") as fh:
                fh.write(json.dumps({
                    "layer": layer_id, "block_start": bs, "prompt_len": P,
                    "seq_len": sequence_length, "kept": int(kept.numel()),
                    "n_prompt": int((kept < min(P, bs)).sum()),
                    "n_confirmed": int(((kept >= min(P, bs)) & (kept < bs)).sum()),
                    "n_suffix": int((kept >= bs + block_len).sum()),
                    "decile_hist": hist.tolist(),
                }) + "\n")

        head_index = torch.arange(keep_k.size(1), device=keep_k.device)[:, None]
        self.cache[layer_id] = {"k": keep_k[:, head_index, keep_indices],
                                "v": keep_v[:, head_index, keep_indices]}

    def clear(self):
        self.cache.clear()


