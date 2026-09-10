"""Run the trained scorer inside Sparse-dLLM's own Dream code.

`baselines/sparse_dllm/dream/` is their implementation. This module subclasses
their `CustomCache` and wraps their decoder layer at import time, so a run can
use their cache, their eviction plumbing, their block schedule and their
decoding while ranking candidates with our student instead of their attention
score.

That is what makes "we only change the scorer" checkable: the baseline and our
method execute the same files, and the selection code is reached identically.
Their directory carries one six-line addition -- a `record_attention` call that
teacher extraction needs and their own cache does not define, so the baseline
never takes the branch. It is written out in `baselines/sparse_dllm/DIFF.md`
and `DIFF.patch`; nothing else differs from upstream.

The alternative -- porting their decode into our loop -- was tried and does not
hold. Matching `alg="entropy"` still left our reveal counts different: theirs
recomputes `int(remaining * (1 - s/t))` from the timestep schedule every step,
ours precomputes `mask_num // steps` once per block. Four GSM8K prompts at
keep_ratio=1.0 agreed on only 63/64, 58/64, 13/64 and 57/64 tokens. Equality
has to come from running the same code, not from aligning it piece by piece.

Two things their cache does not do, both added here because the scorer needs
them and neither changes their selection when the scorer is absent:

  * the student reads a layer's residual stream *before* input_layernorm, so
    the decoder layer is wrapped to hand that tensor to the cache on the
    selection forward;
  * teacher extraction needs the attention a finished block pays to the cached
    columns, so the cache can record it on request.
"""

from __future__ import annotations

import torch

from baselines.sparse_dllm.dream import generation_utils as _their_generation
from baselines.sparse_dllm.dream import modeling_dream as _their_modeling
from baselines.sparse_dllm.dream.Cache import CustomCache as _TheirCache


class StudentScorerCache(_TheirCache):
    """Their cache, ranking candidates with the student.

    Everything up to the ranking is inherited: the block's own columns come out
    first, the budget is `int(candidates * keep_ratio)`, and the survivors are
    re-indexed over KV heads. Only `importance` changes.
    """

    def __init__(self, *args, cache_scorer=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cache_scorer = cache_scorer
        self.layer_hidden_states = {}
        # Teacher collection: keep every candidate in candidate order and record
        # what the finished block attends to.
        self.collect_pool = False
        self.capture_rows = False
        self.pending_rows = {}

    # -- the two hooks the scorer needs ------------------------------------
    def capture_layer_hidden_states(self, layer_id: int, hidden_states) -> None:
        if self.cache_scorer is not None or self.collect_pool:
            self.layer_hidden_states[layer_id] = hidden_states

    def record_attention(self, layer_id: int, q: torch.Tensor, k: torch.Tensor) -> None:
        """Per-row attention over the candidate columns, head-averaged.

        `k` holds the candidates followed by the block's own keys; the block
        columns are dropped so the rows score the cache alone.
        """
        if not self.capture_rows:
            return
        if q.size(1) != k.size(1):
            k = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (q.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        n_candidates = k.size(-2) - q.size(-2)
        self.pending_rows[layer_id] = weights[..., :n_candidates].mean(dim=1).squeeze(0)

    # -- selection ---------------------------------------------------------
    def filter_cache(self, layer_id: int, q_block: torch.Tensor,
                     bef_filtered_len: int, block_len: int):
        # Their path whenever the scorer would not change the outcome. At
        # keep_ratio=1.0 nothing is evicted either way, but their topk still
        # reorders the survivors by importance and ours would not -- and a
        # different summation order in SDPA is enough to diverge under
        # sampling. Delegating keeps keep_ratio=1.0 bit-identical between the
        # baseline and us, which is what lets one run serve both rows.
        if not self.collect_pool and (self.cache_scorer is None
                                      or self.keep_ratios[layer_id] >= 1.0):
            return super().filter_cache(layer_id, q_block, bef_filtered_len, block_len)

        cached = self.get_cache(layer_id)
        # Identical to their first two lines: the block attends to its own keys
        # directly, so they never belong in its cache.
        keep_k = torch.cat([cached["k"][:, :, :bef_filtered_len, :],
                            cached["k"][:, :, bef_filtered_len + block_len:, :]], dim=2)
        keep_v = torch.cat([cached["v"][:, :, :bef_filtered_len, :],
                            cached["v"][:, :, bef_filtered_len + block_len:, :]], dim=2)

        if self.collect_pool:
            # Teacher collection keeps the pool in candidate order so recorded
            # attention columns line up with the entries they belong to.
            self.cache[layer_id] = {"k": keep_k, "v": keep_v}
            return

        hidden_states = self.layer_hidden_states.pop(layer_id, None)
        if hidden_states is None:
            raise RuntimeError(f"missing hidden states for scorer layer {layer_id}")
        if hidden_states.shape[0] != 1:
            raise RuntimeError("scorer selection requires batch_size=1")
        sequence_length = int(hidden_states.shape[1])

        candidate_indices = torch.cat([
            torch.arange(bef_filtered_len, device=hidden_states.device),
            torch.arange(bef_filtered_len + block_len, sequence_length,
                         device=hidden_states.device),
        ])
        if candidate_indices.numel() != keep_k.size(-2):
            raise RuntimeError("scorer candidates do not match cached K/V")
        block_indices = torch.arange(bef_filtered_len, bef_filtered_len + block_len,
                                     device=hidden_states.device)

        scores = self.cache_scorer.forward_layer(
            layer_id, hidden_states.float(), candidate_indices,
            head="score", block_indices=block_indices).float()
        keep_num = int(candidate_indices.numel() * self.keep_ratios[layer_id])
        keep_indices = torch.topk(scores, k=keep_num, dim=-1).indices.squeeze(0).sort().values

        head_index = torch.arange(keep_k.size(1), device=keep_k.device)[:, None]
        self.cache[layer_id] = {"k": keep_k[:, head_index, keep_indices],
                                "v": keep_v[:, head_index, keep_indices]}


# ---------------------------------------------------------------------------
# Wrapping their code without editing it
# ---------------------------------------------------------------------------

_ACTIVE_SCORER: dict = {"scorer": None, "collect_pool": False, "capture_rows": False}
_PATCHED = False


def _patch_once() -> None:
    """Route their cache construction and layer forward through ours."""
    global _PATCHED
    if _PATCHED:
        return

    their_cache_cls = _their_generation.CustomCache

    def cache_factory(*args, **kwargs):
        if _ACTIVE_SCORER["scorer"] is None and not _ACTIVE_SCORER["collect_pool"]:
            return their_cache_cls(*args, **kwargs)
        cache = StudentScorerCache(*args, cache_scorer=_ACTIVE_SCORER["scorer"], **kwargs)
        cache.collect_pool = _ACTIVE_SCORER["collect_pool"]
        return cache

    _their_generation.CustomCache = cache_factory

    # The student reads the layer input, before input_layernorm, which is what
    # it was trained on. Their layer does not expose it, so wrap the forward.
    original_forward = _their_modeling.DreamDecoderLayer.forward

    def forward(self, position_offset, cache_state, customcache, hidden_states,
                *args, **kwargs):
        if cache_state == 1 and hasattr(customcache, "capture_layer_hidden_states"):
            customcache.capture_layer_hidden_states(self.self_attn.layer_idx, hidden_states)
        return original_forward(self, position_offset, cache_state, customcache,
                                hidden_states, *args, **kwargs)

    _their_modeling.DreamDecoderLayer.forward = forward

    _PATCHED = True


def set_scorer(scorer=None, *, collect_pool: bool = False) -> None:
    """Choose what the next cache built by their generate will rank with.

    ``scorer=None`` and ``collect_pool=False`` leaves their own attention score
    in place, which is the baseline.
    """
    _patch_once()
    _ACTIVE_SCORER["scorer"] = scorer
    _ACTIVE_SCORER["collect_pool"] = collect_pool


def load_model(model_path, *, block_length: int = 32, keep_ratio: float = 1.0,
               kernel_size: int = 3, device_map="auto"):
    """Their DreamModel, configured the way their own wrapper configures it."""
    from baselines.sparse_dllm.dream.configuration_dream import DreamConfig
    from baselines.sparse_dllm.dream.modeling_dream import DreamModel

    _patch_once()
    config = DreamConfig.from_pretrained(str(model_path))
    config.block_len = int(block_length)
    config.kernel_size = int(kernel_size)
    config.keep_ratio = float(keep_ratio)
    model = DreamModel.from_pretrained(
        str(model_path), config=config, device_map=device_map,
        torch_dtype=torch.bfloat16).eval()
    return model
