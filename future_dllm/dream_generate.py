"""Sparse-dLLM Dream decoding with a selectable cache ranking rule.

The sampling order and transfer schedule follow the reference
``dream/generation_utils.py::DreamGenerationMixin._sample``. Step 0 samples the
whole block but confirms only its first token; subsequent steps sample masked
rows and use the remaining-mask timestep schedule. Sampling primitives retain
the Dream authors' Apache-2.0 implementation in generation_utils.py.
"""

import torch
import torch.nn.functional as F

from .cache import CustomCache
from .dream_decoding import DreamDecoding
from .generation_utils import sample_tokens

MASK_ID = 151666


def shift_logits(logits):
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


@torch.no_grad()
def generate(model, prompt, steps=None, gen_length=128, block_length=32,
             temperature=0.2, cfg_scale=0., remasking=None,
             mask_id=MASK_ID, cache_scorer=None, *, alg="entropy", top_p=0.95,
             top_k=None, alg_temp=None, eps=1e-3, eviction_method="student",
             on_block_complete=None, on_step=None, oracle_reduce=None):
    """Return prompt + answer using the reference block sampling schedule.

    ``on_block_complete(x, cache, block_index, block_start, selection_input)``
    observes completed blocks for teacher collection. The selection input is
    the full sequence entering step 1, before cache selection and revelation.
    ``on_step(x, block_index, step_index)`` supports reference parity checks.
    Callbacks must not sample or modify x.
    """
    if cfg_scale != 0 or remasking is not None:
        raise ValueError("Dream uses alg/temperature/top_p, not LLaDA cfg_scale/remasking")
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
        raise ValueError("Dream cache decoding requires one non-empty, unpadded prompt")
    if block_length < 1 or model.config.block_len != block_length:
        raise ValueError("block_length must be positive and match model.config.block_len")
    if gen_length < 1 or gen_length % block_length:
        raise ValueError("gen_length must be positive and divisible by block_length")
    if eviction_method not in ("student", "sparse", "oracle"):
        raise ValueError("eviction_method must be student, sparse or oracle")
    if (oracle_reduce is not None) != (eviction_method == "oracle"):
        raise ValueError("oracle_reduce and eviction_method='oracle' go together")
    if model.config.keep_ratio < 1 and eviction_method == "student" and cache_scorer is None:
        raise ValueError("student eviction requires a scorer")

    settings = DreamDecoding(alg=alg, temperature=temperature, top_p=top_p,
                             steps=256 if steps is None else steps, eps=eps,
                             top_k=top_k, alg_temp=alg_temp)
    steps = settings.steps_for_length(gen_length)
    num_blocks = gen_length // block_length
    if steps % num_blocks or steps // num_blocks < 2:
        raise ValueError("steps must divide into at least two steps per block")
    steps_per_block = steps // num_blocks
    x = F.pad(prompt, (0, gen_length), value=mask_id)
    prompt_len = prompt.shape[1]
    timesteps = torch.linspace(1, eps, steps_per_block + 1, device=x.device)

    for block_index in range(num_blocks):
        bs = prompt_len + block_index * block_length
        be = bs + block_length
        selection_input = None

        def step_block(cache):
            """Run one block's reveal schedule against ``cache``.

            Factored out because the oracle runs it twice on the same block:
            once with the whole cache to settle the answer the label is read
            off, then again against the cache that label prunes.
            """
            nonlocal selection_input
            for i in range(steps_per_block):
                cache_state = min(i, 2)
                model_input = x if cache_state != 2 else x[:, bs:be]
                if i == 1 and on_block_complete is not None:
                    selection_input = x.clone()
                logits = model(input_ids=model_input, position_offset=bs,
                               cache_state=cache_state, customcache=cache,
                               attention_mask="full", position_ids=None).logits
                logits = shift_logits(logits)

                if cache_state == 0:
                    _, x0 = sample_tokens(logits[:, bs:be], temperature=temperature,
                                          top_p=top_p, top_k=top_k)
                    x[:, bs] = x0[:, 0]
                    if on_step is not None:
                        on_step(x, block_index, i)
                    continue

                if cache_state == 1:
                    model_input = model_input[:, bs:be]
                    logits = logits[:, bs:be]
                mask_index = model_input == mask_id
                mask_logits = logits[mask_index]
                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k,
                    margin_confidence=alg == "topk_margin", neg_entropy=alg == "entropy")
                t, s = timesteps[i], timesteps[i + 1]
                num_mask_token = mask_index.sum() / mask_index.shape[0]
                number_transfer_tokens = (
                    int(num_mask_token * (1 - s / t)) if i < steps_per_block - 1
                    else int(num_mask_token))
                block_confidence = torch.full_like(model_input, -torch.inf,
                                                   device=model.device, dtype=logits.dtype)
                block_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(block_confidence, number_transfer_tokens)
                    else:
                        block_confidence = F.softmax(block_confidence / alg_temp, dim=-1)
                        transfer_index = torch.multinomial(
                            block_confidence, num_samples=number_transfer_tokens)
                    x_block = torch.zeros_like(model_input, device=model.device,
                                               dtype=torch.long) + mask_id
                    x_block[mask_index] = x0.clone()
                    row_indices = torch.arange(model_input.size(0), device=model.device)
                    row_indices = row_indices.unsqueeze(1).expand_as(transfer_index)
                    x[:, bs:be][row_indices, transfer_index] = x_block[row_indices, transfer_index]
                if on_step is not None:
                    on_step(x, block_index, i)

        if oracle_reduce is None:
            cache = CustomCache(
                n_layers=model.config.num_hidden_layers, device=model.device,
                keep_ratio=model.config.keep_ratio, cache_scorer=cache_scorer,
                prompt_length=prompt_len, generation_length=gen_length,
                eviction_method=eviction_method, baseline_order=True)
            cache.collect_pool = on_block_complete is not None
            if cache.collect_pool and model.config.keep_ratio != 1.0:
                raise ValueError("teacher collection requires keep_ratio=1.0")
            step_block(cache)
        else:
            cache = _oracle_block(model, x, bs, be, prompt_len, gen_length,
                                  step_block, oracle_reduce)

        if on_block_complete is not None:
            on_block_complete(x, cache, block_index, bs, selection_input)
    return x


def _oracle_block(model, x, bs, be, prompt_len, gen_length, step_block, reduce):
    """Decode the block twice: once whole, then against its own label.

    The upper bound the trained scorer is read against. Pass A keeps the entire
    cache, so the block settles to the answer the teacher label is defined on;
    one more forward over the completed block gives the attention that label is
    built from. The block is then returned to masks and decoded again with the
    cache pruned to that label's top-k -- the same budget the scorer gets, but
    with the answer's own attention standing in for a prediction of it.

    Not a guaranteed ceiling: the label is scored against pass A's answer while
    the reported answer comes out of pass B, and the two can diverge.
    """
    row_reduce, group_reduce, per_head = reduce
    n_layers = model.config.num_hidden_layers
    masked_block = x[:, bs:be].clone()

    full = CustomCache(n_layers=n_layers, device=model.device, keep_ratio=1.0,
                       prompt_length=prompt_len, generation_length=gen_length,
                       eviction_method="sparse", baseline_order=True)
    # Keeps the whole pool in candidate order, which is what makes the label's
    # columns line up with the candidates pass B rebuilds.
    full.collect_pool = True
    step_block(full)

    full.capture_rows = True
    full.capture_per_head = per_head
    full.group_reduce = group_reduce
    model(input_ids=x[:, bs:be], position_offset=bs, cache_state=2,
          customcache=full, attention_mask="full", position_ids=None)
    full.capture_rows = False
    axis = 1 if per_head else 0
    label = {layer: (full.pending_rows[layer].amax(axis) if row_reduce == "max"
                     else full.pending_rows[layer].mean(axis))
             for layer in range(n_layers)}
    full.pending_rows.clear()

    x[:, bs:be] = masked_block
    cache = CustomCache(n_layers=n_layers, device=model.device,
                        keep_ratio=model.config.keep_ratio,
                        prompt_length=prompt_len, generation_length=gen_length,
                        eviction_method="oracle", baseline_order=True)
    cache.oracle_label = label
    step_block(cache)
    return cache
