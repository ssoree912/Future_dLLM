"""Block-wise diffusion decoding for LLaDA, with future-attention cache eviction.

The per-block cache is built on step 0 and 1 by running the full sequence;
``filter_cache`` prunes it to ``keep_ratio`` with the trained scorer, and the
remaining steps run against the pruned cache plus the block itself.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .modeling_llada import CustomCache

MASK_ID = 126336


def add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling. float64 per arXiv:2409.02908; temperature 0 is greedy."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


def get_num_transfer_tokens(mask_index, steps):
    """How many tokens each step reveals, under LLaDA's linear noise schedule."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base, remainder = mask_num // steps, mask_num % steps
    counts = torch.zeros(mask_num.size(0), steps, device=mask_index.device,
                         dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        counts[i, :remainder[i]] += 1
    return counts


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=32,
             temperature=0., cfg_scale=0., remasking='low_confidence',
             mask_id=MASK_ID, cache_scorer=None, *, eviction_method="student",
             eviction_accum="none", eviction_accum_decay=1.0, oracle_reduce=None):
    """Generate ``gen_length`` tokens block by block.

    ``cache_scorer`` is a trained ``PromptUtilityStudent``; without one the model
    only runs at ``keep_ratio=1.0`` (no eviction). ``keep_ratio`` comes from
    ``model.config``.

    ``eviction_method`` picks what decides the eviction: ``"student"`` uses the
    trained scorer, ``"sparse"`` uses Sparse-dLLM's attention score, which is
    what makes the baseline row runnable on this backend too, and ``"oracle"``
    uses the block's own teacher label -- see ``_oracle_block``.

    ``eviction_accum="across_blocks"`` carries the scorer's own output forward
    between blocks instead of deciding each block from scratch -- H2O's time
    axis, with the block standing in for the AR step. ``eviction_accum_decay``
    weights the carried history: 1.0 is a plain running sum, 0.0 reproduces the
    per-block default. The state lives here because a cache lasts one block.
    """
    if eviction_method not in ("student", "sparse", "oracle"):
        raise ValueError("eviction_method must be student, sparse or oracle")
    if (oracle_reduce is not None) != (eviction_method == "oracle"):
        raise ValueError("oracle_reduce and eviction_method='oracle' go together")
    if eviction_accum not in ("none", "across_blocks"):
        raise ValueError("eviction_accum must be none or across_blocks")
    accum_state = {} if eviction_accum == "across_blocks" else None
    if model.config.keep_ratio < 1 and eviction_method == "student" and cache_scorer is None:
        raise ValueError("student eviction requires a scorer")
    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long,
                   device=model.device)
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        num_transfer = get_num_transfer_tokens(
            x[:, block_start:block_end] == mask_id, steps_per_block)

        def step_block(cache):
            """Run one block's reveal schedule against ``cache``.

            Factored out because the oracle runs it twice on the same block:
            once with the whole cache to settle the answer the label is read
            off, then again against the cache that label prunes.
            """
            for i in range(steps_per_block):
                cache_state = 2 if i > 1 else i
                model_input = x if cache_state != 2 else x[:, block_start:block_end]
                mask_index = (model_input == mask_id)

                logits = model(model_input, block_start, cache_state, cache).logits
                x0 = torch.argmax(add_gumbel_noise(logits, temperature), dim=-1)

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, -1, torch.unsqueeze(x0, -1)), -1)
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                target = x if cache_state != 2 else x[:, block_start:block_end]
                if cache_state != 2:
                    x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, target)
                confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))
                for j in range(confidence.shape[0]):
                    reveal = torch.topk(confidence[j], k=num_transfer[j, i]).indices
                    target[j, reveal] = x0[j, reveal]

        if oracle_reduce is None:
            # A fresh cache per block: nothing is carried over, so the selection is
            # made once against the block that will use it.
            # baseline_order matches Sparse-dLLM at keep_ratio=1.0. Their
            # modeling_llada.filter_cache scores and top-k's unconditionally, so
            # even when the budget keeps everything the survivors come back ordered
            # by importance, not by position. Keeping natural order there changes
            # the order of the float sums in attention and, through sampling, the
            # tokens -- the same divergence the Dream path hit.
            cache = CustomCache(
                n_layers=model.config.n_layers, device=model.device,
                keep_ratio=model.config.keep_ratio,
                cache_scorer=cache_scorer, prompt_length=prompt_len,
                generation_length=gen_length,
                eviction_method=eviction_method, baseline_order=True,
                accum_state=accum_state, accum_decay=eviction_accum_decay)
            step_block(cache)
        else:
            _oracle_block(model, x, block_start, block_end, prompt_len,
                          gen_length, step_block, oracle_reduce)

    return x


def _oracle_block(model, x, bs, be, prompt_len, gen_length, step_block, reduce):
    """Decode the block twice: once whole, then against its own label.

    The LLaDA counterpart of ``dream_generate._oracle_block``, and the same
    procedure: pass A keeps the entire cache, so the block settles to the answer
    the teacher label is defined on; one more forward over the completed block
    gives the attention that label is built from; the block then goes back to
    masks and is decoded again against a cache pruned to that label's top-k --
    the same budget the scorer gets, with the answer's own attention standing in
    for a prediction of it.

    Not a guaranteed ceiling: the label is scored against pass A's answer while
    the reported answer comes out of pass B, and the two can diverge.

    No ``attention_mask="full"`` on the capture forward, unlike Dream: LLaDA is
    natively masked and already attends both ways, so there is no causal mask to
    override.
    """
    row_reduce, group_reduce, per_head = reduce
    n_layers = model.config.n_layers
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
    model(x[:, bs:be], bs, 2, full)
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
