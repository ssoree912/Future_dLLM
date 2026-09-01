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
             mask_id=MASK_ID, cache_scorer=None):
    """Generate ``gen_length`` tokens block by block.

    ``cache_scorer`` is a trained ``PromptUtilityStudent``; without one the model
    only runs at ``keep_ratio=1.0`` (no eviction). ``keep_ratio`` comes from
    ``model.config``.
    """
    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long,
                   device=model.device)
    x[:, :prompt_len] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        # A fresh cache per block: nothing is carried over, so the selection is
        # made once against the block that will use it.
        cache = CustomCache(
            n_layers=model.config.n_layers, device=model.device,
            keep_ratio=model.config.keep_ratio,
            cache_scorer=cache_scorer, prompt_length=prompt_len,
            generation_length=gen_length)

        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        num_transfer = get_num_transfer_tokens(
            x[:, block_start:block_end] == mask_id, steps_per_block)

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

    return x
