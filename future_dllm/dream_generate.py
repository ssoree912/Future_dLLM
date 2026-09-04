"""Block-wise diffusion decoding for Dream, with future-attention cache eviction.

Same three cache states and same per-block cache lifetime as ``llada_generate``.
Two things are Dream's and not LLaDA's, and both come from Dream having been
adapted from an autoregressive Qwen2 rather than trained masked from scratch:

*Shifted logits.* Row ``r`` predicts token ``r+1``, so the token at position
``r`` is read from row ``r-1``. Dream's own ``generation_utils`` applies the
identical one-line shift. Without it the decode still runs and still agrees with
a no-cache reference -- it just emits text offset by one token, which reads as
fluent-looking garbage.

*A seeded first token.* Because of that shift, a block cannot be decoded from
the block's own rows alone: its first token comes from the row before it, which
a block-only forward does not have. Following Sparse-dLLM, step 0 -- a full
sequence forward, where that row exists and is correct -- confirms exactly that
one token before anything else is revealed. From step 2 on, the shift's
meaningless first row lands on an already-confirmed position and is masked out,
so the block-only window stays the block itself and the candidate pool is
identical to the Sparse-dLLM baseline's.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .cache import CustomCache
from .llada_generate import add_gumbel_noise, get_num_transfer_tokens

# Dream-org/Dream-v0-Instruct-7B config.mask_token_id. Prefer the value read off
# the config (``Backend.mask_id``) wherever one is available: LLaDA's 126336 is
# an ordinary token in Dream's 152k vocabulary, so a mismatched constant masks
# nothing and raises nothing.
MASK_ID = 151666


def shift_logits(logits: torch.Tensor) -> torch.Tensor:
    """Move each row's prediction onto the position it describes."""
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=32,
             temperature=0., cfg_scale=0., remasking='low_confidence',
             mask_id=MASK_ID, cache_scorer=None):
    """Generate ``gen_length`` tokens block by block.

    ``cache_scorer`` is a trained ``PromptUtilityStudent``; without one the model
    only runs at ``keep_ratio=1.0`` (no eviction). ``keep_ratio`` and the layer
    count come from ``model.config``.
    """
    # filter_cache cuts using config.block_len, fixed at load time, while the
    # blocks below are built from this call's block_length. If they disagree the
    # cache drops the wrong columns -- and the candidate-count check that would
    # catch it is skipped at keep_ratio=1.0, which is exactly the teacher path.
    assert model.config.block_len == block_length, (
        f"config.block_len={model.config.block_len} was set for a different "
        f"block_length than {block_length}")

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
            n_layers=model.config.num_hidden_layers, device=model.device,
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
            logits = shift_logits(logits)
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
            if cache_state == 0:
                # Seed the block's first token from the one forward that can
                # read it. +inf puts it at the head of this step's top-k rather
                # than adding a reveal, so the schedule's budget is unchanged.
                #
                # At the default steps == gen_length this reveals exactly that
                # one token, matching Sparse-dLLM's step 0. With fewer steps the
                # schedule's first budget is larger, so step 0 also reveals the
                # next num_transfer[0]-1 by confidence where the baseline would
                # reveal only the seed -- worth knowing before comparing runs at
                # a non-default --steps.
                confidence[:, block_start] = np.inf
            for j in range(confidence.shape[0]):
                reveal = torch.topk(confidence[j], k=num_transfer[j, i]).indices
                target[j, reveal] = x0[j, reveal]

    return x
