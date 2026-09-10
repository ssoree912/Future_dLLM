"""lm-eval multiple-choice scoring for diffusion LMs.

lm-eval scores `output_type: multiple_choice` by taking each choice's
loglikelihood and picking the largest -- that convention is what the local_mc
tasks are written against, and it is what this implements.

A diffusion LM cannot give that loglikelihood the way an autoregressive model
does, so the value comes from LLaDA's published Monte Carlo estimator of the
ELBO (`nll_type=mc`, `log_type=ftb`): mask the continuation at a sampled rate
`t`, score the masked positions, and weight by `1/t`. Averaging over
`diffusion_steps` draws gives an unbiased estimate.

Both eval wrappers use this one implementation, so the estimator cannot differ
between a Sparse-dLLM baseline row and ours -- only the cache eviction does.
A subclass supplies three things:

    _make_cache(keep_ratio, prompt_length, generation_length)
    _forward(input_ids, position_offset, cache_state, cache)  -> logits
    _shift(logits)                                            -> logits

`_shift` exists because Dream predicts token r+1 from row r; LLaDA predicts in
place and returns the tensor unchanged.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance


class DiffusionLikelihoodMixin:
    """Requires _make_cache / _forward / _shift and the usual length knobs."""

    # -- tokenisation ------------------------------------------------------
    def _encode_pair(self, context: str, continuation: str) -> Tuple[list[int], list[int]]:
        """Tokenize a request without breaking tokens across the text boundary."""
        trailing_spaces = len(context) - len(context.rstrip())
        if trailing_spaces:
            continuation = context[-trailing_spaces:] + continuation
            context = context[:-trailing_spaces]

        whole = self.tokenizer.encode(context + continuation, add_special_tokens=False)
        prefix = self.tokenizer.encode(context, add_special_tokens=False)
        target = whole[len(prefix):]
        if self.tokenizer.eos_token_id is not None:
            target.append(int(self.tokenizer.eos_token_id))

        reserved_target = len(target)
        if self._keep_ratio < 1.0:
            reserved_target = ((reserved_target + self._block_len - 1)
                               // self._block_len) * self._block_len
        if reserved_target >= self._max_seq_len:
            raise ValueError(f"continuation needs {reserved_target} tokens, "
                             f"exceeding max_seq_len {self._max_seq_len}")
        prefix_limit = min(self._max_prompt_len, self._max_seq_len - reserved_target)
        prefix = prefix[-prefix_limit:]
        if not prefix:
            prefix = [int(self.prefix_token_id)]
        return prefix, target

    # -- the estimator -----------------------------------------------------
    def _forward_process(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample the forward diffusion process used by LLaDA's evaluator."""
        batch_size, seq_len = batch.shape
        offset = torch.rand(1, device=batch.device, dtype=torch.float32)
        strata = torch.arange(batch_size, device=batch.device, dtype=torch.float32)
        time = (offset + strata / batch_size) % 1
        mask_probability = ((1.0 - self._sampling_eps) * time + self._sampling_eps
                            ).unsqueeze(1).expand(batch_size, seq_len)
        mask = torch.rand((batch_size, seq_len), device=batch.device) < mask_probability
        mask[:, 0] = False
        mask[:, -1] = False
        return torch.where(mask, self._mask_id, batch), mask_probability

    @torch.no_grad()
    def _full_sequence_logits(self, batch: torch.Tensor) -> torch.Tensor:
        cache = self._make_cache(keep_ratio=1.0, prompt_length=0, generation_length=0)
        # Token i is denoised from row i, once the backend's shift has put each
        # row on the position it describes.
        return self._shift(self._forward(batch, 0, 0, cache))

    @torch.no_grad()
    def _sparse_target_logits(self, clean: torch.Tensor, noisy: torch.Tensor,
                              prefix_length: int, target_length: int) -> torch.Tensor:
        """Score a candidate through generation-equivalent sparse block states."""
        if clean.shape[0] != 1:
            raise RuntimeError("sparse likelihood requires batch_size=1")

        generation_length = ((target_length + self._block_len - 1)
                             // self._block_len) * self._block_len
        padding = generation_length - target_length
        if padding:
            pad = torch.full((1, padding), self._mask_id, dtype=clean.dtype,
                             device=clean.device)
            clean = torch.cat([clean, pad], dim=1)
            noisy = torch.cat([noisy, pad], dim=1)

        block_logits = []
        for local_start in range(0, generation_length, self._block_len):
            block_start = prefix_length + local_start
            block_end = block_start + self._block_len

            # Match block-wise generation: completed blocks are confirmed and
            # blocks after the current one have not begun denoising yet.
            model_input = noisy.clone()
            model_input[:, prefix_length:block_start] = clean[:, prefix_length:block_start]
            model_input[:, block_end:] = self._mask_id

            cache = self._make_cache(keep_ratio=self._keep_ratio,
                                     prompt_length=prefix_length,
                                     generation_length=generation_length)
            full = self._shift(self._forward(model_input, block_start, 1, cache))
            logits = self._shift(self._forward(
                model_input[:, block_start:block_end], block_start, 2, cache))
            if self._logit_shift:
                # Under the shift a block cannot supply its own first row: that
                # token is described by the row before the block, which the
                # block-only forward does not hold. The full forward just run to
                # build the cache does hold it.
                logits = torch.cat([full[:, block_start:block_start + 1],
                                    logits[:, 1:]], dim=1)
            valid_length = min(self._block_len, target_length - local_start)
            block_logits.append(logits[:, :valid_length])
        return torch.cat(block_logits, dim=1)

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix: list[int], target: list[int]) -> float:
        sequence = torch.tensor(prefix + target, dtype=torch.long, device=self.device)
        sparse = self._keep_ratio < 1.0
        likelihood_batch_size = 1 if sparse else int(self.batch_size)
        accumulated_loss, completed = 0.0, 0

        while completed < self._diffusion_steps:
            current_batch_size = min(likelihood_batch_size,
                                     self._diffusion_steps - completed)
            clean = sequence.unsqueeze(0).repeat(current_batch_size, 1)
            noisy, mask_probability = self._forward_process(clean)
            perturbed = clean.clone()
            perturbed[:, -len(target):] = noisy[:, -len(target):]
            masked = perturbed.eq(self._mask_id)

            if sparse:
                logits = self._sparse_target_logits(
                    clean, perturbed, prefix_length=len(prefix),
                    target_length=len(target))
                labels = clean[:, -len(target):]
                masked = masked[:, -len(target):]
                mask_probability = mask_probability[:, -len(target):]
            else:
                logits = self._full_sequence_logits(perturbed)
                labels = clean

            token_loss = F.cross_entropy(logits[masked], labels[masked], reduction="none")
            weighted_loss = token_loss / mask_probability[masked]
            accumulated_loss += float(weighted_loss.sum() / current_batch_size) * current_batch_size
            completed += current_batch_size
        return accumulated_loss / completed

    # -- lm-eval entry point -----------------------------------------------
    def loglikelihood(self, requests: List[Instance], disable_tqdm: bool = False):
        from tqdm import tqdm

        results = []
        for request in tqdm(requests, disable=(disable_tqdm or self.rank != 0),
                            desc="diffusion loglikelihood"):
            prefix, target = self._encode_pair(*request.args)
            results.append((-self._eval_target_nll_mc(prefix, target), False))
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "rolling likelihood is not defined for the diffusion evaluator")
