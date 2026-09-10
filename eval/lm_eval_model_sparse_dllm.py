"""lm-eval model for the Sparse-dLLM baseline, registered as ``Sparse_dLLM_Dream``.

This drives *their* code. ``baselines/sparse_dllm/dream/`` is
OpenMOSS/Sparse-dLLM's Dream implementation -- their ``CustomCache``, their
eviction, their block schedule, their ``diffusion_generate`` -- carrying one
six-line addition documented in ``baselines/sparse_dllm/DIFF.md``. This file
replaces OpenCompass with lm-eval so the numbers land in our tasks, truncation
and result files.

Both the baseline and our method run through here, at their published Dream
settings (``alg="entropy"``, temperature 0.2, top_p 0.95). Running ours on our
old greedy ``low_confidence`` loop instead was tried and abandoned: aligning
the decode piece by piece never reached equality, because their reveal count is
recomputed from the timestep schedule each step where ours is fixed per block.
Four GSM8K prompts at keep_ratio=1.0 matched on only 63/64, 58/64, 13/64 and
57/64 tokens. Sampling makes runs seed-dependent, so seed as they do (2025).

`student_path` selects which ranking rule runs inside their cache:

    (unset)                 their attention score      -> the baseline row
    student_path=<ckpt>     the trained scorer         -> our row

Both execute the same files. At ``keep_ratio=1.0`` the scorer is never reached
and the two are bit-identical -- verified on five GSM8K prompts -- so one
no-eviction run serves as the reference for both.

Multiple choice follows lm-eval's own convention for `output_type:
multiple_choice` -- score every choice by loglikelihood, take the largest --
through the shared `DiffusionLikelihoodMixin`. Sparse-dLLM scores those
benchmarks generatively in OpenCompass instead, so this column is not directly
comparable to their published numbers; it is comparable between the rows here,
which is what the table needs. Nothing about decoding enters that path: the
estimator only masks and runs forwards, so `alg`, temperature and the reveal
schedule never apply and the two rows differ in eviction alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.diffusion_likelihood import DiffusionLikelihoodMixin  # noqa: E402
from eval.lm_eval_model import _generation_kwargs  # noqa: E402


@register_model("Sparse_dLLM_Dream")
class SparseDLLMDream(DiffusionLikelihoodMixin, HFLM):
    def __init__(
        self,
        pretrained: str,
        keep_ratio: float = 0.5,
        kernel_size: int = 3,
        block_len: int = 32,
        max_seq_len: int = 2048,
        max_prompt_len: int = 0,
        student_path: str = "",
        alg: str = "entropy",
        temperature: float = 0.2,
        top_p: float = 0.95,
        alg_temp: float = 0.0,
        diffusion_steps: int = 32,
        sampling_eps: float = 1e-3,
        **kwargs,
    ):
        from future_dllm import load_prompt_utility_student
        from future_dllm.sparse_dllm_student import load_model as load_their_model
        from future_dllm.sparse_dllm_student import set_scorer

        self._block_len = int(block_len)
        self._max_seq_len = int(max_seq_len)
        self._max_prompt_len = int(max_prompt_len) or self._max_seq_len
        self._keep_ratio = float(keep_ratio)
        self._alg = str(alg)
        self._temperature = float(temperature)
        self._top_p = None if top_p is None or float(top_p) <= 0 else float(top_p)
        self._alg_temp = float(alg_temp)
        self._diffusion_steps = int(diffusion_steps)
        self._sampling_eps = float(sampling_eps)
        self._logit_shift = True            # Dream predicts token r+1 from row r

        if not 0.0 < self._keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")

        # Their config knobs, set the way their own wrapper sets them.
        model = load_their_model(str(pretrained), block_length=self._block_len,
                                 keep_ratio=self._keep_ratio,
                                 kernel_size=int(kernel_size))
        model.generation_config.do_sample = self._temperature > 0

        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)          # cache state is per sequence
        kwargs.setdefault("trust_remote_code", True)
        super().__init__(pretrained=model, **kwargs)

        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(
                f"model landed on {device}, not CUDA - rerun rather than "
                "evaluate on CPU")

        # Chooses what the caches their generate builds will rank with.
        scorer = load_prompt_utility_student(student_path, device) if student_path else None
        set_scorer(scorer)
        self._scorer = scorer
        self._kernel_size = int(kernel_size)
        self._scorer_path = student_path or "none (their attention score)"

        print(f"[Sparse_dLLM_Dream] scorer={self._scorer_path} "
              f"keep_ratio={self._keep_ratio} "
              f"kernel_size={kernel_size} block_len={self._block_len} "
              f"max_seq_len={self._max_seq_len} alg={self._alg} "
              f"temperature={self._temperature} top_p={self._top_p}", flush=True)

    # -- DiffusionLikelihoodMixin hooks ------------------------------------
    def _shift(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    @property
    def _mask_id(self) -> int:
        return int(self.model.config.mask_token_id)

    @property
    def _n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def _make_cache(self, keep_ratio, prompt_length, generation_length):
        """Their cache, or their cache with the scorer, exactly as generate does."""
        from baselines.sparse_dllm.dream.Cache import CustomCache
        from future_dllm.sparse_dllm_student import StudentScorerCache
        if self._scorer is None:
            return CustomCache(n_layers=self._n_layers, device=self.device,
                               kernel_size=self._kernel_size, keep_ratio=keep_ratio)
        return StudentScorerCache(n_layers=self._n_layers, device=self.device,
                                  kernel_size=self._kernel_size,
                                  keep_ratio=keep_ratio, cache_scorer=self._scorer)

    def _forward(self, input_ids, position_offset, cache_state, cache):
        return self.model(position_offset=position_offset, cache_state=cache_state,
                          customcache=cache, input_ids=input_ids).logits

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        from tqdm import tqdm

        # Same crash-resume store as the future_dllm wrapper: answers are
        # appended and fsynced as they are produced, so a killed run replays
        # what is on disk instead of starting over.
        store_path = os.environ.get("FUTURE_DLLM_RESUME", "")
        done, store = {}, None
        if store_path:
            if os.path.exists(store_path):
                with open(store_path) as fh:
                    for line in fh:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        done[record["key"]] = record["text"]
            os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
            store = open(store_path, "a")
            print(f"[Sparse_dLLM_Dream] resume store: {len(done)} answers on disk",
                  flush=True)

        results = []
        bar = tqdm(total=len(requests), disable=(disable_tqdm or self.rank != 0),
                   desc="sparse_dllm generate_until")
        for request in requests:
            context, raw_kwargs = request.args
            key = hashlib.md5(
                (context + repr(sorted(raw_kwargs.items()))).encode()).hexdigest()
            if key in done:
                results.append(done[key])
                bar.update(1)
                continue

            gen_kwargs = _generation_kwargs(raw_kwargs, self.max_gen_toks)
            gen_length = int(gen_kwargs["gen_length"])
            if gen_length % self._block_len:      # blocks have to divide the budget
                gen_length += self._block_len - gen_length % self._block_len

            if self.add_bos_token:
                context = self.tokenizer.bos_token + context
            prompt_limit = min(self._max_prompt_len, self._max_seq_len - gen_length)
            if prompt_limit < 1:
                raise ValueError(
                    f"generation length {gen_length} leaves no prompt space "
                    f"within max_seq_len {self._max_seq_len}")
            context_enc, _ = self.tok_batch_encode(
                [context], truncation=self.truncation,
                left_truncate_len=prompt_limit)

            # Their entry point, with their own block-wise cache and eviction.
            out = self.model.diffusion_generate(
                context_enc.to(self.device),
                max_new_tokens=gen_length,
                steps=gen_length,
                block_length=self._block_len,
                temperature=self._temperature,
                top_p=self._top_p,
                alg=self._alg,
                alg_temp=self._alg_temp,
                return_dict_in_generate=True,
                output_history=False,
            )
            text = self.tokenizer.decode(
                out.sequences[0, context_enc.shape[1]:], skip_special_tokens=True)
            for term in gen_kwargs.get("until") or []:
                if term:
                    text = text.split(term)[0]
            results.append(text)
            if store is not None:
                store.write(json.dumps({"key": key, "text": text}) + "\n")
                store.flush()
                os.fsync(store.fileno())
            bar.update(1)
        bar.close()
        if store is not None:
            store.close()
        return results
