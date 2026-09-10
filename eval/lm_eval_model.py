"""lm-eval model for future_dllm, registered as ``LLaDA_future`` and ``Dream_future``.

Self-contained: it subclasses lm-eval's own ``HFLM`` for tokenisation and
plumbing, and replaces generation with future_dllm's block-wise ``generate()``,
so the cache knobs are reachable from ``--model_args``:

    --model_args "pretrained=<model>,keep_ratio=0.1,student_path=<checkpoint>"

``keep_ratio`` below 1.0 needs a trained scorer; 1.0 disables eviction.

Both registered names run the same class; the family comes from the
checkpoint's config, so the name only has to exist for lm-eval's registry.

Multiple-choice tasks use the diffusion Monte Carlo likelihood estimator. With
eviction enabled, each candidate continuation is scored block by block through
the same student-selected sparse cache used during generation.

Dream's autoregressive logit shift applies to both likelihood paths as well as
generation. It matters most here: at keep_ratio=1.0 only
``_full_sequence_logits`` runs, and a missing shift there does not raise -- it
just drags multiple-choice accuracy down toward chance.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from eval.diffusion_likelihood import DiffusionLikelihoodMixin

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "model" / "LLaDA-8B-Instruct"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _generation_kwargs(raw: dict, default_max_gen_toks: int) -> dict:
    """Take the task's own generation settings and read them as diffusion ones.

    Diffusion decoding needs its token budget up front, so the task's
    ``max_gen_toks`` — or lm-eval's default when the task does not set one —
    becomes the block schedule, one denoising step per token.

    ``do_sample: false`` means greedy, which for Gumbel-max sampling is
    temperature 0. Tasks pair it with ``temperature: 1``, meaning "unused";
    taking that literally would sample.
    """
    out = dict(raw)
    gen_length = int(out.get("gen_length", out.get("max_gen_toks", default_max_gen_toks)))
    out["gen_length"] = gen_length
    out.setdefault("steps", gen_length)
    if not out.get("do_sample", False):
        out["temperature"] = 0.0
    return out


@register_model("LLaDA_future", "Dream_future")
class FutureDLLM(DiffusionLikelihoodMixin, HFLM):
    def __init__(
        self,
        pretrained: str = str(DEFAULT_MODEL),
        keep_ratio: float = 1.0,
        block_len: int = 32,
        max_seq_len: int = 4096,
        max_prompt_len: int = 0,
        student_path: str = "",
        selection: str = "student",
        dtype: str = "bfloat16",
        diffusion_steps: int = 32,
        sampling_eps: float = 1e-3,
        nll_type: str = "mc",
        log_type: str = "ftb",
        **kwargs,
    ):
        from future_dllm import load_model, load_prompt_utility_student

        self._block_len = int(block_len)
        self._max_seq_len = int(max_seq_len)
        self._max_prompt_len = int(max_prompt_len) or self._max_seq_len
        self._keep_ratio = float(keep_ratio)
        self._diffusion_steps = int(diffusion_steps)
        self._sampling_eps = float(sampling_eps)
        self._nll_type = str(nll_type)
        self._log_type = str(log_type)

        if not 0.0 < self._keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        if self._max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if int(max_prompt_len) < 0:
            raise ValueError("max_prompt_len must be non-negative")
        if self._diffusion_steps < 1:
            raise ValueError("diffusion_steps must be positive")
        if not 0.0 < self._sampling_eps <= 1.0:
            raise ValueError("sampling_eps must be in (0, 1]")
        if self._nll_type != "mc" or self._log_type != "ftb":
            raise ValueError(
                "this wrapper supports the official diffusion likelihood settings "
                "nll_type=mc,log_type=ftb"
            )

        if str(dtype) not in ("bfloat16", ""):
            raise ValueError("future_dllm backends load in bfloat16")
        self._selection = str(selection)
        model, backend = load_model(
            str(pretrained), max_seq_len=self._max_seq_len,
            block_length=self._block_len, keep_ratio=self._keep_ratio,
            selection=self._selection)
        self._backend = backend
        self._logit_shift = backend.logit_shift
        self._generate = backend.generate
        self._n_layers = backend.n_layers
        self._fallback_mask_id = backend.mask_id
        if self._max_seq_len > backend.native_max_seq_len:
            print(f"warning: max_seq_len={self._max_seq_len} exceeds the checkpoint's "
                  f"trained context {backend.native_max_seq_len}", flush=True)

        # HFLM skips its own loading when handed a live model, but still needs
        # the path to find the tokenizer.
        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)      # cache state is per sequence
        kwargs.setdefault("trust_remote_code", True)
        super().__init__(pretrained=model, **kwargs)

        # This box drops CUDA initialisation now and then, and device_map="auto"
        # answers by placing the model on CPU. That produces answers rather than a
        # crash, and the resume store would keep them - so refuse to start.
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(
                f"model landed on {device}, not CUDA - CUDA init likely failed; "
                "rerun rather than evaluate on CPU")

        self._scorer = None
        if student_path:
            self._scorer = load_prompt_utility_student(student_path, device)
        elif float(keep_ratio) < 1.0 and self._selection != "sparse_dllm":
            raise ValueError(
                "eviction needs a trained scorer: pass student_path=<checkpoint>, "
                "selection=sparse_dllm for the paper baseline, or keep_ratio=1.0 "
                "to run without eviction")
        # _forward_process writes this id into the noised batch, so a wrong one
        # corrupts every likelihood score without raising anywhere.
        tokenizer_mask = getattr(self.tokenizer, "mask_token_id", None)
        if tokenizer_mask is not None and int(tokenizer_mask) != backend.mask_id:
            raise RuntimeError(
                f"tokenizer mask_token_id {int(tokenizer_mask)} disagrees with the "
                f"{backend.name} config's {backend.mask_id}")
        print(f"[{backend.name}_future] selection={self._selection} "
              f"keep_ratio={keep_ratio} block_len={block_len} "
              f"max_seq_len={self._max_seq_len} "
              f"max_prompt_len={self._max_prompt_len} "
              f"logit_shift={backend.logit_shift} "
              f"scorer={student_path or 'none (no eviction)'}", flush=True)

    # -- DiffusionLikelihoodMixin hooks ------------------------------------
    def _shift(self, logits: torch.Tensor) -> torch.Tensor:
        """Move each row's prediction onto the position it describes.

        Dream was adapted from an autoregressive Qwen2, so row r predicts token
        r+1. LLaDA predicts in place and this is the identity.
        """
        if not self._logit_shift:
            return logits
        return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    def _make_cache(self, keep_ratio, prompt_length, generation_length):
        from future_dllm import CustomCache
        return CustomCache(
            n_layers=self._n_layers, device=self.device, keep_ratio=keep_ratio,
            selection=self._selection, cache_scorer=self._scorer,
            prompt_length=prompt_length, generation_length=generation_length)

    def _forward(self, input_ids, position_offset, cache_state, cache):
        return self.model(input_ids, position_offset, cache_state, cache).logits

    @property
    def _mask_id(self) -> int:
        token_id = getattr(self.tokenizer, "mask_token_id", None)
        return self._fallback_mask_id if token_id is None else int(token_id)

    def _call_generate(self, context_enc, gen_kwargs, gen_length):
        return self._generate(
            self.model, context_enc.to(self.device),
            steps=int(gen_kwargs["steps"]), gen_length=gen_length,
            block_length=self._block_len,
            temperature=float(gen_kwargs.get("temperature", 0.0)),
            cfg_scale=float(gen_kwargs.get("cfg_scale", 0.0)),
            remasking=gen_kwargs.get("remasking") or "low_confidence",
            cache_scorer=self._scorer)

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        from tqdm import tqdm

        # lm-eval's own --use_cache only writes once the whole batch returns, so a
        # crash part-way through loses everything. This driver segfaults often
        # enough that a 200-item run rarely finishes, so each answer is appended
        # and fsynced as it is produced and a restart replays what is on disk.
        store_path = os.environ.get("FUTURE_DLLM_RESUME", "")
        done, store = {}, None
        if store_path:
            if os.path.exists(store_path):
                with open(store_path) as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue          # a line the crash cut in half
                        done[rec["key"]] = rec["text"]
            os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
            store = open(store_path, "a")
            print(f"[{self._backend.name}_future] resume store: "
                  f"{len(done)} answers on disk", flush=True)

        results = []
        bar = tqdm(total=len(requests), disable=(disable_tqdm or self.rank != 0),
                   desc="future_dllm generate_until")
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
                    f"within max_seq_len {self._max_seq_len}"
                )
            context_enc, _ = self.tok_batch_encode(
                [context], truncation=self.truncation,
                left_truncate_len=prompt_limit)

            out = self._call_generate(context_enc, gen_kwargs, gen_length)
            text = self.tokenizer.decode(out[0, context_enc.shape[1]:],
                                         skip_special_tokens=True)
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
