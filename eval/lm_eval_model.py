"""lm-eval model for future_dllm, registered as ``LLaDA_future``.

Self-contained: it subclasses lm-eval's own ``HFLM`` for tokenisation and
plumbing, and replaces generation with future_dllm's block-wise ``generate()``,
so the cache knobs are reachable from ``--model_args``:

    --model_args "pretrained=<model>,keep_ratio=0.1,student_path=<checkpoint>"

``keep_ratio`` below 1.0 needs a trained scorer; 1.0 disables eviction.

Multiple-choice tasks use LLaDA's diffusion Monte Carlo likelihood estimator.
With eviction enabled, each candidate continuation is scored block by block
through the same student-selected sparse cache used during generation.
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


@register_model("LLaDA_future")
class LLaDAFuture(HFLM):
    def __init__(
        self,
        pretrained: str = str(DEFAULT_MODEL),
        keep_ratio: float = 1.0,
        block_len: int = 32,
        max_seq_len: int = 4096,
        max_prompt_len: int = 0,
        student_path: str = "",
        dtype: str = "bfloat16",
        diffusion_steps: int = 32,
        sampling_eps: float = 1e-3,
        nll_type: str = "mc",
        log_type: str = "ftb",
        **kwargs,
    ):
        from transformers import AutoConfig
        from future_dllm import LLaDAModelLM, generate, load_prompt_utility_student
        from future_dllm.llada_generate import MASK_ID

        self._generate = generate
        self._block_len = int(block_len)
        self._max_seq_len = int(max_seq_len)
        self._max_prompt_len = int(max_prompt_len) or self._max_seq_len
        self._keep_ratio = float(keep_ratio)
        self._diffusion_steps = int(diffusion_steps)
        self._sampling_eps = float(sampling_eps)
        self._nll_type = str(nll_type)
        self._log_type = str(log_type)
        self._fallback_mask_id = MASK_ID

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
                "this wrapper supports the official LLaDA likelihood settings "
                "nll_type=mc,log_type=ftb"
            )

        config = AutoConfig.from_pretrained(str(pretrained), trust_remote_code=True)
        native_limit = int(getattr(config, "max_sequence_length", self._max_seq_len))
        if self._max_seq_len > native_limit:
            print(f"warning: max_seq_len={self._max_seq_len} exceeds the checkpoint's "
                  f"trained context {native_limit}", flush=True)
        config.max_sequence_length = self._max_seq_len
        config.block_len = int(block_len)
        config.keep_ratio = float(keep_ratio)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(str(dtype), torch.bfloat16)
        model = LLaDAModelLM.from_pretrained(
            str(pretrained), config=config, device_map="auto",
            torch_dtype=torch_dtype, trust_remote_code=True).eval()

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
        elif float(keep_ratio) < 1.0:
            raise ValueError(
                "eviction needs a trained scorer: pass student_path=<checkpoint>, "
                "or keep_ratio=1.0 to run without eviction")
        print(f"[LLaDA_future] keep_ratio={keep_ratio} block_len={block_len} "
              f"max_seq_len={self._max_seq_len} "
              f"max_prompt_len={self._max_prompt_len} "
              f"scorer={student_path or 'none (no eviction)'}", flush=True)

    @property
    def _mask_id(self) -> int:
        token_id = getattr(self.tokenizer, "mask_token_id", None)
        return self._fallback_mask_id if token_id is None else int(token_id)

    def _encode_pair(self, context: str, continuation: str) -> Tuple[list[int], list[int]]:
        """Tokenize a request without breaking tokens across the text boundary."""
        trailing_spaces = len(context) - len(context.rstrip())
        if trailing_spaces:
            continuation = context[-trailing_spaces:] + continuation
            context = context[:-trailing_spaces]

        whole = self.tokenizer.encode(
            context + continuation, add_special_tokens=False
        )
        prefix = self.tokenizer.encode(context, add_special_tokens=False)
        target = whole[len(prefix):]
        if self.tokenizer.eos_token_id is not None:
            target.append(int(self.tokenizer.eos_token_id))

        reserved_target = len(target)
        if self._keep_ratio < 1.0:
            reserved_target = (
                (reserved_target + self._block_len - 1) // self._block_len
            ) * self._block_len
        if reserved_target >= self._max_seq_len:
            raise ValueError(
                f"continuation needs {reserved_target} tokens, exceeding "
                f"max_seq_len {self._max_seq_len}"
            )
        prefix_limit = min(self._max_prompt_len, self._max_seq_len - reserved_target)
        prefix = prefix[-prefix_limit:]
        if not prefix:
            prefix = [int(self.prefix_token_id)]
        return prefix, target

    def _forward_process(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample the forward diffusion process used by LLaDA's evaluator."""
        batch_size, seq_len = batch.shape
        offset = torch.rand(1, device=batch.device, dtype=torch.float32)
        strata = torch.arange(batch_size, device=batch.device, dtype=torch.float32)
        time = (offset + strata / batch_size) % 1
        mask_probability = (
            (1.0 - self._sampling_eps) * time + self._sampling_eps
        ).unsqueeze(1).expand(batch_size, seq_len)
        mask = torch.rand((batch_size, seq_len), device=batch.device) < mask_probability
        mask[:, 0] = False
        mask[:, -1] = False
        noisy = torch.where(mask, self._mask_id, batch)
        return noisy, mask_probability

    @torch.no_grad()
    def _full_sequence_logits(self, batch: torch.Tensor) -> torch.Tensor:
        from future_dllm import CustomCache

        cache = CustomCache(
            n_layers=self.model.config.n_layers,
            device=batch.device,
            keep_ratio=1.0,
        )
        # LLaDA denoises token i from logits[i], as in its generation loop.
        return self.model(batch, 0, 0, cache).logits

    @torch.no_grad()
    def _sparse_target_logits(
        self,
        clean: torch.Tensor,
        noisy: torch.Tensor,
        prefix_length: int,
        target_length: int,
    ) -> torch.Tensor:
        """Score a candidate using generation-equivalent sparse block states."""
        from future_dllm import CustomCache

        if clean.shape[0] != 1:
            raise RuntimeError("sparse likelihood requires batch_size=1")

        generation_length = (
            (target_length + self._block_len - 1) // self._block_len
        ) * self._block_len
        padding = generation_length - target_length
        if padding:
            pad = torch.full(
                (1, padding), self._mask_id, dtype=clean.dtype, device=clean.device
            )
            clean = torch.cat([clean, pad], dim=1)
            noisy = torch.cat([noisy, pad], dim=1)

        block_logits = []
        for local_start in range(0, generation_length, self._block_len):
            block_start = prefix_length + local_start
            block_end = block_start + self._block_len

            # Match block-wise generation: completed blocks are confirmed and
            # blocks after the current one have not begun denoising yet.
            model_input = noisy.clone()
            model_input[:, prefix_length:block_start] = clean[
                :, prefix_length:block_start
            ]
            model_input[:, block_end:] = self._mask_id

            cache = CustomCache(
                n_layers=self.model.config.n_layers,
                device=model_input.device,
                keep_ratio=self._keep_ratio,
                cache_scorer=self._scorer,
                prompt_length=prefix_length,
                generation_length=generation_length,
            )
            self.model(model_input, block_start, 1, cache)
            logits = self.model(
                model_input[:, block_start:block_end], block_start, 2, cache
            ).logits
            valid_length = min(self._block_len, target_length - local_start)
            block_logits.append(logits[:, :valid_length])

        return torch.cat(block_logits, dim=1)

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix: list[int], target: list[int]) -> float:
        sequence = torch.tensor(prefix + target, dtype=torch.long, device=self.device)
        sparse = self._keep_ratio < 1.0
        likelihood_batch_size = 1 if sparse else int(self.batch_size)
        accumulated_loss = 0.0
        completed = 0

        while completed < self._diffusion_steps:
            current_batch_size = min(
                likelihood_batch_size, self._diffusion_steps - completed
            )
            clean = sequence.unsqueeze(0).repeat(current_batch_size, 1)
            noisy, mask_probability = self._forward_process(clean)
            perturbed = clean.clone()
            perturbed[:, -len(target):] = noisy[:, -len(target):]
            masked = perturbed.eq(self._mask_id)

            if sparse:
                logits = self._sparse_target_logits(
                    clean,
                    perturbed,
                    prefix_length=len(prefix),
                    target_length=len(target),
                )
                labels = clean[:, -len(target):]
                masked = masked[:, -len(target):]
                mask_probability = mask_probability[:, -len(target):]
            else:
                logits = self._full_sequence_logits(perturbed)
                labels = clean
            token_loss = F.cross_entropy(
                logits[masked], labels[masked], reduction="none"
            )
            weighted_loss = token_loss / mask_probability[masked]
            batch_loss = weighted_loss.sum() / current_batch_size
            accumulated_loss += float(batch_loss) * current_batch_size
            completed += current_batch_size

        return accumulated_loss / completed

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Score continuations with full or student-pruned diffusion NLL."""
        from tqdm import tqdm

        results = []
        iterator = tqdm(
            requests,
            disable=self.rank != 0,
            desc="LLaDA diffusion loglikelihood",
        )
        for request in iterator:
            prefix, target = self._encode_pair(*request.args)
            nll = self._eval_target_nll_mc(prefix, target)
            results.append((-nll, False))
        return results

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "rolling likelihood is not defined for the LLaDA diffusion evaluator"
        )

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
            print(f"[LLaDA_future] resume store: {len(done)} answers on disk", flush=True)

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
