"""lm-eval model that runs dLLM-Cache's LLaDA decoding, registered as ``LLaDA_dllmcache``.

Same shape as ``lm_eval_model_fastdllm.py``: the harness, the task definitions and
the local parquet data all come from ``Future_dLLM/eval``; only the decoding loop
and the cache come from ``dLLM-cache`` - its ``utils/generate_function.generate``
plus the adaptive feature cache (``dllm_cache``), registered onto the checkpoint's
own remote-code transformer blocks by ``register_cache_LLaDA``.

    --model_args "pretrained=<model>,gen_length=256,steps=256,block_length=32,is_feature_cache=True,prompt_interval_steps=50,gen_interval_steps=7,transfer_ratio=0.25"

``is_feature_cache=False`` gives the uncached baseline, which is what dLLM-Cache's
own scripts compare against.

Chat template is off by default here, matching this repo's ``run_eval.sh`` rather
than dLLM-Cache's scripts (which pass ``--apply_chat_template
--fewshot_as_multiturn``), so the numbers sit in the same table as the other rows.

dLLM-Cache raises ``NotImplementedError`` for likelihood, so the multiple-choice
tasks here use the same Monte-Carlo diffusion estimator as the Fast-dLLM row -
plain uncached forwards, which the feature cache does not touch either way.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

REPO_ROOT = Path(__file__).resolve().parent.parent          # Future_dLLM/
WORKSPACE = REPO_ROOT.parent                                # /workspace/dllm
DEFAULT_MODEL = REPO_ROOT / "model" / "LLaDA-8B-Instruct"
DEFAULT_DLLMCACHE = WORKSPACE / "dLLM-cache"

_CHAT_SENTINEL = "DLLMCACHE_SENTINEL_BODY"
_BLOCK_MODULE = "model.transformer.blocks"


def _import_dllmcache(root: Path):
    """Import dLLM-Cache's generate function and cache machinery.

    The repo exposes very generic top-level names (``utils``, ``metrics``,
    ``eval_model``), and the task yamls resolve ``!function metrics....`` against
    the scratch tasks directory, so the path goes on the *end* of sys.path and the
    resolved module files are checked to be the ones intended.
    """
    root = Path(root).resolve()
    if not (root / "utils" / "generate_function.py").is_file():
        raise FileNotFoundError(f"no utils/generate_function.py under {root}")
    if str(root) not in sys.path:
        sys.path.append(str(root))

    import utils.generate_function as gen_module              # noqa: E402
    from dllm_cache.cache import dLLMCache, dLLMCacheConfig   # noqa: E402
    from dllm_cache.hooks import register_cache_LLaDA         # noqa: E402

    resolved = Path(gen_module.__file__).resolve()
    if root not in resolved.parents:
        raise RuntimeError(
            f"'utils' resolved to {resolved}, not dLLM-Cache's - a name collision")
    return gen_module.generate, dLLMCache, dLLMCacheConfig, register_cache_LLaDA


def _round_up(value: int, multiple: int) -> int:
    remainder = value % multiple
    return value if remainder == 0 else value + multiple - remainder


@register_model("LLaDA_dllmcache")
class LLaDADLLMCache(HFLM):
    def __init__(
        self,
        pretrained: str = str(DEFAULT_MODEL),
        dllmcache_path: str = str(DEFAULT_DLLMCACHE),
        gen_length: int = 256,
        steps: int = 0,
        block_length: int = 32,
        is_feature_cache: bool = True,
        is_cfg_cache: bool = False,
        prompt_interval_steps: int = 50,
        gen_interval_steps: int = 7,
        cfg_interval_steps: int = 1,
        transfer_ratio: float = 0.25,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        mask_id: int = 126336,
        max_seq_len: int = 4096,
        max_prompt_len: int = 0,
        dtype: str = "bfloat16",
        device: str = "cuda",
        chat_template: bool = False,
        show_speed: bool = True,
        mc_num: int = 32,
        mc_batch_size: int = 4,
        **kwargs,
    ):
        import transformers

        generate, dLLMCache, dLLMCacheConfig, register_cache_LLaDA = _import_dllmcache(
            Path(dllmcache_path))
        self._generate = generate

        self._gen_length = int(gen_length)
        self._steps = int(steps)
        self._block_length = int(block_length)
        self._is_feature_cache = bool(is_feature_cache)
        self._is_cfg_cache = bool(is_cfg_cache)
        self._cfg_scale = float(cfg_scale)
        self._remasking = str(remasking)
        self._mask_id = int(mask_id)
        self._max_seq_len = int(max_seq_len)
        self._max_prompt_len = int(max_prompt_len) or self._max_seq_len
        self._chat_template = bool(chat_template)
        self._show_speed = bool(show_speed)
        self._mc_num = int(mc_num)
        self._mc_batch_size = int(mc_batch_size)

        if self._block_length < 1:
            raise ValueError("block_length must be positive")
        if self._gen_length < 0:
            raise ValueError("gen_length must be non-negative (0 = take it from the task)")
        if self._steps < 0:
            raise ValueError("steps must be non-negative (0 = one step per token)")
        if self._max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if int(max_prompt_len) < 0:
            raise ValueError("max_prompt_len must be non-negative")
        if self._mc_num % self._mc_batch_size:
            raise ValueError("mc_num must be a multiple of mc_batch_size")

        # The cache hooks patch the checkpoint's own remote-code blocks
        # (tf_block.attention, tf_block.rotary_emb.forward), so the model has to be
        # the checkpoint's class, not another repo's vendored copy of it.
        config = transformers.AutoConfig.from_pretrained(
            str(pretrained), trust_remote_code=True)
        native_limit = int(getattr(config, "max_sequence_length", self._max_seq_len))
        if self._max_seq_len > native_limit:
            print(f"warning: max_seq_len={self._max_seq_len} exceeds the checkpoint's "
                  f"trained context {native_limit}", flush=True)
        config.max_sequence_length = self._max_seq_len
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(str(dtype), torch.bfloat16)
        model = transformers.AutoModel.from_pretrained(
            str(pretrained), config=config, torch_dtype=torch_dtype,
            trust_remote_code=True).to(str(device)).eval()

        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)   # the cache keeps per-sequence state
        kwargs.setdefault("trust_remote_code", True)
        super().__init__(pretrained=model, **kwargs)

        # This box drops CUDA initialisation now and then and the model then lands
        # on CPU, which answers slowly rather than crashing - and the resume store
        # would keep those answers. Refuse to start instead.
        model_device = next(model.parameters()).device
        if model_device.type != "cuda":
            raise RuntimeError(
                f"model landed on {model_device}, not CUDA - CUDA init likely failed; "
                "rerun rather than evaluate on CPU")

        # register_cache_LLaDA leaves its target as None and then fails on iteration
        # if the module name does not match, so check it here with a clear message.
        if self._is_feature_cache and _BLOCK_MODULE not in dict(model.named_modules()):
            raise RuntimeError(
                f"'{_BLOCK_MODULE}' not found in the checkpoint - dLLM-Cache's hooks "
                "cannot be registered")

        if self._is_feature_cache:
            dLLMCache.new_instance(
                prompt_interval_steps=int(prompt_interval_steps),
                gen_interval_steps=int(gen_interval_steps),
                transfer_ratio=float(transfer_ratio),
                cfg_interval_steps=int(cfg_interval_steps) if is_cfg_cache else 1,
            )
            register_cache_LLaDA(model, _BLOCK_MODULE)
        else:
            dLLMCache.new_instance(
                prompt_interval_steps=1,
                gen_interval_steps=1,
                transfer_ratio=0,
                cfg_interval_steps=int(cfg_interval_steps) if is_cfg_cache else 1,
            )

        self._chat_prefix_ids, self._chat_suffix_ids = self._split_chat_template()

        cache_desc = (f"feature_cache prompt/{prompt_interval_steps} gen/{gen_interval_steps} "
                      f"transfer_ratio={transfer_ratio}" if self._is_feature_cache
                      else "no feature cache")
        print(f"[LLaDA_dllmcache] {cache_desc} cfg_cache={self._is_cfg_cache} "
              f"gen_length={self._gen_length or 'task'} steps={self._steps or 'gen_length'} "
              f"block_length={self._block_length} max_seq_len={self._max_seq_len} "
              f"max_prompt_len={self._max_prompt_len} "
              f"chat_template={self._chat_template}", flush=True)

    # ------------------------------------------------------------------ prompts

    def _split_chat_template(self) -> Tuple[list, list]:
        """Tokenise the instruct template once, around a sentinel body."""
        if not self._chat_template:
            return [], []
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": _CHAT_SENTINEL}],
            add_generation_prompt=True, tokenize=False)
        head, sep, tail = rendered.partition(_CHAT_SENTINEL)
        if not sep:
            raise RuntimeError("chat template did not render the sentinel body")
        return self.tokenizer(head)["input_ids"], self.tokenizer(tail)["input_ids"]

    def _encode_prompt(self, context: str, prompt_limit: int) -> torch.Tensor:
        if self._chat_template:
            body_limit = prompt_limit - len(self._chat_prefix_ids) - len(self._chat_suffix_ids)
            if body_limit < 1:
                raise ValueError(f"prompt budget {prompt_limit} does not fit the chat template")
            body = self.tokenizer(context)["input_ids"][-body_limit:]
            ids = self._chat_prefix_ids + body + self._chat_suffix_ids
        else:
            if self.add_bos_token and self.tokenizer.bos_token:
                context = self.tokenizer.bos_token + context
            ids = self.tokenizer(context)["input_ids"][-prompt_limit:]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    # --------------------------------------------------------------- generation

    def _schedule(self, raw_kwargs: dict) -> Tuple[int, int]:
        """Resolve (gen_length, steps).

        dLLM-Cache's loop is a fixed ``range(steps // num_blocks)`` with a per-step
        token quota and no early exit, so ``steps=0`` (one step per token) makes the
        quota exactly one token per step, as in their own scripts.
        """
        gen_length = self._gen_length
        if gen_length <= 0:
            gen_length = int(raw_kwargs.get(
                "gen_length", raw_kwargs.get("max_gen_toks", self.max_gen_toks)))
        gen_length = _round_up(gen_length, self._block_length)   # generate() asserts this
        steps = self._steps or gen_length
        num_blocks = gen_length // self._block_length
        if steps % num_blocks:
            steps = _round_up(steps, num_blocks)                 # and this
        return gen_length, steps

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        from tqdm import tqdm

        # lm-eval's own --use_cache only writes once the whole batch returns, so a
        # crash part-way through loses everything. Each answer is appended and
        # fsynced as it is produced, and a restart replays what is on disk.
        store_path = os.environ.get("DLLMCACHE_RESUME", "")
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
            print(f"[LLaDA_dllmcache] resume store: {len(done)} answers on disk", flush=True)

        results = []
        total_tokens, generated = 0, 0
        started = time.time()
        bar = tqdm(total=len(requests), disable=(disable_tqdm or self.rank != 0),
                   desc="dllmcache generate_until")
        for request in requests:
            context, raw_kwargs = request.args
            key = hashlib.md5(
                (context + repr(sorted(raw_kwargs.items()))).encode()).hexdigest()
            if key in done:
                results.append(done[key])
                bar.update(1)
                continue

            gen_length, steps = self._schedule(raw_kwargs)
            prompt_limit = min(self._max_prompt_len, self._max_seq_len - gen_length)
            if prompt_limit < 1:
                raise ValueError(
                    f"generation length {gen_length} leaves no prompt space "
                    f"within max_seq_len {self._max_seq_len}")
            prompt_ids = self._encode_prompt(context, prompt_limit)
            attention_mask = torch.ones_like(prompt_ids)

            # dLLM-Cache's generate returns the generated span only, not the
            # prompt-plus-answer sequence.
            out = self._generate(
                input_ids=prompt_ids, attention_mask=attention_mask, model=self.model,
                steps=steps, gen_length=gen_length, block_length=self._block_length,
                temperature=0.0, cfg_scale=self._cfg_scale, remasking=self._remasking,
                mask_id=self._mask_id)

            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            for term in raw_kwargs.get("until") or []:
                if term:
                    text = text.split(term)[0]
            total_tokens += len(self.tokenizer(text)["input_ids"])
            generated += 1

            results.append(text)
            if store is not None:
                store.write(json.dumps({"key": key, "text": text}) + "\n")
                store.flush()
                os.fsync(store.fileno())
            bar.update(1)
        bar.close()
        if store is not None:
            store.close()

        if self._show_speed and generated:
            elapsed = time.time() - started
            print(f"[LLaDA_dllmcache] generated {generated} answers, {total_tokens} tokens "
                  f"in {elapsed:.1f}s ({total_tokens / elapsed:.2f} tok/s)", flush=True)
        return results

    # --------------------------------------------------------------- likelihood

    def _forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor):
        """The same forward diffusion the Fast-dLLM row's likelihood uses."""
        b, l = batch.shape
        target_len = int(l - prompt_index.sum())
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b),
                                       steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len, device=batch.device)]
        is_mask = torch.cat(
            (torch.zeros(b, int(prompt_index.sum()), dtype=torch.bool, device=batch.device),
             is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self._mask_id, batch)
        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def _get_loglikelihood(self, prefix: list, target: list) -> float:
        seq = torch.tensor(prefix + target, dtype=torch.long,
                           device=self.device).unsqueeze(0)
        seq = seq.repeat((self._mc_batch_size, 1))
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        losses = []
        for _ in range(self._mc_num // self._mc_batch_size):
            perturbed, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed == self._mask_id
            logits = self.model(perturbed).logits[:, :perturbed.shape[1]]
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices],
                                   reduction="none") / p_mask[mask_indices]
            losses.append(float(loss.sum() / self._mc_batch_size))
        return -sum(losses) / len(losses)

    def _encode_pair(self, context: str, continuation: str) -> Tuple[list, list]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        continuation_enc = whole_enc[len(context_enc):]

        keep = self._max_seq_len - len(continuation_enc)
        if keep < 1:
            raise ValueError(
                f"continuation needs {len(continuation_enc)} tokens, exceeding "
                f"max_seq_len {self._max_seq_len}")
        context_enc = context_enc[-min(keep, self._max_prompt_len):]
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Monte-Carlo diffusion likelihood; dLLM-Cache itself does not implement one.

        Uncached full forwards, matching the Fast-dLLM row so the two are
        comparable. The feature cache only applies to the denoising loop, so it
        would not change these numbers anyway.
        """
        from tqdm import tqdm

        out = []
        for request in tqdm(requests, disable=self.rank != 0,
                            desc="dllmcache diffusion loglikelihood"):
            prefix, target = self._encode_pair(*request.args)
            out.append((self._get_loglikelihood(prefix, target), False))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "rolling likelihood is not defined for the LLaDA diffusion evaluator")
