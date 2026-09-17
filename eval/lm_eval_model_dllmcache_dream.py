"""lm-eval model that runs dLLM-Cache's Dream decoding, registered as ``Dream_dllmcache``.

The Dream counterpart of ``lm_eval_model_dllmcache.py``. Dream ships its own
``diffusion_generate``, so unlike LLaDA there is no separate generate function to
import - the method is the cache alone: ``register_cache_Dream`` hooks the
checkpoint's ``model.layers`` and the adaptive feature cache reuses block outputs
across denoising steps.

    --model_args "pretrained=<model>,max_new_tokens=256,diffusion_steps=256,is_feature_cache=True,prompt_interval_steps=25,gen_interval_steps=2,transfer_ratio=0.25"

``is_feature_cache=False`` gives the uncached baseline their scripts compare
against. ``add_bos_token`` defaults to True, matching dLLM-Cache's Dream scripts.

Likelihood is not implemented - dLLM-Cache's own Dream wrapper raises there too,
and this repo's Dream branch records that lm-eval's diffusion loglikelihood gives
near-chance numbers for Dream, so multiple choice is excluded.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

REPO_ROOT = Path(__file__).resolve().parent.parent          # Future_dLLM/
WORKSPACE = REPO_ROOT.parent                                # /workspace/dllm
DEFAULT_MODEL = REPO_ROOT / "model" / "Dream-v0-Instruct-7B"
DEFAULT_DLLMCACHE = WORKSPACE / "dLLM-cache"

_CHAT_SENTINEL = "DLLMCACHE_SENTINEL_BODY"
_BLOCK_MODULE = "model.layers"


def _import_dllmcache_dream(root: Path):
    """Import dLLM-Cache's cache machinery, checking it is really theirs.

    The repo exposes generic top-level names (``utils``, ``metrics``), and the
    task yamls resolve ``!function metrics....`` against the scratch tasks
    directory, so the path goes on the end of sys.path.
    """
    root = Path(root).resolve()
    if not (root / "dllm_cache").is_dir():
        raise FileNotFoundError(f"no dllm_cache package under {root}")
    if str(root) not in sys.path:
        sys.path.append(str(root))

    from dllm_cache import cache as cache_module                # noqa: E402
    from dllm_cache.cache import dLLMCache                      # noqa: E402
    from dllm_cache.hooks import register_cache_Dream           # noqa: E402

    resolved = Path(cache_module.__file__).resolve()
    if root not in resolved.parents:
        raise RuntimeError(
            f"'dllm_cache' resolved to {resolved}, not dLLM-Cache's - a name collision")
    return dLLMCache, register_cache_Dream


@register_model("Dream_dllmcache")
class DreamDLLMCache(HFLM):
    def __init__(
        self,
        pretrained: str = str(DEFAULT_MODEL),
        dllmcache_path: str = str(DEFAULT_DLLMCACHE),
        max_new_tokens: int = 0,
        diffusion_steps: int = 0,
        is_feature_cache: bool = True,
        is_cfg_cache: bool = False,
        prompt_interval_steps: int = 25,
        gen_interval_steps: int = 2,
        cfg_interval_steps: int = 1,
        transfer_ratio: float = 0.25,
        alg: str = "entropy",
        alg_temp: float = 0.0,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_seq_len: int = 2048,
        max_prompt_len: int = 0,
        dtype: str = "bfloat16",
        device: str = "cuda",
        chat_template: bool = False,
        show_speed: bool = True,
        **kwargs,
    ):
        import transformers

        dLLMCache, register_cache_Dream = _import_dllmcache_dream(Path(dllmcache_path))

        self._max_new_tokens = int(max_new_tokens)
        self._diffusion_steps = int(diffusion_steps)
        self._is_feature_cache = bool(is_feature_cache)
        self._is_cfg_cache = bool(is_cfg_cache)
        self._alg = str(alg)
        self._alg_temp = float(alg_temp)
        self._temperature = float(temperature)
        self._top_p = top_p if top_p is None else float(top_p)
        self._top_k = top_k if top_k is None else int(top_k)
        self._max_seq_len = int(max_seq_len)
        self._max_prompt_len = int(max_prompt_len) or self._max_seq_len
        self._chat_template = bool(chat_template)
        self._show_speed = bool(show_speed)
        self._cache = dLLMCache

        if self._max_new_tokens < 0 or self._diffusion_steps < 0:
            raise ValueError("max_new_tokens and diffusion_steps must be non-negative "
                             "(0 = take the budget from the task)")
        if self._max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if int(max_prompt_len) < 0:
            raise ValueError("max_prompt_len must be non-negative")

        # The hooks patch the checkpoint's own remote-code layers, so the model has
        # to be the checkpoint's class.
        config = transformers.AutoConfig.from_pretrained(
            str(pretrained), trust_remote_code=True)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(str(dtype), torch.bfloat16)
        model = transformers.AutoModel.from_pretrained(
            str(pretrained), config=config, torch_dtype=torch_dtype,
            trust_remote_code=True).to(str(device)).eval()

        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("add_bos_token", True)   # dLLM-Cache's Dream scripts set this
        super().__init__(pretrained=model, **kwargs)

        model_device = next(model.parameters()).device
        if model_device.type != "cuda":
            raise RuntimeError(
                f"model landed on {model_device}, not CUDA - CUDA init likely failed; "
                "rerun rather than evaluate on CPU")

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
            register_cache_Dream(model, _BLOCK_MODULE)
        else:
            dLLMCache.new_instance(
                prompt_interval_steps=1, gen_interval_steps=1, transfer_ratio=0,
                cfg_interval_steps=int(cfg_interval_steps) if is_cfg_cache else 1,
            )

        self._chat_prefix_ids, self._chat_suffix_ids = self._split_chat_template()

        cache_desc = (f"feature_cache prompt/{prompt_interval_steps} "
                      f"gen/{gen_interval_steps} transfer_ratio={transfer_ratio}"
                      if self._is_feature_cache else "no feature cache")
        print(f"[Dream_dllmcache] {cache_desc} alg={self._alg} "
              f"max_new_tokens={self._max_new_tokens or 'task'} "
              f"steps={self._diffusion_steps or 'max_new_tokens'} "
              f"max_seq_len={self._max_seq_len} max_prompt_len={self._max_prompt_len} "
              f"add_bos_token={self.add_bos_token} "
              f"chat_template={self._chat_template}", flush=True)

    # ------------------------------------------------------------------ prompts

    def _split_chat_template(self) -> Tuple[list, list]:
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
        new_tokens = self._max_new_tokens
        if new_tokens <= 0:
            new_tokens = int(raw_kwargs.get(
                "gen_length", raw_kwargs.get("max_gen_toks", self.max_gen_toks)))
        return new_tokens, self._diffusion_steps or new_tokens

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        from tqdm import tqdm

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
            print(f"[Dream_dllmcache] resume store: {len(done)} answers on disk", flush=True)

        results = []
        total_tokens, generated = 0, 0
        started = time.time()
        bar = tqdm(total=len(requests), disable=(disable_tqdm or self.rank != 0),
                   desc="dream dllmcache generate_until")
        for request in requests:
            context, raw_kwargs = request.args
            key = hashlib.md5(
                (context + repr(sorted(raw_kwargs.items()))).encode()).hexdigest()
            if key in done:
                results.append(done[key])
                bar.update(1)
                continue

            new_tokens, steps = self._schedule(raw_kwargs)
            prompt_limit = min(self._max_prompt_len, self._max_seq_len - new_tokens)
            if prompt_limit < 1:
                raise ValueError(
                    f"generation length {new_tokens} leaves no prompt space "
                    f"within max_seq_len {self._max_seq_len}")
            prompt_ids = self._encode_prompt(context, prompt_limit)

            # The cache keys its reuse off the step counter, so it is reset per item.
            self._cache().reset_cache(prompt_ids.shape[1])
            out = self.model.diffusion_generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=steps,
                temperature=self._temperature,
                top_p=self._top_p,
                top_k=self._top_k,
                alg=self._alg,
                alg_temp=self._alg_temp,
            )
            answer_ids = out.sequences[0][prompt_ids.shape[1]:]
            total_tokens += int((answer_ids != self.tokenizer.eos_token_id).sum())
            generated += 1

            text = self.tokenizer.decode(answer_ids.tolist()).split(
                self.tokenizer.eos_token)[0]
            for term in raw_kwargs.get("until") or []:
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

        if self._show_speed and generated:
            elapsed = time.time() - started
            print(f"[Dream_dllmcache] generated {generated} answers, {total_tokens} tokens "
                  f"in {elapsed:.1f}s ({total_tokens / elapsed:.2f} tok/s)", flush=True)
        return results

    # --------------------------------------------------------------- likelihood

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "dLLM-Cache does not implement Dream loglikelihood, and this harness's "
            "diffusion likelihood returns near-chance numbers for Dream")

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "rolling likelihood is not defined for the Dream diffusion evaluator")
