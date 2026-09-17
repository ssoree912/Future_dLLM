"""OpenCompass wrappers for the two published cache methods, on both families.

The rows this file adds sit beside ``model.py`` (Ours / full cache / Sparse-dLLM)
in the same OpenCompass configs, so the multiple-choice suite is scored the same
generative way for every row:

    LLaDAFastDLLMOC     Fast-dLLM v1, LLaDA    (Fast-dLLM/v1/llada)
    DreamFastDLLMOC     Fast-dLLM v1, Dream    (Fast-dLLM/v1/dream)
    LLaDADLLMCacheOC    dLLM-Cache, LLaDA      (dLLM-cache)
    DreamDLLMCacheOC    dLLM-Cache, Dream      (dLLM-cache)

Each keeps its family's prompt and decode conventions exactly as
``llada_model.py`` / ``model.py`` establish them - raw prompt and stop words for
LLaDA, chat template and EOS-cut for Dream - and changes only what the method
itself changes: which generate function runs and how the cache is set up.
Decoding hyper-parameters stay at each *method's* own published values, which is
how the lm-eval rows for these two were produced as well.

One hard constraint: ``Fast-dLLM/v1/llada`` and ``Fast-dLLM/v1/dream`` each ship
a top-level package called ``model``, and only one can win an import in a
process. OpenCompass's ``--debug`` runs tasks in-process, so a single config
must not mix ``LLaDAFastDLLMOC`` with ``DreamFastDLLMOC``. The per-family configs
already keep them apart.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from opencompass.models.base import BaseModel

from .model import _convert_base_messages

WORKSPACE = Path(__file__).resolve().parent.parent.parent
FASTDLLM_LLADA = WORKSPACE / "Fast-dLLM" / "v1" / "llada"
FASTDLLM_DREAM = WORKSPACE / "Fast-dLLM" / "v1" / "dream"
DLLMCACHE = WORKSPACE / "dLLM-cache"


def _sample_seed(seed: int, key: str) -> int:
    """Identical to ``future_dllm.dream_decoding.sample_seed``, inlined.

    Importing it would mean putting another repo root on sys.path, and both
    checkouts here ship a package called ``future_dllm`` - the one that wins is
    whichever path went in first. Three lines are cheaper than that hazard, and
    copying them keeps our rows on exactly their per-item seeds.
    """
    import hashlib
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _seeded(seed: int, key: str) -> None:
    item_seed = _sample_seed(seed, key)
    random.seed(item_seed)
    np.random.seed(item_seed % (2**32))
    torch.manual_seed(item_seed)
    torch.cuda.manual_seed_all(item_seed)


def _load_by_path(name: str, path: Path):
    """Import one file without touching sys.path - see _sample_seed for why."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_from(root: Path, marker: str, *names):
    """Put ``root`` first on sys.path and import, checking what we actually got.

    ``model`` and ``utils`` are generic enough that a silent collision would
    load a different architecture rather than fail, so the module that ``marker``
    names is required to resolve under ``root``.
    """
    root = root.resolve()
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    import importlib
    probe = importlib.import_module(marker)
    resolved = Path(probe.__file__).resolve()
    if root not in resolved.parents:
        raise RuntimeError(f"{marker!r} resolved to {resolved}, not under {root} - "
                           "two repos' packages collided in one process")
    return [importlib.import_module(n) for n in names]


# ---------------------------------------------------------------- LLaDA family


class _LLaDAFamilyOC(BaseModel):
    """Prompt and decode handling shared by the LLaDA rows.

    Copied from ``llada_model.LLaDAFutureOC``: no chat template, generation
    budget rounded up to a whole block, ``skip_special_tokens=True``, stop words
    applied. Only ``_build`` and ``_generate_one`` differ per method.
    """

    def __init__(self, path: str, max_seq_len: int, block_length: int,
                 meta_template: Optional[dict], seed: int):
        # Only the LLaDA variable, never FUTURE_DLLM_MODEL: that one defaults to
        # a Dream checkpoint, and a Dream path must never reach a LLaDA run.
        path = path or os.environ.get("FUTURE_DLLM_LLADA_MODEL", "")
        if not path:
            raise ValueError("set FUTURE_DLLM_LLADA_MODEL or pass path=")
        super().__init__(path=path, max_seq_len=max_seq_len,
                         meta_template=meta_template)
        repo = Path(__file__).resolve().parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from transformers import AutoTokenizer

        self._path = path
        self._block_length = int(block_length)
        self._seed = int(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tokenizer.truncation_side = "left"   # OpenCompass's own default

    def _encode(self, text: str, max_length: int):
        return self.tokenizer.batch_encode_plus(
            [text], return_tensors="pt", padding=True, truncation=True,
            add_special_tokens=True, max_length=max_length)

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        text = _convert_base_messages([prompt])[0]
        return len(self.tokenizer(text,
                                  add_special_tokens=add_special_tokens)["input_ids"])

    def _gen_length(self, max_out_len: int) -> int:
        gen_length = int(max_out_len)
        if gen_length % self._block_length:
            gen_length = (gen_length // self._block_length + 1) * self._block_length
        return gen_length

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int,
                 stopping_criteria: List[str] = []) -> List[str]:
        gen_length = self._gen_length(max_out_len)
        outputs = []
        for text in _convert_base_messages(inputs):
            _seeded(self._seed, text)
            ids = self._encode(text, self.max_seq_len)["input_ids"].to(self.model.device)
            answer_ids = self._generate_one(ids, gen_length)
            text_out = self.tokenizer.decode(answer_ids.tolist(),
                                             skip_special_tokens=True)
            for stop in stopping_criteria:
                text_out = text_out.split(stop)[0]
            outputs.append(text_out)
        return outputs


class LLaDAFastDLLMOC(_LLaDAFamilyOC):
    """Fast-dLLM v1 on LLaDA: block KV cache, dual cache, parallel decoding."""

    def __init__(self, path: str = "", max_seq_len: int = 2048,
                 block_length: int = 32, use_cache: bool = True,
                 dual_cache: bool = True, threshold: float = 0.9,
                 llada_steps: int = 0, llada_temperature: float = 0.0,
                 llada_remasking: str = "low_confidence", llada_seed: int = 2025,
                 meta_template: Optional[dict] = None):
        super().__init__(path, max_seq_len, block_length, meta_template, llada_seed)
        generate_mod, modeling, config_mod = _import_from(
            FASTDLLM_LLADA, "model.modeling_llada",
            "generate", "model.modeling_llada", "model.configuration_llada")
        self._fast = generate_mod
        self._use_cache = bool(use_cache)
        self._dual_cache = bool(dual_cache)
        self._threshold = float(threshold)
        self._steps = int(llada_steps)
        self._temperature = float(llada_temperature)
        self._remasking = str(llada_remasking)

        # Their get_transfer_index takes a float64 softmax over the whole
        # sequence; at 2048 that is a 2 GB intermediate on top of 16 GB of
        # weights. The chunked replacement is bit-identical (verified on GPU at
        # L=4224: max probability difference 1.55e-15, zero differing decisions).
        lean = _load_by_path(
            "_fastdllm_lean",
            Path("/workspace/dllm/Future_dLLM/eval/lm_eval_model_fastdllm.py"))
        generate_mod.get_transfer_index = lean._lean_get_transfer_index
        generate_mod.get_transfer_index_dynamic = lean._lean_get_transfer_index_dynamic

        config = config_mod.LLaDAConfig.from_pretrained(self._path)
        config.max_sequence_length = max_seq_len
        config.flash_attention = True          # falls back to SDPA without flash_attn
        self.model = modeling.LLaDAModelLM.from_pretrained(
            self._path, config=config, torch_dtype=torch.bfloat16).to("cuda").eval()
        self._mask_id = 126336
        print(f"[LLaDAFastDLLMOC] use_cache={self._use_cache} "
              f"dual_cache={self._dual_cache} threshold={self._threshold} "
              f"block_length={self._block_length} max_seq_len={max_seq_len} "
              f"seed={self._seed}", flush=True)

    def _generate_one(self, ids, gen_length):
        # One denoising step per token unless the config says otherwise: the
        # threshold schedule ignores the quota and stops a block early anyway,
        # and dual cache needs the budget because its refinement loop is bounded
        # by steps/num_blocks rather than by the remaining mask count.
        steps = self._steps or gen_length
        common = dict(steps=steps, gen_length=gen_length,
                      block_length=self._block_length, temperature=self._temperature,
                      remasking=self._remasking, mask_id=self._mask_id,
                      threshold=self._threshold, factor=None)
        if self._use_cache and self._dual_cache:
            out, _ = self._fast.generate_with_dual_cache(self.model, ids, **common)
        elif self._use_cache:
            out, _ = self._fast.generate_with_prefix_cache(self.model, ids, **common)
        else:
            out, _ = self._fast.generate(self.model, ids, **common)
        return out[0, ids.shape[1]:]


class LLaDADLLMCacheOC(_LLaDAFamilyOC):
    """dLLM-Cache on LLaDA: adaptive feature cache over the denoising steps."""

    def __init__(self, path: str = "", max_seq_len: int = 2048,
                 block_length: int = 32, is_feature_cache: bool = True,
                 prompt_interval_steps: int = 50, gen_interval_steps: int = 7,
                 transfer_ratio: float = 0.25, llada_steps: int = 0,
                 llada_temperature: float = 0.0, llada_cfg_scale: float = 0.0,
                 llada_remasking: str = "low_confidence", llada_seed: int = 2025,
                 meta_template: Optional[dict] = None):
        super().__init__(path, max_seq_len, block_length, meta_template, llada_seed)
        gen_mod, cache_mod, hooks_mod = _import_from(
            DLLMCACHE, "utils.generate_function",
            "utils.generate_function", "dllm_cache.cache", "dllm_cache.hooks")
        self._generate = gen_mod.generate
        self._cache = cache_mod.dLLMCache
        self._steps = int(llada_steps)
        self._temperature = float(llada_temperature)
        self._cfg_scale = float(llada_cfg_scale)
        self._remasking = str(llada_remasking)
        self._mask_id = 126336

        import transformers
        config = transformers.AutoConfig.from_pretrained(
            self._path, trust_remote_code=True)
        config.max_sequence_length = max_seq_len
        self.model = transformers.AutoModel.from_pretrained(
            self._path, config=config, torch_dtype=torch.bfloat16,
            trust_remote_code=True).to("cuda").eval()

        if is_feature_cache:
            self._cache.new_instance(
                prompt_interval_steps=int(prompt_interval_steps),
                gen_interval_steps=int(gen_interval_steps),
                transfer_ratio=float(transfer_ratio), cfg_interval_steps=1)
            hooks_mod.register_cache_LLaDA(self.model, "model.transformer.blocks")
        else:
            self._cache.new_instance(prompt_interval_steps=1, gen_interval_steps=1,
                                     transfer_ratio=0, cfg_interval_steps=1)
        print(f"[LLaDADLLMCacheOC] feature_cache={is_feature_cache} "
              f"prompt/{prompt_interval_steps} gen/{gen_interval_steps} "
              f"transfer_ratio={transfer_ratio} block_length={self._block_length} "
              f"max_seq_len={max_seq_len} seed={self._seed}", flush=True)

    def _generate_one(self, ids, gen_length):
        steps = self._steps or gen_length
        self._cache().reset_cache(ids.shape[1])
        out = self._generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), model=self.model,
            steps=steps, gen_length=gen_length, block_length=self._block_length,
            temperature=self._temperature, cfg_scale=self._cfg_scale,
            remasking=self._remasking, mask_id=self._mask_id)
        return out[0]          # their generate returns the answer span only


# ---------------------------------------------------------------- Dream family


class _DreamFamilyOC(BaseModel):
    """Prompt and decode handling shared by the Dream rows.

    Copied from ``model.DreamFutureOC``: chat template as a single user turn,
    specials kept and the answer cut at the first EOS, stop words dropped.
    """

    def __init__(self, path: str, max_seq_len: int,
                 meta_template: Optional[dict], seed: int):
        path = path or os.environ.get("FUTURE_DLLM_MODEL", "")
        if not path:
            raise ValueError("set FUTURE_DLLM_MODEL or pass path=")
        super().__init__(path=path, max_seq_len=max_seq_len,
                         meta_template=meta_template)
        repo = Path(__file__).resolve().parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from transformers import AutoTokenizer

        self._path = path
        self._seed = int(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tokenizer.truncation_side = "left"

    def _encode(self, text: str, max_length: int):
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False)
        return self.tokenizer.batch_encode_plus(
            [chat], return_tensors="pt", padding=True, truncation=True,
            add_special_tokens=False, max_length=max_length)

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        text = _convert_base_messages([prompt])[0]
        return len(self.tokenizer(text,
                                  add_special_tokens=add_special_tokens)["input_ids"])

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        # No stopping_criteria parameter: GenInferencer only passes stop words to
        # a generate() that names one, and their Dream path drops them.
        outputs = []
        for text in _convert_base_messages(inputs):
            _seeded(self._seed, text)
            ids = self._encode(text, self.max_seq_len)["input_ids"].to(self.model.device)
            answer_ids = self._generate_one(ids, int(max_out_len))
            decoded = self.tokenizer.decode(answer_ids.tolist())
            outputs.append(decoded.split(self.tokenizer.eos_token)[0])
        return outputs


class DreamFastDLLMOC(_DreamFamilyOC):
    """Fast-dLLM v1 on Dream: block cache plus their confidence_threshold decode."""

    def __init__(self, path: str = "", max_seq_len: int = 2048,
                 block_length: int = 32, use_cache: bool = True,
                 dual_cache: bool = True, threshold: float = 0.9,
                 dream_alg: str = "confidence_threshold", dream_alg_temp: float = 0.0,
                 dream_steps: int = 0, dream_temperature: float = 0.0,
                 dream_top_p: Optional[float] = None, dream_seed: int = 2025,
                 meta_template: Optional[dict] = None):
        super().__init__(path, max_seq_len, meta_template, dream_seed)
        import types
        modeling, config_mod, plain, block = _import_from(
            FASTDLLM_DREAM, "model.modeling_dream",
            "model.modeling_dream", "model.configuration_dream",
            "model.generation_utils", "model.generation_utils_block")
        self._block_length = int(block_length)
        self._use_cache = bool(use_cache)
        self._dual_cache = bool(dual_cache)
        self._threshold = float(threshold)
        self._alg = str(dream_alg)
        self._alg_temp = float(dream_alg_temp)
        self._steps = int(dream_steps)
        self._temperature = float(dream_temperature)
        self._top_p = dream_top_p if dream_top_p is None else float(dream_top_p)

        config = config_mod.DreamConfig.from_pretrained(self._path)
        self.model = modeling.DreamModel.from_pretrained(
            self._path, config=config, torch_dtype=torch.bfloat16).to("cuda").eval()
        mixin = block.DreamGenerationMixin if self._use_cache else plain.DreamGenerationMixin
        self.model.diffusion_generate = types.MethodType(mixin.diffusion_generate, self.model)
        self.model._sample = types.MethodType(mixin._sample, self.model)
        print(f"[DreamFastDLLMOC] use_cache={self._use_cache} "
              f"dual_cache={self._dual_cache} alg={self._alg} "
              f"threshold={self._threshold} temperature={self._temperature} "
              f"top_p={self._top_p} max_seq_len={max_seq_len} seed={self._seed}",
              flush=True)

    def _generate_one(self, ids, max_out_len):
        # entropy walks one token per step; the threshold schedule fills several
        # at once, which is why Fast-dLLM's own scripts drop the budget there.
        steps = self._steps or (max_out_len // self._block_length
                                if self._alg == "confidence_threshold" else max_out_len)
        call = dict(attention_mask=torch.ones_like(ids), max_new_tokens=max_out_len,
                    output_history=False, return_dict_in_generate=True, steps=steps,
                    temperature=self._temperature, top_p=self._top_p, top_k=None,
                    alg=self._alg, alg_temp=self._alg_temp, threshold=self._threshold)
        if self._use_cache:
            call["block_length"] = self._block_length   # only the block module reads these
            call["dual_cache"] = self._dual_cache
        out = self.model.diffusion_generate(ids, **call)
        return out.sequences[0][ids.shape[1]:]


class DreamDLLMCacheOC(_DreamFamilyOC):
    """dLLM-Cache on Dream: feature cache over the checkpoint's own decode."""

    def __init__(self, path: str = "", max_seq_len: int = 2048,
                 is_feature_cache: bool = True, prompt_interval_steps: int = 25,
                 gen_interval_steps: int = 2, transfer_ratio: float = 0.25,
                 dream_alg: str = "entropy", dream_alg_temp: float = 0.0,
                 dream_steps: int = 256, dream_temperature: float = 0.2,
                 dream_top_p: float = 0.95, dream_seed: int = 2025,
                 meta_template: Optional[dict] = None):
        super().__init__(path, max_seq_len, meta_template, dream_seed)
        cache_mod, hooks_mod = _import_from(
            DLLMCACHE, "dllm_cache.cache", "dllm_cache.cache", "dllm_cache.hooks")
        self._cache = cache_mod.dLLMCache
        self._alg = str(dream_alg)
        self._alg_temp = float(dream_alg_temp)
        self._steps = int(dream_steps)
        self._temperature = float(dream_temperature)
        self._top_p = float(dream_top_p)

        import transformers
        config = transformers.AutoConfig.from_pretrained(
            self._path, trust_remote_code=True)
        self.model = transformers.AutoModel.from_pretrained(
            self._path, config=config, torch_dtype=torch.bfloat16,
            trust_remote_code=True).to("cuda").eval()

        if is_feature_cache:
            self._cache.new_instance(
                prompt_interval_steps=int(prompt_interval_steps),
                gen_interval_steps=int(gen_interval_steps),
                transfer_ratio=float(transfer_ratio), cfg_interval_steps=1)
            hooks_mod.register_cache_Dream(self.model, "model.layers")
        else:
            self._cache.new_instance(prompt_interval_steps=1, gen_interval_steps=1,
                                     transfer_ratio=0, cfg_interval_steps=1)
        print(f"[DreamDLLMCacheOC] feature_cache={is_feature_cache} "
              f"prompt/{prompt_interval_steps} gen/{gen_interval_steps} "
              f"transfer_ratio={transfer_ratio} alg={self._alg} "
              f"temperature={self._temperature} top_p={self._top_p} "
              f"max_seq_len={max_seq_len} seed={self._seed}", flush=True)

    def _generate_one(self, ids, max_out_len):
        steps = min(self._steps, max_out_len) if self._steps else max_out_len
        self._cache().reset_cache(ids.shape[1])
        out = self.model.diffusion_generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_out_len,
            output_history=False, return_dict_in_generate=True, steps=steps,
            temperature=self._temperature, top_p=self._top_p, top_k=None,
            alg=self._alg, alg_temp=self._alg_temp)
        return out.sequences[0][ids.shape[1]:]
