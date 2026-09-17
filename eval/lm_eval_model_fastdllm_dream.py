"""lm-eval model that runs Fast-dLLM v1's Dream decoding, registered as ``Dream_fastdllm``.

The Dream counterpart of ``lm_eval_model_fastdllm.py``: the harness, the tasks and
the data come from ``Future_dLLM/eval``; the model and the decoding loop come from
``Fast-dLLM/v1/dream``. Dream switches cache modes by swapping which module's
``DreamGenerationMixin`` is bound to the model:

    use_cache=False  -> model/generation_utils.py         (no cache; no blocks)
    use_cache=True   -> model/generation_utils_block.py   (block cache, dual_cache)

and decodes either by entropy over a fixed step budget (``alg=entropy``) or by
confidence threshold (``alg=confidence_threshold,threshold=0.9``), which is how
Fast-dLLM's own ``dream/eval_gsm8k.sh`` spells its five variants.

    --model_args "pretrained=<model>,max_new_tokens=256,diffusion_steps=32,alg=confidence_threshold,threshold=0.9,use_cache=True,dual_cache=True"

Both Fast-dLLM Dream scripts pass ``add_bos_token=true``, so it defaults to True
here - unlike the LLaDA wrappers, where the repo's own harness leaves it off.

Likelihood is not implemented: the Dream branch of this repo records that
diffusion loglikelihood through lm-eval returns near-chance numbers for Dream
(piqa 0.45 where direct scoring gives 0.825), so the multiple-choice tasks are
excluded rather than reported.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

REPO_ROOT = Path(__file__).resolve().parent.parent          # Future_dLLM/
WORKSPACE = REPO_ROOT.parent                                # /workspace/dllm
DEFAULT_MODEL = REPO_ROOT / "model" / "Dream-v0-Instruct-7B"
DEFAULT_FASTDLLM = WORKSPACE / "Fast-dLLM" / "v1" / "dream"

_CHAT_SENTINEL = "FASTDLLM_SENTINEL_BODY"


def _import_fastdllm_dream(root: Path):
    """Import Fast-dLLM's Dream model and both generation mixins.

    ``Fast-dLLM/v1/dream`` and ``Fast-dLLM/v1/llada`` each carry a regular package
    called ``model``. Whichever directory is on sys.path first wins for the whole
    process, so this module must never share a process with the LLaDA wrapper -
    hence its own entry point, ``eval/run_fastdllm_dream.py``. The resolved file is
    checked so a collision fails loudly instead of loading the wrong architecture.
    """
    root = Path(root).resolve()
    if not (root / "model" / "modeling_dream.py").is_file():
        raise FileNotFoundError(f"no model/modeling_dream.py under {root}")
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))

    from model import modeling_dream                                    # noqa: E402
    from model.configuration_dream import DreamConfig                   # noqa: E402
    from model.generation_utils import DreamGenerationMixin as Plain    # noqa: E402
    from model.generation_utils_block import DreamGenerationMixin as Block  # noqa: E402

    resolved = Path(modeling_dream.__file__).resolve()
    if root not in resolved.parents:
        raise RuntimeError(
            f"'model' resolved to {resolved}, not Fast-dLLM's dream package - a "
            "name collision, most likely with the llada wrapper in the same process")
    return modeling_dream.DreamModel, DreamConfig, Plain, Block


@register_model("Dream_fastdllm")
class DreamFastDLLM(HFLM):
    def __init__(
        self,
        pretrained: str = str(DEFAULT_MODEL),
        fastdllm_path: str = str(DEFAULT_FASTDLLM),
        max_new_tokens: int = 0,
        diffusion_steps: int = 0,
        block_length: int = 32,
        use_cache: bool = False,
        dual_cache: bool = False,
        threshold: float = 0.9,
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
        DreamModel, DreamConfig, Plain, Block = _import_fastdllm_dream(Path(fastdllm_path))

        self._max_new_tokens = int(max_new_tokens)
        self._diffusion_steps = int(diffusion_steps)
        self._block_length = int(block_length)
        self._use_cache = bool(use_cache)
        self._dual_cache = bool(dual_cache)
        self._threshold = float(threshold)
        self._alg = str(alg)
        self._alg_temp = float(alg_temp)
        self._temperature = float(temperature)
        self._top_p = top_p if top_p is None else float(top_p)
        self._top_k = top_k if top_k is None else int(top_k)
        self._max_seq_len = int(max_seq_len)
        self._max_prompt_len = int(max_prompt_len) or self._max_seq_len
        self._chat_template = bool(chat_template)
        self._show_speed = bool(show_speed)

        if self._block_length < 1:
            raise ValueError("block_length must be positive")
        if self._max_new_tokens < 0 or self._diffusion_steps < 0:
            raise ValueError("max_new_tokens and diffusion_steps must be non-negative "
                             "(0 = take the budget from the task)")
        if self._max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")
        if int(max_prompt_len) < 0:
            raise ValueError("max_prompt_len must be non-negative")
        if self._dual_cache and not self._use_cache:
            raise ValueError("dual_cache=True needs use_cache=True")

        config = DreamConfig.from_pretrained(str(pretrained))
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(str(dtype), torch.bfloat16)
        model = DreamModel.from_pretrained(
            str(pretrained), config=config, torch_dtype=torch_dtype,
            trust_remote_code=True).to(str(device)).eval()

        # Fast-dLLM rebinds these two per call; the mode is fixed for a whole run
        # here, so bind once and keep the model's behaviour stable.
        mixin = Block if self._use_cache else Plain
        model.diffusion_generate = types.MethodType(mixin.diffusion_generate, model)
        model._sample = types.MethodType(mixin._sample, model)

        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("add_bos_token", True)   # both Fast-dLLM Dream scripts set this
        super().__init__(pretrained=model, **kwargs)

        model_device = next(model.parameters()).device
        if model_device.type != "cuda":
            raise RuntimeError(
                f"model landed on {model_device}, not CUDA - CUDA init likely failed; "
                "rerun rather than evaluate on CPU")

        self._chat_prefix_ids, self._chat_suffix_ids = self._split_chat_template()

        mode = "dual_cache" if self._dual_cache else ("block_cache" if self._use_cache else "no_cache")
        print(f"[Dream_fastdllm] {mode} alg={self._alg} threshold={self._threshold} "
              f"max_new_tokens={self._max_new_tokens or 'task'} "
              f"steps={self._diffusion_steps or 'max_new_tokens'} "
              f"block_length={self._block_length} max_seq_len={self._max_seq_len} "
              f"max_prompt_len={self._max_prompt_len} "
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
        """Resolve (max_new_tokens, diffusion_steps).

        With ``alg=entropy`` the step budget is the whole generation length, one
        token per step. The threshold schedule fills several positions per step,
        which is why Fast-dLLM's own scripts drop it to length/block_length there.
        """
        new_tokens = self._max_new_tokens
        if new_tokens <= 0:
            new_tokens = int(raw_kwargs.get(
                "gen_length", raw_kwargs.get("max_gen_toks", self.max_gen_toks)))
        steps = self._diffusion_steps or new_tokens
        return new_tokens, steps

    def _call_generate(self, prompt_ids: torch.Tensor, new_tokens: int, steps: int):
        call = dict(
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
            threshold=self._threshold,
        )
        if self._use_cache:
            # only the block module reads these two
            call["block_length"] = self._block_length
            call["dual_cache"] = self._dual_cache
        return self.model.diffusion_generate(prompt_ids, **call)

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        from tqdm import tqdm

        store_path = os.environ.get("FASTDLLM_RESUME", "")
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
            print(f"[Dream_fastdllm] resume store: {len(done)} answers on disk", flush=True)

        results = []
        total_tokens, generated = 0, 0
        started = time.time()
        bar = tqdm(total=len(requests), disable=(disable_tqdm or self.rank != 0),
                   desc="dream fastdllm generate_until")
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

            out = self._call_generate(prompt_ids, new_tokens, steps)
            answer_ids = out.sequences[0][prompt_ids.shape[1]:]
            total_tokens += int((answer_ids != self.tokenizer.eos_token_id).sum())
            generated += 1

            # Fast-dLLM keeps the special tokens in, cuts at the first eos, then
            # applies the task's stop strings.
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
            print(f"[Dream_fastdllm] generated {generated} answers, {total_tokens} tokens "
                  f"in {elapsed:.1f}s ({total_tokens / elapsed:.2f} tok/s)", flush=True)
        return results

    # --------------------------------------------------------------- likelihood

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "Dream's diffusion loglikelihood returns near-chance numbers through this "
            "harness, so the multiple-choice tasks are excluded rather than reported")

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "rolling likelihood is not defined for the Dream diffusion evaluator")
