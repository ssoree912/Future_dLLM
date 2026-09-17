"""lm-eval model that runs Fast-dLLM v1's LLaDA decoding, registered as ``LLaDA_fastdllm``.

The harness, the task definitions and the local parquet data all come from
``Future_dLLM/eval``; only the model and the decoding loop come from
``Fast-dLLM/v1/llada`` - its ``model/modeling_llada.py`` (the one with the KV
cache and ``replace_position`` hooks) plus ``generate.py``'s three variants:

    use_cache=False                -> generate()                  (no cache)
    use_cache=True                 -> generate_with_prefix_cache()
    use_cache=True,dual_cache=True -> generate_with_dual_cache()

and, orthogonally, ``threshold`` / ``factor`` for parallel decoding.

    --model_args "pretrained=<model>,gen_length=256,steps=256,block_length=32,use_cache=True,dual_cache=True,threshold=0.9"

Chat template: Fast-dLLM applies it inside ``generate_until`` for any path
containing "instruct", as a single user turn wrapping the whole few-shot
prompt. That is reproduced here (``chat_template=True``), so the runner must
*not* pass lm-eval's ``--apply_chat_template`` for generative tasks. Set
``chat_template=False`` to fall back to the raw prompt.

Multiple choice uses Fast-dLLM's Monte-Carlo diffusion likelihood, which runs a
plain uncached forward - the cache and parallel-decoding knobs do not apply to
it.
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
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

REPO_ROOT = Path(__file__).resolve().parent.parent          # Future_dLLM/
WORKSPACE = REPO_ROOT.parent                                # /workspace/dllm
DEFAULT_MODEL = REPO_ROOT / "model" / "LLaDA-8B-Instruct"
DEFAULT_FASTDLLM = WORKSPACE / "Fast-dLLM" / "v1" / "llada"

_CHAT_SENTINEL = "FASTDLLM_SENTINEL_BODY"


def _import_fastdllm(root: Path):
    """Import Fast-dLLM's generate/model modules.

    ``generate.py`` does ``from model.modeling_llada import LLaDAModelLM``, an
    absolute import that only resolves with ``Fast-dLLM/v1/llada`` on sys.path.
    Future_dLLM's own root is on sys.path too and carries a ``model/`` symlink
    directory, but that one has no ``__init__.py``, so the real package here
    wins the import regardless of order.
    """
    root = Path(root).resolve()
    if not (root / "generate.py").is_file():
        raise FileNotFoundError(f"no generate.py under {root}")
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    import generate as fastdllm_generate               # noqa: E402
    from model.configuration_llada import LLaDAConfig  # noqa: E402
    from model.modeling_llada import LLaDAModelLM      # noqa: E402
    return fastdllm_generate, LLaDAConfig, LLaDAModelLM


def _round_up(value: int, multiple: int) -> int:
    remainder = value % multiple
    return value if remainder == 0 else value + multiple - remainder


# --------------------------------------------------------------------------
# Memory-lean confidence, injected into Fast-dLLM's generate module.
#
# Its get_transfer_index materialises softmax(logits.to(float64)) over the whole
# sequence and then keeps one probability per position. At LongBench's 4096-token
# context that intermediate is 4224 x 126464 x 8 B = 4.0 GB on top of the 16 GB of
# weights and the prefix KV cache, which does not fit on a 24 GB card. Chunking
# the same computation over positions - float64 throughout, log-sum-exp instead of
# a normalised softmax - gives the same number to float64 rounding at a few
# hundred MB. Everything downstream is transcribed from the originals.
# --------------------------------------------------------------------------

_CONFIDENCE_CHUNK = 128


def _chosen_token_probability(logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """softmax(logits.double())[x0], one chunk of positions at a time."""
    out = torch.empty(x0.shape, dtype=torch.float64, device=x0.device)
    for start in range(0, logits.shape[1], _CONFIDENCE_CHUNK):
        stop = start + _CONFIDENCE_CHUNK
        piece = logits[:, start:stop].to(torch.float64)
        chosen = piece.gather(-1, x0[:, start:stop].unsqueeze(-1)).squeeze(-1)
        out[:, start:stop] = torch.exp(chosen - torch.logsumexp(piece, dim=-1))
    return out


def _lean_get_transfer_index(logits, temperature, remasking, mask_index, x,
                             num_transfer_tokens, threshold=None):
    from generate import add_gumbel_noise

    x0 = torch.argmax(add_gumbel_noise(logits, temperature=temperature), dim=-1)
    if remasking == "low_confidence":
        x0_p = _chosen_token_probability(logits, x0)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        # always unmask the most confident position, even below the threshold
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        return x0, (transfer_index | force_mask) & mask_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = torch.clamp(
        num_transfer_tokens.to(dtype=torch.long, device=confidence.device), min=0)

    _, idx = torch.sort(confidence, dim=1, descending=True)
    b, length = confidence.shape
    cols = torch.arange(length, device=confidence.device).unsqueeze(0).expand(b, length)
    select_sorted = cols < num_transfer_tokens.unsqueeze(1).expand(b, length)
    transfer_int = torch.zeros(b, length, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    return x0, transfer_int.bool() & mask_index


def _lean_get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x,
                                     num_transfer_tokens, factor=1):
    import numpy as np
    from generate import add_gumbel_noise

    x0 = torch.argmax(add_gumbel_noise(logits, temperature=temperature), dim=-1)
    if remasking == "low_confidence":
        x0_p = _chosen_token_probability(logits, x0)
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        if int(num_transfer_tokens[j]) == 0:
            continue
        ns = list(range(1, num_transfer_tokens[j] + 1))
        threshs = [1 - factor / (n + 1) for n in ns]
        threshs[0] = -1                      # at least one token is transferred
        sorted_confidence = torch.sort(confidence[j][mask_index[j]], dim=-1,
                                       descending=True)[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break
        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1
        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


@register_model("LLaDA_fastdllm")
class LLaDAFastDLLM(HFLM):
    def __init__(
        self,
        pretrained: str = str(DEFAULT_MODEL),
        fastdllm_path: str = str(DEFAULT_FASTDLLM),
        gen_length: int = 256,
        steps: int = 0,
        block_length: int = 32,
        use_cache: bool = False,
        dual_cache: bool = False,
        threshold: Optional[float] = None,
        factor: Optional[float] = None,
        remasking: str = "low_confidence",
        mask_id: int = 126336,
        max_seq_len: int = 4096,
        max_prompt_len: int = 0,
        dtype: str = "bfloat16",
        device: str = "cuda",
        flash_attention: bool = True,
        chat_template: bool = True,
        lean_confidence: bool = True,
        show_speed: bool = True,
        mc_num: int = 128,
        mc_batch_size: int = 4,
        **kwargs,
    ):
        fastdllm_generate, LLaDAConfig, LLaDAModelLM = _import_fastdllm(Path(fastdllm_path))
        self._fast = fastdllm_generate
        self._lean_confidence = bool(lean_confidence)
        if self._lean_confidence:
            # generate() and friends resolve these as module globals, so replacing
            # them here is enough. See the note above the replacements.
            fastdllm_generate.get_transfer_index = _lean_get_transfer_index
            fastdllm_generate.get_transfer_index_dynamic = _lean_get_transfer_index_dynamic

        self._gen_length = int(gen_length)
        self._steps = int(steps)
        self._block_length = int(block_length)
        self._use_cache = bool(use_cache)
        self._dual_cache = bool(dual_cache)
        self._threshold = None if threshold is None else float(threshold)
        self._factor = None if factor is None else float(factor)
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
        if self._dual_cache and not self._use_cache:
            raise ValueError("dual_cache=True needs use_cache=True")
        if self._threshold is not None and self._factor is not None:
            raise ValueError("threshold and factor are two parallel schedules; pick one")
        if self._mc_num % self._mc_batch_size:
            raise ValueError("mc_num must be a multiple of mc_batch_size")

        # Fast-dLLM's own config class, not the checkpoint's remote-code one: its
        # model builds a ModelConfig out of fields (train_max_sequence_length and
        # friends) that only this class defines.
        config = LLaDAConfig.from_pretrained(str(pretrained))
        native_limit = int(getattr(config, "max_sequence_length", self._max_seq_len))
        if self._max_seq_len > native_limit:
            print(f"warning: max_seq_len={self._max_seq_len} exceeds the checkpoint's "
                  f"trained context {native_limit}", flush=True)
        config.max_sequence_length = self._max_seq_len
        # Fast-dLLM's eval script turns this on; the attention module falls back to
        # torch SDPA (non-causal, same maths) when flash_attn is not installed.
        config.flash_attention = bool(flash_attention)
        if flash_attention:
            try:
                import flash_attn  # noqa: F401
            except ModuleNotFoundError:
                print("note: flash_attn is not installed, using torch SDPA instead", flush=True)

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}.get(str(dtype), torch.bfloat16)
        model = LLaDAModelLM.from_pretrained(
            str(pretrained), config=config, torch_dtype=torch_dtype).to(str(device)).eval()

        # HFLM skips its own loading when handed a live model, but still needs the
        # path to find the tokenizer.
        kwargs.setdefault("tokenizer", str(pretrained))
        kwargs.setdefault("batch_size", 1)   # the generate() variants are written for one sequence
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

        self._chat_prefix_ids, self._chat_suffix_ids = self._split_chat_template()

        method = ("dual_cache" if self._dual_cache
                  else ("prefix_cache" if self._use_cache else "no_cache"))
        parallel = (f"threshold={self._threshold}" if self._threshold is not None
                    else (f"factor={self._factor}" if self._factor is not None
                          else "one token/step"))
        print(f"[LLaDA_fastdllm] {method} {parallel} gen_length={self._gen_length or 'task'} "
              f"steps={self._steps or 'gen_length'} block_length={self._block_length} "
              f"max_seq_len={self._max_seq_len} max_prompt_len={self._max_prompt_len} "
              f"chat_template={self._chat_template} "
              f"lean_confidence={self._lean_confidence}", flush=True)

    # ------------------------------------------------------------------ prompts

    def _split_chat_template(self) -> Tuple[list, list]:
        """Tokenise the instruct template once, around a sentinel body.

        Keeping the two halves lets an over-long prompt be truncated inside the
        user turn without eating the template's own header, which left-truncating
        the rendered string would do.
        """
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
        """Resolve (gen_length, steps) for one request.

        ``gen_length=0`` means "take the task's own budget". ``steps=0`` means one
        denoising step per token, which is what every Fast-dLLM variant is safe
        with: the quota schedule then moves one token per step, the threshold and
        factor schedules ignore the quota and stop early, and dual cache - whose
        refinement loop is bounded by ``steps // num_blocks`` rather than by a
        mask count - gets a full budget instead of a single pass per block.
        """
        gen_length = self._gen_length
        if gen_length <= 0:
            gen_length = int(raw_kwargs.get(
                "gen_length", raw_kwargs.get("max_gen_toks", self.max_gen_toks)))
        gen_length = _round_up(gen_length, self._block_length)   # generate() asserts divisibility
        steps = self._steps or gen_length
        num_blocks = gen_length // self._block_length
        if steps % num_blocks:
            steps = _round_up(steps, num_blocks)                 # generate() asserts this too
        return gen_length, steps

    def _call_generate(self, prompt_ids: torch.Tensor, gen_length: int, steps: int):
        common = dict(steps=steps, gen_length=gen_length, block_length=self._block_length,
                      temperature=0., remasking=self._remasking, mask_id=self._mask_id,
                      threshold=self._threshold, factor=self._factor)
        if self._use_cache and self._dual_cache:
            return self._fast.generate_with_dual_cache(self.model, prompt_ids, **common)
        if self._use_cache:
            return self._fast.generate_with_prefix_cache(self.model, prompt_ids, **common)
        return self._fast.generate(self.model, prompt_ids, **common)

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        from tqdm import tqdm

        # lm-eval's own --use_cache only writes once the whole batch returns, so a
        # crash part-way through loses everything. Each answer is appended and
        # fsynced as it is produced, and a restart replays what is on disk.
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
            print(f"[LLaDA_fastdllm] resume store: {len(done)} answers on disk", flush=True)

        results = []
        total_tokens, total_nfe, generated = 0, 0, 0
        started = time.time()
        bar = tqdm(total=len(requests), disable=(disable_tqdm or self.rank != 0),
                   desc="fastdllm generate_until")
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

            out, nfe = self._call_generate(prompt_ids, gen_length, steps)

            # Fast-dLLM decodes the answer with the special tokens still in, cuts it
            # at the task's stop strings, then re-encodes to drop them.
            text = self.tokenizer.decode(out[0, prompt_ids.shape[1]:],
                                         skip_special_tokens=False)
            for term in raw_kwargs.get("until") or []:
                if term:
                    text = text.split(term)[0]
            answer_ids = self.tokenizer(text)["input_ids"]
            total_tokens += sum(1 for t in answer_ids if t != self.tokenizer.pad_token_id)
            total_nfe += nfe
            generated += 1
            text = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

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
            print(f"[LLaDA_fastdllm] generated {generated} answers, {total_tokens} tokens "
                  f"in {elapsed:.1f}s ({total_tokens / elapsed:.2f} tok/s), "
                  f"NFE {total_nfe} (avg {total_nfe / generated:.1f})", flush=True)
        return results

    # --------------------------------------------------------------- likelihood

    def _forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor):
        """Fast-dLLM's forward diffusion for the Monte-Carlo likelihood."""
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
        """Fast-dLLM's Monte-Carlo diffusion likelihood (cfg off, greedy check off).

        Uncached full forwards - the cache and parallel-decoding knobs above have
        no effect here.
        """
        from tqdm import tqdm

        out = []
        for request in tqdm(requests, disable=self.rank != 0,
                            desc="fastdllm diffusion loglikelihood"):
            prefix, target = self._encode_pair(*request.args)
            out.append((self._get_loglikelihood(prefix, target), False))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "rolling likelihood is not defined for the LLaDA diffusion evaluator")
