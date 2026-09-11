"""OpenCompass model wrapper around this repo's Dream decoding.

Why OpenCompass at all: multiple choice. Dream and LLaDA both use bidirectional
attention, so no autoregressive loglikelihood exists for them -- feed the whole
sequence and position r reads token r+1 directly, which is measurable (NLL 1.94
with the answer visible against 5.50 with it masked). lm-eval's
``output_type: multiple_choice`` therefore cannot be used as written, and the
Monte-Carlo diffusion estimator that would replace it scores a different
quantity from the one Sparse-dLLM reports. They score MMLU / ARC-C / PIQA /
GPQA generatively in OpenCompass, so matching them means running OpenCompass's
own templates, postprocessors and evaluators rather than reimplementing them.

Self-contained on purpose. The only third-party import is ``opencompass``, a
pip dependency like ``lm_eval``. Nothing is read from a Sparse-dLLM checkout and
nothing of ours is written into the OpenCompass tree, which stays vanilla
(``git status`` clean) so the version is exactly what the pin says.

Decoding comes from ``future_dllm.dream_generate``, which
``scripts/check_dream_decoding.py`` verifies against Sparse-dLLM's
``diffusion_generate`` down to tokens, per-step history and RNG consumption.
That verification is what lets this wrapper stand in for theirs.

    eviction_method="sparse"    their attention score   -> baseline row
    eviction_method="student"   the trained scorer      -> our row

Prompt handling mirrors ``Sparse_dLLM_DreamCausalLMInstruct.generate`` step for
step, because on an instruct model scored generatively the template decides the
score as much as the cache policy does. One deliberate deviation, applied to
every row so the comparison stays internally valid: per-prompt seeding instead
of one global seed, so a rerun or a resumed shard reproduces the same answer.

``truncation_side="left"`` is not a deviation. It is what OpenCompass itself
sets in ``HuggingFaceBaseModel._load_tokenizer`` (``DEFAULT_TOKENIZER_KWARGS``,
alongside ``padding_side='left'``), and Sparse-dLLM's wrapper subclasses that
model and calls the inherited loader, so it gets the same. It is set explicitly
here only because this class derives from ``BaseModel``, which loads no
tokenizer of its own. The ``truncation_side='right'`` elsewhere in that file
belongs to ``get_ppl_tokenwise``, a scoring path this suite does not use.

One difference that is not a deviation: they narrow ``steps`` by mutating
``self.diffusion_config`` in place, which leaks the smaller value into later
datasets; ``DreamDecoding.steps_for_length`` recomputes it per call. At the
official ``max_out_len=256`` the two agree.
"""

from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np
import torch
from opencompass.models.base import BaseModel


def _convert_base_messages(inputs):
    """Flatten OpenCompass prompts to plain strings.

    Verbatim from Sparse-dLLM's ``dream_wrapper_instruct.py``: a PromptList is
    joined on ``prompt`` with the roles dropped, and the whole thing is handed
    to the chat template as a single user turn.
    """
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            outputs.append(''.join(item['prompt'] for item in _input))
    return outputs


class DreamFutureOC(BaseModel):
    """Dream with future-attention cache eviction, driven by OpenCompass."""

    def __init__(
        self,
        path: str = "",
        max_seq_len: int = 2048,
        block_length: int = 32,
        keep_ratio: float = 1.0,
        eviction_method: str = "student",
        student_path: str = "",
        dream_alg: str = "entropy",
        dream_temperature: float = 0.2,
        dream_top_p: float = 0.95,
        dream_steps: int = 256,
        dream_seed: int = 2025,
        meta_template: Optional[dict] = None,
    ):
        # The config that builds this is parsed by mmengine in lazy-import
        # mode, which cannot call os.environ.get, so the paths arrive empty and
        # are resolved here instead -- same variables scripts/run_eval.sh uses.
        path = path or os.environ.get("FUTURE_DLLM_MODEL", "")
        if not path:
            raise ValueError("set FUTURE_DLLM_MODEL or pass path= in the config")
        if eviction_method == "student":
            student_path = student_path or os.environ.get("FUTURE_DLLM_STUDENT", "")

        # No **kwargs: OpenCompass strips only run_cfg / max_out_len /
        # batch_size / abbr / summarizer_abbr / pred_postprocessor /
        # min_out_len before building, so anything else arriving here is a real
        # config key and a typo should raise instead of being swallowed.
        super().__init__(path=path, max_seq_len=max_seq_len,
                         meta_template=meta_template)
        import sys
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from transformers import AutoTokenizer

        from future_dllm import load_model, load_prompt_utility_student
        from future_dllm.dream_decoding import DreamDecoding, sample_seed

        self._block_length = int(block_length)
        self._keep_ratio = float(keep_ratio)
        self._eviction_method = str(eviction_method)
        self._seed = int(dream_seed)
        self._sample_seed = sample_seed
        self._decoding = DreamDecoding(alg=dream_alg, temperature=dream_temperature,
                                       top_p=dream_top_p, steps=dream_steps)

        self.model, self._backend = load_model(
            path, max_seq_len=max_seq_len, block_length=self._block_length,
            keep_ratio=self._keep_ratio)
        self.model.eval()
        # load_model injects these onto the config before from_pretrained, and
        # dream_generate reads them back off model.config rather than from its
        # own arguments. Reading them back here turns "the config said 0.5"
        # into "the model is running at 0.5", which is the claim the whole
        # comparison rests on and is otherwise invisible in the logs.
        if (self.model.config.keep_ratio != self._keep_ratio
                or self.model.config.block_len != self._block_length):
            raise RuntimeError(
                f"eviction settings did not reach the model: config says "
                f"keep_ratio={self.model.config.keep_ratio} "
                f"block_len={self.model.config.block_len}, expected "
                f"{self._keep_ratio} / {self._block_length}")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tokenizer.truncation_side = "left"   # OpenCompass's own default

        self._scorer = None
        if self._eviction_method == "student":
            if not student_path:
                raise ValueError("eviction_method='student' requires a scorer: "
                                 "set FUTURE_DLLM_STUDENT or pass student_path=")
            self._scorer = load_prompt_utility_student(
                student_path, next(self.model.parameters()).device)
        elif self._eviction_method != "sparse":
            raise ValueError(f"unknown eviction_method {self._eviction_method!r}; "
                             "expected 'sparse' or 'student'")

        if self._keep_ratio >= 1.0:
            eviction = "none (keep_ratio=1.0)"
        elif self._eviction_method == "sparse":
            eviction = "sparse (baseline attention score, no checkpoint)"
        else:
            eviction = f"student ({student_path})"
        print(f"[DreamFutureOC] keep_ratio={self._keep_ratio} "
              f"block_length={self._block_length} max_seq_len={max_seq_len} "
              f"eviction={eviction} seed={self._seed} "
              f"decoding={self._decoding.metadata()}", flush=True)

    def _encode(self, text: str, max_length: int):
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False)
        return self.tokenizer(chat, return_tensors="pt", truncation=True,
                              add_special_tokens=True, max_length=max_length)

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        # Measured on the un-templated string, as theirs is: OpenCompass uses
        # this only to sort and shard prompts, so it has to mean the same thing
        # on both rows, not be maximally accurate.
        text = _convert_base_messages([prompt])[0]
        return len(self.tokenizer(text,
                                  add_special_tokens=add_special_tokens)["input_ids"])

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int) -> List[str]:
        # GenInferencer inspects this signature and only passes
        # stopping_criteria if it is declared. Dream's block schedule cannot
        # stop early, and their wrapper drops stop words for Dream too, so
        # leaving it undeclared matches the baseline instead of silently
        # accepting a knob that does nothing.
        from future_dllm.dream_generate import generate

        gen_length = int(max_out_len)
        if gen_length % self._block_length:
            # generation_utils.py:389 asserts rather than rounding. Rounding
            # here would quietly give one row a longer budget than the other.
            raise ValueError(
                f"max_out_len ({gen_length}) must be divisible by block_length "
                f"({self._block_length})")

        eos = self.tokenizer.eos_token
        outputs = []
        for text in _convert_base_messages(inputs):
            item_seed = self._sample_seed(self._seed, text)
            random.seed(item_seed)
            np.random.seed(item_seed % (2**32))
            torch.manual_seed(item_seed)
            torch.cuda.manual_seed_all(item_seed)

            tokens = self._encode(text, self.max_seq_len)
            ids = tokens["input_ids"].to(self.model.device)
            out = generate(self.model, ids, gen_length=gen_length,
                           block_length=self._block_length,
                           eviction_method=self._eviction_method,
                           cache_scorer=self._scorer,
                           **self._decoding.generation_kwargs(gen_length))
            # Their decode keeps special tokens and cuts at the first EOS, so
            # anything the block schedule emits after EOS is dropped.
            text_out = self.tokenizer.decode(out[0, ids.shape[1]:].tolist())
            outputs.append(text_out.split(eos)[0] if eos else text_out)
        return outputs
