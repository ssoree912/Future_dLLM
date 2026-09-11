"""OpenCompass model wrapper for LLaDA, mirroring Sparse-dLLM's LLaDA wrapper.

Deliberately a separate file from ``model.py`` rather than a shared base class
with the Dream wrapper. The two agree on almost nothing that matters:

                        Dream (model.py)              LLaDA (here)
    prompt              chat template, user turn      raw string, no template
    gen length          raises if not a block         rounded up to a block
    decode              keeps specials, cuts at EOS   skip_special_tokens=True
    stop words          dropped                       applied
    decoding            entropy, T=0.2, top_p=0.95    low_confidence, greedy

Those are not our choices -- each side matches what Sparse-dLLM does for that
family (``llada_wrapper.py`` against ``dream_wrapper_instruct.py``). Grafting
one family's handling onto the other is the exact mistake this repo already
made once with Dream's decoding, so the two live apart and share only the
prompt flattening and the seeding helper.

Note that LLaDA gets no chat template even for the instruct checkpoint: their
``Sparse_dLLM_LLaDACausalLM.generate`` tokenizes the flattened prompt directly,
and ``eval_sparse_dllm_llada_chat.py`` passes no meta_template. That looks like
an oversight from here, but it is what produced their published numbers, so it
is what the baseline row has to do.

STATUS: code only. Prompt construction is verified against their reference by
scripts/check_oc_prompt.py on the tokenizer alone; generation has never been
executed, because no LLaDA checkpoint or scorer is present in this repo.
"""

from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np
import torch
from opencompass.models.base import BaseModel

from .model import _convert_base_messages


class LLaDAFutureOC(BaseModel):
    """LLaDA with future-attention cache eviction, driven by OpenCompass."""

    def __init__(
        self,
        path: str = "",
        max_seq_len: int = 2048,
        block_length: int = 32,
        keep_ratio: float = 1.0,
        eviction_method: str = "student",
        student_path: str = "",
        llada_steps: int = 256,
        llada_temperature: float = 0.0,
        llada_cfg_scale: float = 0.0,
        llada_remasking: str = "low_confidence",
        llada_seed: int = 2025,
        meta_template: Optional[dict] = None,
    ):
        path = path or os.environ.get("FUTURE_DLLM_LLADA_MODEL", "")
        if not path:
            raise ValueError("set FUTURE_DLLM_LLADA_MODEL or pass path= in the config")
        if eviction_method == "student":
            student_path = student_path or os.environ.get("FUTURE_DLLM_LLADA_STUDENT", "")

        super().__init__(path=path, max_seq_len=max_seq_len,
                         meta_template=meta_template)
        import sys
        from pathlib import Path
        repo = Path(__file__).resolve().parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from transformers import AutoTokenizer

        from future_dllm import load_model, load_prompt_utility_student
        from future_dllm.dream_decoding import sample_seed

        self._block_length = int(block_length)
        self._keep_ratio = float(keep_ratio)
        self._eviction_method = str(eviction_method)
        self._seed = int(llada_seed)
        self._sample_seed = sample_seed
        self._decoding = dict(steps=int(llada_steps),
                              temperature=float(llada_temperature),
                              cfg_scale=float(llada_cfg_scale),
                              remasking=str(llada_remasking))

        self.model, self._backend = load_model(
            path, max_seq_len=max_seq_len, block_length=self._block_length,
            keep_ratio=self._keep_ratio)
        self.model.eval()
        if (self.model.config.keep_ratio != self._keep_ratio
                or self.model.config.block_len != self._block_length):
            raise RuntimeError(
                f"eviction settings did not reach the model: config says "
                f"keep_ratio={self.model.config.keep_ratio} "
                f"block_len={self.model.config.block_len}, expected "
                f"{self._keep_ratio} / {self._block_length}")
        # Their _load_model also sets config.kernel_size. Ours is not a config
        # field: cache.py takes the pooling width as pool_kernel_size and fixes
        # it at 3, which is the value their published config passes.
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tokenizer.truncation_side = "left"   # OpenCompass's own default

        self._scorer = None
        if self._eviction_method == "student":
            if not student_path:
                raise ValueError("eviction_method='student' requires a scorer: "
                                 "set FUTURE_DLLM_LLADA_STUDENT or pass student_path=")
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
        print(f"[LLaDAFutureOC] keep_ratio={self._keep_ratio} "
              f"block_length={self._block_length} max_seq_len={max_seq_len} "
              f"eviction={eviction} seed={self._seed} "
              f"decoding={self._decoding}", flush=True)

    def _encode(self, text: str, max_length: int):
        # No chat template: theirs tokenizes the flattened prompt as is.
        return self.tokenizer.batch_encode_plus(
            [text], return_tensors="pt", padding=True, truncation=True,
            add_special_tokens=True, max_length=max_length)

    def get_token_len(self, prompt: str, add_special_tokens: bool = True) -> int:
        text = _convert_base_messages([prompt])[0]
        return len(self.tokenizer(text,
                                  add_special_tokens=add_special_tokens)["input_ids"])

    @torch.no_grad()
    def generate(self, inputs: List[str], max_out_len: int,
                 stopping_criteria: List[str] = []) -> List[str]:
        # stopping_criteria is declared because GenInferencer only passes it to
        # a generate() that names it, and LLaDA -- unlike their Dream path --
        # actually cuts the decoded string on the stop words.
        from future_dllm.llada_generate import generate

        # Their order: round the budget up to a whole block first, then clamp
        # steps to it. Reversing the two would let steps exceed the rounded
        # budget and trip the steps % num_blocks assert inside generate().
        gen_length = int(max_out_len)
        if gen_length % self._block_length:
            gen_length = (gen_length // self._block_length + 1) * self._block_length
        steps = min(self._decoding["steps"], gen_length)

        outputs = []
        for text in _convert_base_messages(inputs):
            item_seed = self._sample_seed(self._seed, text)
            random.seed(item_seed)
            np.random.seed(item_seed % (2**32))
            torch.manual_seed(item_seed)
            torch.cuda.manual_seed_all(item_seed)

            ids = self._encode(text, self.max_seq_len)["input_ids"].to(self.model.device)
            out = generate(self.model, ids, gen_length=gen_length,
                           block_length=self._block_length, steps=steps,
                           temperature=self._decoding["temperature"],
                           cfg_scale=self._decoding["cfg_scale"],
                           remasking=self._decoding["remasking"],
                           mask_id=self._backend.mask_id,
                           eviction_method=self._eviction_method,
                           cache_scorer=self._scorer)
            text_out = self.tokenizer.decode(out[0, ids.shape[1]:].tolist(),
                                             skip_special_tokens=True)
            for stop in stopping_criteria:
                text_out = text_out.split(stop)[0]
            outputs.append(text_out)
        return outputs
