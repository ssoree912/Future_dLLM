"""eval_llada_mc_k02.py at keep_ratio 0.1: same row, the other ratio.

The 0.1 row's ARC-C / PIQA / GPQA on issue #3 were scored by lm-eval's
multiple-choice loglikelihood, which is not the harness the 0.2 row (or
Sparse-dLLM's published numbers) uses. Rerunning them here puts both ratios on
the same generative scoring, so the two rows can be read against each other.

Everything else -- datasets, decoding, checkpoint, max_out_len -- is inherited,
so the two ratios can only ever differ in the ratio itself.

    FUTURE_DLLM_LLADA_MODEL=... FUTURE_DLLM_LLADA_STUDENT=... \
        scripts/run_oc_mc.sh llada-k01
"""
from mmengine.config import read_base

with read_base():
    from .eval_llada_mc_k02 import datasets, eval, infer, models

keep_ratio = 0.1
models = [dict(m, abbr="llada-ours-k0.1", keep_ratio=keep_ratio) for m in models]

work_dir = "outputs/oc_llada_mc_k0.1"
