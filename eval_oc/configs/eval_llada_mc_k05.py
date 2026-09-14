"""eval_llada_mc_k02.py at keep_ratio 0.5: same row, the third ratio.

Note that issue #3's existing 0.5 row came from a different checkpoint (the
delayed `checkpoint-best-15c3dfff`), which is not in this tree. This config
runs the same scorer as the 0.1 and 0.2 rows, so its numbers belong beside
those two rather than in the posted 0.5 row.

    FUTURE_DLLM_LLADA_MODEL=... FUTURE_DLLM_LLADA_STUDENT=... \
        scripts/run_oc_mc.sh llada-k05
"""
from mmengine.config import read_base

with read_base():
    from .eval_llada_mc_k02 import datasets, eval, infer, models

keep_ratio = 0.5
models = [dict(m, abbr="llada-ours-k0.5", keep_ratio=keep_ratio) for m in models]

work_dir = "outputs/oc_llada_mc_k0.5"
