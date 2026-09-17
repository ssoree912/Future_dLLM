"""eval_llada_mc_k01.py, ARC-C alone.

The 0.1 row's ARC-C was scored on the dev split (295 items); every baseline row
-- llada-full, llada-dllmcache, llada-fastdllm, and the Dream rows -- was scored
on test (1,165), so the two cannot share a column. Only ARC-C moved, so PIQA
(1,838) and GPQA diamond (198) keep their existing numbers and are left out
here rather than re-run for nothing.

Everything else is inherited from the 0.1 config, so this run can only differ
from it in which datasets it covers.

    FUTURE_DLLM_LLADA_MODEL=... FUTURE_DLLM_LLADA_STUDENT=... \
        scripts/run_oc_mc.sh llada-k01-arc
"""
from mmengine.config import read_base

with read_base():
    from .eval_llada_mc_k01 import eval, infer, models
    from ..datasets.ARC_c.ARC_c_gen_test import ARC_c_datasets

datasets = [*ARC_c_datasets]

work_dir = "outputs/oc_llada_mc_k0.1_arc_test"
