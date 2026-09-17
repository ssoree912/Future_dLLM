"""OpenCompass multiple-choice rows for the two published cache methods, Dream.

Same datasets, prompts, max_seq_len and max_out_len as ``eval_dream_mc.py``, so
these rows drop straight into that table. What differs per row is only the
method: Fast-dLLM's block/dual KV cache with parallel decoding, and dLLM-Cache's
adaptive feature cache.

Cache intervals come from each repo's own Dream scripts. Neither ships an
ARC-C or PIQA script, so the GPQA values are used throughout -- GPQA is in this
suite and is the closest multiple-choice analogue they tuned.

    FUTURE_DLLM_MODEL=$PWD/model/Dream-v0-Instruct-7B scripts/run_oc_mc.sh cache-dream
"""
from mmengine.config import read_base

from eval_oc.cache_methods_model import DreamDLLMCacheOC, DreamFastDLLMOC
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets

    from ..datasets.arc_c.arc_c_gen_test import ARC_c_datasets
    from ..datasets.gpqa.gpqa_gen_5shot import gpqa_datasets

datasets = [*gpqa_datasets, *ARC_c_datasets, *piqa_datasets]

max_seq_len = 2048
max_out_len = 256

models = [
    dict(
        type=DreamFastDLLMOC,
        abbr="dream-fastdllm",
        path="",                        # <- FUTURE_DLLM_MODEL
        use_cache=True,
        dual_cache=True,
        threshold=0.9,
        block_length=32,
        dream_alg="confidence_threshold",   # their own contribution on Dream
        dream_alg_temp=0.0,
        dream_steps=0,                     # 0 = max_out_len / block_length here
        dream_temperature=0.0,             # dream/eval.py default, unset by their scripts
        dream_top_p=None,
        dream_seed=2025,
        max_seq_len=max_seq_len,
        max_out_len=max_out_len,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=DreamDLLMCacheOC,
        abbr="dream-dllmcache",
        path="",
        is_feature_cache=True,
        prompt_interval_steps=10,       # their run_Dream_gpqa_Instruct.sh
        gen_interval_steps=8,
        transfer_ratio=0.25,
        dream_alg="entropy",
        dream_alg_temp=0.0,
        dream_steps=256,
        dream_temperature=0.2,          # Dream's own README setting, which they follow
        dream_top_p=0.95,
        dream_seed=2025,
        max_seq_len=max_seq_len,
        max_out_len=max_out_len,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=1,
                task=dict(type=OpenICLInferTask)),
)
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=1,
                task=dict(type=OpenICLEvalTask)),
)

work_dir = "outputs/oc_dream_cache_methods"
