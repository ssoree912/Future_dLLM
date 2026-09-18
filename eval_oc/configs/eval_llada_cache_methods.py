"""OpenCompass multiple-choice rows for the two published cache methods, LLaDA.

Same datasets, prompts, max_seq_len and max_out_len as ``eval_llada_mc.py``, so
these rows drop straight into that table. What differs per row is only the
method: Fast-dLLM's block/dual KV cache with parallel decoding, and dLLM-Cache's
adaptive feature cache.

Cache intervals come from each repo's own LLaDA scripts. Neither ships an
ARC-C or PIQA script, so the GPQA values are used throughout -- GPQA is in this
suite and is the closest multiple-choice analogue they tuned.

    FUTURE_DLLM_LLADA_MODEL=$PWD/model/LLaDA-8B-Instruct scripts/run_oc_mc.sh cache-llada
"""
from mmengine.config import read_base

from eval_oc.cache_methods_model import LLaDADLLMCacheOC, LLaDAFastDLLMOC
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
        type=LLaDAFastDLLMOC,
        abbr="llada-fastdllm",
        path="",                        # <- FUTURE_DLLM_LLADA_MODEL
        use_cache=True,
        dual_cache=True,
        threshold=0.9,
        block_length=32,
        llada_steps=0,                  # 0 = one denoising step per token
        llada_temperature=0.0,
        llada_remasking="low_confidence",
        llada_seed=2025,
        max_seq_len=max_seq_len,
        max_out_len=max_out_len,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=LLaDADLLMCacheOC,
        abbr="llada-dllmcache",
        path="",
        is_feature_cache=True,
        prompt_interval_steps=50,       # their run_LLaDA_gpqa_Instruct.sh
        gen_interval_steps=6,
        transfer_ratio=0.25,
        block_length=32,
        llada_steps=0,
        llada_temperature=0.0,
        llada_cfg_scale=0.0,
        llada_remasking="low_confidence",
        llada_seed=2025,
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

work_dir = "outputs/oc_llada_cache_methods"
