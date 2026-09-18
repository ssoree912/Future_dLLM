"""Dream-v0-Instruct-7B as it ships: no eviction, no cache method, greedy.

The top row of the table -- the checkpoint's own diffusion_generate with nothing
wrapped around it. DreamDLLMCacheOC with is_feature_cache=False registers no
hooks, so what runs is the stock path.

temperature 0.2 with top_p 0.95 and alg=entropy: the combination Dream's own
README uses in its usage example. Greedy on this decoder commits an eos at
position 0 and repeats that deterministically - 96% empty answers on multi_news,
17% on GSM8K - so it is not a usable setting for this checkpoint.
"""
from mmengine.config import read_base

from eval_oc.cache_methods_model import DreamDLLMCacheOC
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from ..datasets.arc_c.arc_c_gen_test import ARC_c_datasets
    from ..datasets.gpqa.gpqa_gen_5shot import gpqa_datasets
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets

datasets = [*gpqa_datasets, *ARC_c_datasets, *piqa_datasets]

models = [
    dict(
        type=DreamDLLMCacheOC,
        abbr="dream-origin",
        path="",                       # <- FUTURE_DLLM_MODEL
        is_feature_cache=False,        # no hooks: the stock decode
        dream_alg="entropy",
        dream_alg_temp=0.0,
        dream_steps=256,
        dream_temperature=0.2,
        dream_top_p=0.95,
        dream_seed=2025,
        max_seq_len=2048,
        max_out_len=256,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(type=LocalRunner, max_num_workers=1,
                         task=dict(type=OpenICLInferTask)))
eval = dict(partitioner=dict(type=NaivePartitioner),
            runner=dict(type=LocalRunner, max_num_workers=1,
                        task=dict(type=OpenICLEvalTask)))
work_dir = "outputs/oc_dream_origin"
