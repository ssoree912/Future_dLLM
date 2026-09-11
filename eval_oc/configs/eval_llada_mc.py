"""OpenCompass multiple-choice sweep for LLaDA: Full cache / Sparse-dLLM / Ours.

The LLaDA counterpart of eval_dream_mc.py. Settings come from Sparse-dLLM's own
published config (``myeval/eval_performance/eval_sparse_dllm_llada_chat.py``):
steps 256 and block_length 32 -- which their config passes to override the
wrapper's 128 default, so it is written out explicitly here rather than left to
a default -- with greedy sampling, low-confidence remasking, kernel_size 3,
keep_ratio 0.5, seed 2025, max_seq_len 2048 and max_out_len 256.

Note the decoding differs from Dream's on every axis (greedy and
low-confidence, against entropy at temperature 0.2 with top_p 0.95). That is
each model's own native schedule, not a choice: LLaDA is natively masked, Dream
was adapted from an autoregressive Qwen2.

STATUS: never executed. No LLaDA checkpoint or trained scorer is present in this
repo, so only the `sparse` and full-cache rows could run at all today, and
neither has. Prompt construction is verified against their reference by
scripts/check_oc_prompt.py; everything downstream of the tokenizer is untested.

    FUTURE_DLLM_LLADA_MODEL=... FUTURE_DLLM_LLADA_STUDENT=... \
        scripts/run_oc_mc.sh llada
"""
from mmengine.config import read_base

from eval_oc.llada_model import LLaDAFutureOC
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets

    from ..datasets.gpqa.gpqa_gen_5shot import gpqa_datasets

datasets = [*gpqa_datasets, *ARC_c_datasets, *piqa_datasets]

max_seq_len = 2048
max_out_len = 256
keep_ratio = 0.5

_rows = [
    ("llada-full", 1.0, "sparse"),
    ("llada-sparse-k0.5", keep_ratio, "sparse"),
    ("llada-ours-k0.5", keep_ratio, "student"),
]

models = [
    dict(
        type=LLaDAFutureOC,
        abbr=abbr,
        path="",                        # <- FUTURE_DLLM_LLADA_MODEL
        student_path="",                # <- FUTURE_DLLM_LLADA_STUDENT
        keep_ratio=row_keep_ratio,
        eviction_method=method,
        block_length=32,
        llada_steps=256,
        llada_temperature=0.0,
        llada_cfg_scale=0.0,
        llada_remasking="low_confidence",
        llada_seed=2025,
        max_seq_len=max_seq_len,
        max_out_len=max_out_len,
        batch_size=1,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
    for abbr, row_keep_ratio, method in _rows
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

work_dir = "outputs/oc_llada_mc"
