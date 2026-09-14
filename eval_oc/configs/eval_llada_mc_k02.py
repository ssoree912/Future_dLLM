"""OpenCompass multiple-choice sweep for LLaDA at keep_ratio 0.2: our row only.

Same harness and decoding as eval_llada_mc.py -- Sparse-dLLM's published LLaDA
settings (steps 256, block_length 32, greedy, low-confidence remasking,
kernel_size 3, seed 2025, max_seq_len 2048, max_out_len 256) -- with two
deliberate differences:

  * keep_ratio 0.2 rather than 0.5, the ratio this row is being measured at;
  * the student row alone. The full-cache and Sparse-dLLM rows are the same
    three datasets again at ~10 GPU-hours each, and neither is what this run is
    for. Adding them later costs nothing here: drop the row back into `_rows`.

The LongBench and generative benchmarks for this checkpoint go through lm-eval
(scripts/run_eval.sh); only ARC-C, PIQA and GPQA are scored here, because
OpenCompass scores them generatively the way Sparse-dLLM reports them.

    FUTURE_DLLM_LLADA_MODEL=$PWD/model/LLaDA-8B-Instruct \
    FUTURE_DLLM_LLADA_STUDENT=$PWD/artifacts/ckpts/<run>/checkpoint-epoch6 \
        scripts/run_oc_mc.sh llada-k02
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
keep_ratio = 0.2

_rows = [
    ("llada-ours-k0.2", keep_ratio, "student"),
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

work_dir = "outputs/oc_llada_mc_k0.2"
