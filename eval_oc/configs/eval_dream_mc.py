"""OpenCompass multiple-choice sweep: Full cache / Sparse-dLLM / Ours.

Three rows over one Dream checkpoint, differing only in how the per-block KV
cache is pruned:

    dream-full          keep_ratio 1.0, no eviction          -- the ceiling
    dream-sparse-k0.5   Sparse-dLLM's attention score        -- the baseline
    dream-ours-k0.5     the trained prompt-utility scorer    -- this work

``max_seq_len``, ``max_out_len``, ``kernel_size``, ``keep_ratio`` and ``seed``
are Sparse-dLLM's own published settings for Dream-Instruct
(``myeval/eval_performance/eval_sparse_dllm_dream_chat.py``), so the baseline
row here should reproduce their reported numbers rather than a re-tuned variant
of them.

Datasets are imported from the installed OpenCompass package, untouched. The
one exception is GPQA: their 5-shot direct-answer config has no equivalent in
vanilla OpenCompass (which ships 0-shot CoT variants), so it is vendored under
``eval_oc/datasets/`` with attribution instead of being patched into the
OpenCompass tree.

MMLU is not here. At ~9 s/item it is 14042 x 3 rows = 42k generations, four
fifths of the whole suite's cost on its own, so it gets its own config
(``eval_dream_mmlu.py``) that can be scheduled independently. What is left is
ordered cheapest first -- GPQA 198, ARC-C 1172, PIQA 1838 -- so a configuration
error surfaces in minutes.

    scripts/run_oc_mc.sh

mmengine parses this file in lazy-import mode (any config using ``read_base``
is), where every imported name is a placeholder and calling one raises. So the
file holds literals only: ``path`` and ``student_path`` are left empty and the
wrapper resolves them from ``FUTURE_DLLM_MODEL`` / ``FUTURE_DLLM_STUDENT``, the
same environment variables ``scripts/run_eval.sh`` already uses for lm-eval.
"""
from mmengine.config import read_base

from eval_oc.model import DreamFutureOC
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
keep_ratio = 0.5                        # Sparse-dLLM's published Dream setting

# (abbr, keep_ratio, eviction_method). The full-cache row prunes nothing, so
# its eviction_method is inert -- it is the shared no-eviction ceiling.
_rows = [
    ("dream-full", 1.0, "sparse"),
    ("dream-sparse-k0.5", keep_ratio, "sparse"),
    ("dream-ours-k0.5", keep_ratio, "student"),
]

models = [
    dict(
        type=DreamFutureOC,
        abbr=abbr,                      # distinct, or the rows collide on disk
        path="",                        # <- FUTURE_DLLM_MODEL
        student_path="",                # <- FUTURE_DLLM_STUDENT, student row only
        keep_ratio=row_keep_ratio,
        eviction_method=method,
        block_length=32,
        dream_steps=256,
        dream_alg="entropy",
        dream_temperature=0.2,
        dream_top_p=0.95,
        dream_seed=2025,
        max_seq_len=max_seq_len,
        max_out_len=max_out_len,
        batch_size=1,                   # the block schedule is per-sequence
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

work_dir = "outputs/oc_dream_mc"
