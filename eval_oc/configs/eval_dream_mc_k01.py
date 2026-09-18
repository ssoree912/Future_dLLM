"""Keep 0.1, greedy -- the multiple-choice half, ours only.

Companion to the lm-eval run of the same two rows: same model, same keep ratio,
same decoding (``temperature=0``, ``top_p=0.95``, ``steps`` cap 512, seed 2025).
These four benchmarks cannot go through lm-eval, whose ``multiple_choice`` path
scores by loglikelihood -- undefined for a bidirectionally-attending model.

MMLU is excluded as asked: 14042 questions against 3201 for the other three.

Keep 0.1 is where the two rows are expected to separate. At 0.2 the baseline
loses almost nothing (GSM8K -1.29, LongBench -0.84), so there is no headroom to
win; at 0.1 on GSM8K it was 46.50 against 60.50 for ours.

    scripts/run_oc_mc.sh k01
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets

    from ..datasets.ARC_c.ARC_c_test_gen import ARC_c_datasets
    from ..datasets.gpqa.gpqa_gen_5shot import gpqa_datasets
    from .eval_dream_mc import eval, infer, models as _base_models

datasets = [*gpqa_datasets, *ARC_c_datasets, *piqa_datasets]

_base = _base_models[0]
models = [
    dict(_base, abbr='dream-ours-k0.1', keep_ratio=0.1,
         eviction_method='student', student_path='',
         dream_temperature=0.0, dream_steps=512),
]

work_dir = 'outputs/oc_dream_mc_k01'
