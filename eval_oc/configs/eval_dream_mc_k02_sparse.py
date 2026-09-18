"""Sparse-dLLM baseline at keep 0.2, greedy -- the multiple-choice half.

The companion to the lm-eval run of the same row: same model, same keep ratio,
same decoding (``temperature=0``, ``top_p=0.95``, ``steps`` cap 512, seed 2025).
These four benchmarks cannot go through lm-eval at all, because its
``multiple_choice`` path scores by loglikelihood and Dream attends
bidirectionally, so a position reads its own answer token.

MMLU is excluded as asked; it is 14042 questions against 3201 for the other
three combined.

    scripts/run_oc_mc.sh k02-sparse
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets

    from ..datasets.ARC_c.ARC_c_test_gen import ARC_c_datasets
    from ..datasets.gpqa.gpqa_gen_5shot import gpqa_datasets
    from .eval_dream_mc import eval, infer, models as _base_models

datasets = [*gpqa_datasets, *ARC_c_datasets, *piqa_datasets]

# read_base hands back real dicts. Rebuild rather than mutate in place: the same
# objects are shared with the imported module.
models = [
    dict(_base_models[0],
         abbr='dream-sparse-k0.2',
         keep_ratio=0.2,
         eviction_method='sparse',
         student_path='',
         dream_temperature=0.0,
         dream_steps=512)
]

work_dir = 'outputs/oc_dream_mc_k02_sparse'
