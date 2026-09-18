"""Keep 0.5 at temperature 0.2 -- the multiple-choice half of our row.

Companion to the lm-eval sweep of the same student: same checkpoint, same keep
ratio, same decoding (``alg=entropy``, ``temperature=0.2``, ``top_p=0.95``,
``steps`` cap 512, seed 2025). These benchmarks cannot go through lm-eval, whose
``multiple_choice`` path scores by loglikelihood -- undefined for a
bidirectionally-attending model.

Distinct from eval_dream_mc_k01_t02.py, the same row at keep 0.1. eval_dream_mc_k01.py is the greedy (``temperature=0``)
variant for the student trained under that decoding. The two are not
interchangeable: the scorer is fit to the labels its teacher was extracted
under, and the wrapper now refuses the mismatch rather than quietly producing a
row that looks like the others.

MMLU is excluded as before: 14042 questions against 3201 for the other three.

    FUTURE_DLLM_STUDENT=<checkpoint-best> scripts/run_oc_mc.sh k05-t02
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets

    from ..datasets.arc_c.arc_c_gen_test import ARC_c_datasets
    from ..datasets.gpqa.gpqa_gen_5shot import gpqa_datasets
    from .eval_dream_mc import eval, infer, models as _base_models

datasets = [*gpqa_datasets, *ARC_c_datasets, *piqa_datasets]

_base = _base_models[0]
models = [
    dict(_base, abbr='dream-ours-k0.5-t0.2', keep_ratio=0.5,
         eviction_method='student', student_path='',
         dream_temperature=0.2, dream_top_p=0.95, dream_steps=512,
         dream_alg='entropy', dream_seed=2025),
]

work_dir = 'outputs/oc_dream_mc_k05_t02'
