"""MMLU on its own, because it dominates the cost of the suite.

57 subjects, 14042 questions, three cache configurations: 42k generations at
256 diffusion steps each, against 9.6k for the rest of the multiple-choice
suite combined. Splitting it out means the other three datasets can produce
numbers while this one is still scheduled, and it can be subsampled or deferred
without touching the configs that are already running.

Same model rows and same settings as eval_dream_mc.py -- only the dataset
differs.

    scripts/run_oc_mc.sh mmlu
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen_79e572 import mmlu_datasets

    from .eval_dream_mc import eval, infer, models

datasets = mmlu_datasets

work_dir = 'outputs/oc_dream_mmlu'
