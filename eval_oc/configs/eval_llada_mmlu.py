"""MMLU for LLaDA, subsampled to 25 questions per subject.

The LLaDA counterpart of eval_dream_mmlu.py, split out for the same reason: at
three rows the full 14042-question set is roughly four fifths of the whole
multiple-choice suite's GPU time.

    This number is therefore NOT directly comparable to Sparse-dLLM's published
    MMLU, which is the full set -- only across the three rows here.

STATUS: never executed; see eval_llada_mc.py.

    scripts/run_oc_mc.sh llada-mmlu
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen_79e572 import mmlu_datasets

    from .eval_llada_mc import eval, infer, models

MMLU_PER_SUBJECT = 25

datasets = [
    dict(d, reader_cfg=dict(d['reader_cfg'],
                            test_range=f'[0:{MMLU_PER_SUBJECT}]'))
    for d in mmlu_datasets
]

work_dir = 'outputs/oc_llada_mmlu'
