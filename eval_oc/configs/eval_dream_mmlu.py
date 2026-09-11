"""MMLU, subsampled to 25 questions per subject.

MMLU dominates the cost of the multiple-choice suite: 14042 questions against
9.6k for GPQA + ARC-C + PIQA combined, and at three cache rows and 256
diffusion steps per answer the full set is roughly 42k generations -- on the
order of five GPU-days. Twenty-five per subject keeps all 57 subjects and their
5-shot prompts intact at 1425 x 3 generations, about eleven hours.

    This number is therefore NOT directly comparable to Sparse-dLLM's published
    MMLU, which is the full set. It is comparable across the three rows here,
    which is what the cache-eviction claim needs: same questions, same prompts,
    same decoding, only the eviction policy differs.

Per-subject subsampling rather than a flat head of the concatenated set, so the
subject mix -- and with it the difficulty mix -- stays the one MMLU defines.
Raise MMLU_PER_SUBJECT to 100 or drop the slice entirely for a comparable run.

    scripts/run_oc_mc.sh mmlu
"""
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen_79e572 import mmlu_datasets

    from .eval_dream_mc import eval, infer, models

MMLU_PER_SUBJECT = 25

# read_base returns real dicts, so this is an ordinary mutation. reader_cfg is
# rebuilt rather than mutated in place: the same dict object is shared with the
# imported module, and editing it would leak into any other config read in the
# same process.
datasets = [
    dict(d, reader_cfg=dict(d['reader_cfg'],
                            test_range=f'[0:{MMLU_PER_SUBJECT}]'))
    for d in mmlu_datasets
]

work_dir = 'outputs/oc_dream_mmlu'
