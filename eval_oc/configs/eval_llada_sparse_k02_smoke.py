"""Sparse-dLLM on LLaDA at keep_ratio 0.2: two items per dataset.

Same datasets, prompts and decoding as every other LLaDA row here; only the
eviction differs. keep_ratio 0.2 rather than the 0.5 in eval_llada_mc.py,
matching the generative sweep this pairs with. No scorer needed - the sparse
method scores by attention.
"""
from mmengine.config import read_base

with read_base():
    from .eval_llada_sparse_k02 import datasets, eval, infer, models


datasets = [dict(d) for d in datasets]
for _d in datasets:
    _d['reader_cfg'] = dict(_d['reader_cfg'], test_range='[0:2]')

work_dir = 'outputs/oc_llada_sparse_k02_smoke'
