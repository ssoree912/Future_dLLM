"""Sparse-dLLM on LLaDA at keep_ratio 0.1, the multiple-choice suite.

Same datasets, prompts and decoding as every other LLaDA row here; only the
eviction differs. keep_ratio 0.1 rather than the 0.5 in eval_llada_mc.py,
matching the generative sweep this pairs with. No scorer needed - the sparse
method scores by attention.
"""
from mmengine.config import read_base

with read_base():
    from .eval_llada_mc import datasets, eval, infer, models

models = [dict(m, abbr='llada-sparse-k0.1', keep_ratio=0.1)
          for m in models if m['abbr'] == 'llada-sparse-k0.5']

work_dir = 'outputs/oc_llada_sparse_k01'
