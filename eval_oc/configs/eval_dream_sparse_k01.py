"""Sparse-dLLM on Dream at keep_ratio 0.1, greedy.

Same datasets, prompts and max_out_len as eval_dream_mc.py; keep_ratio 0.1 and
temperature 0 instead of their 0.5 / 0.2, to pair with the generative sweep at
those settings.
"""
from mmengine.config import read_base

with read_base():
    from .eval_dream_mc import datasets, eval, infer, models

models = [dict(m, abbr='dream-sparse-k0.1', keep_ratio=0.1,
               dream_temperature=0.0, dream_top_p=0.95)
          for m in models if m['abbr'] == 'dream-sparse-k0.5']

work_dir = 'outputs/oc_dream_sparse_k01'
