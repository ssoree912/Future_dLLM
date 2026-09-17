"""llada-full alone: the no-eviction ceiling, scored the same way as our rows.

keep_ratio 1.0 makes eviction_method inert, so this row needs no scorer. One
row per config because --debug never releases a loaded model.
"""
from mmengine.config import read_base

with read_base():
    from .eval_llada_mc import datasets, eval, infer, models

models = [m for m in models if m['abbr'] == 'llada-full']

work_dir = 'outputs/oc_llada_full'
