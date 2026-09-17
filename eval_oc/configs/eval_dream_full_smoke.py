"""dream-full alone: the no-eviction ceiling, scored the same way as our rows.

keep_ratio 1.0 makes eviction_method inert, so this row needs no scorer. One
row per config because --debug never releases a loaded model.
"""
from mmengine.config import read_base

with read_base():
    from .eval_dream_mc import datasets, eval, infer, models

models = [m for m in models if m['abbr'] == 'dream-full']

datasets = [dict(d) for d in datasets]
for _d in datasets:
    _d['reader_cfg'] = dict(_d['reader_cfg'], test_range='[0:2]')

work_dir = 'outputs/oc_dream_full_smoke'
