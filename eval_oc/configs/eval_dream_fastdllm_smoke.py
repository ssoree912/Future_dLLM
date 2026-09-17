"""dream-fastdllm alone: one model row per invocation.

OpenCompass's --debug keeps tasks in-process and never releases a loaded model,
so a config with two rows holds two checkpoints at once and the second one OOMs.
Splitting the rows is what keeps a run inside one card.
"""
from mmengine.config import read_base

with read_base():
    from .eval_dream_cache_methods import datasets, eval, infer, models

models = [m for m in models if m['abbr'] == 'dream-fastdllm']

datasets = [dict(d) for d in datasets]
for _d in datasets:
    _d['reader_cfg'] = dict(_d['reader_cfg'], test_range='[0:2]')

work_dir = 'outputs/oc_dream_fastdllm_smoke'
