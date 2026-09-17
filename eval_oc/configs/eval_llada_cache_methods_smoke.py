"""Two items per dataset: proves the chain before committing GPU days.

    scripts/run_oc_mc.sh cache-llada-smoke
"""
from mmengine.config import read_base

with read_base():
    from .eval_llada_cache_methods import datasets, eval, infer, models

datasets = [dict(d) for d in datasets]
for _d in datasets:
    _d['reader_cfg'] = dict(_d['reader_cfg'], test_range='[0:2]')

work_dir = 'outputs/oc_llada_cache_methods_smoke'
