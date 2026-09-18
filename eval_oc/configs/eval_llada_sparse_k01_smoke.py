"""Two items per dataset."""
from mmengine.config import read_base

with read_base():
    from .eval_llada_sparse_k01 import datasets, eval, infer, models

datasets = [dict(d) for d in datasets]
for _d in datasets:
    _d['reader_cfg'] = dict(_d['reader_cfg'], test_range='[0:2]')

work_dir = 'outputs/oc_llada_sparse_k01_smoke'
