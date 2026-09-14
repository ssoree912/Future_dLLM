"""Two items per dataset of eval_llada_mc_k0.2.py: proves the chain before the run.

The LLaDA OpenCompass path had never been executed before this checkpoint, and
its failure modes -- an unloadable scorer, a prompt built the wrong way, a
postprocessor that never matches -- all surface as a bad accuracy number rather
than a crash. On the full suite that number costs ten GPU-hours to find out.

    scripts/run_oc_mc.sh llada-k02-smoke
"""
from mmengine.config import read_base

with read_base():
    from .eval_llada_mc_k02 import datasets, eval, infer, models

datasets = [dict(d) for d in datasets]
for _d in datasets:
    _d['reader_cfg'] = dict(_d['reader_cfg'], test_range='[0:2]')

work_dir = 'outputs/oc_llada_mc_k0.2_smoke'
