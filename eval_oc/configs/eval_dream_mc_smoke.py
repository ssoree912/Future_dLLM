"""Two items per dataset, one model row: proves the chain before the real run.

Exists because the failure modes of this integration -- a wrong chat template,
an unloadable scorer, a postprocessor that never matches -- all look like a bad
accuracy number, and on the full suite that number costs a day of GPU to get.
Here it costs minutes.

    scripts/run_oc_mc.sh smoke
"""
from mmengine.config import read_base

with read_base():
    from .eval_dream_mc import datasets, eval, infer, models

# read_base hands back real dicts, so these are ordinary mutations.
datasets = [dict(d) for d in datasets if not d['abbr'].startswith('lukaemon')]
for _d in datasets:
    _d['reader_cfg'] = dict(_d['reader_cfg'], test_range='[0:2]')

# The student row exercises every code path the other two do, plus the scorer.
models = [m for m in models if m['abbr'] == 'dream-ours-k0.5']

work_dir = 'outputs/oc_dream_mc_smoke'
