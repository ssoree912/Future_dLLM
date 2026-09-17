"""Two items per dataset, the rows that can run without a scorer.

The LLaDA OpenCompass path has never been executed -- its docstring says so --
because no LLaDA checkpoint was present where it was written. One is present
here, so this config exercises everything downstream of the tokenizer that
check_oc_prompt.py cannot reach: model load, eviction settings reaching the
config, generate(), decode, stop words, postprocessor, evaluator.

The student row is excluded: no trained LLaDA scorer exists in this repo, and
`eviction_method='student'` raises without one. full and sparse together still
cover both branches of the eviction switch.

    FUTURE_DLLM_LLADA_MODEL=$PWD/model/LLaDA-8B-Instruct scripts/run_oc_mc.sh llada-smoke
"""
from mmengine.config import read_base

with read_base():
    from .eval_llada_mc import datasets, eval, infer, models

datasets = [dict(d) for d in datasets]
for _d in datasets:
    _d['reader_cfg'] = dict(_d['reader_cfg'], test_range='[0:2]')

# One row: two checkpoints cannot share a card under --debug.
models = [m for m in models if m['abbr'] == 'llada-full']

work_dir = 'outputs/oc_llada_mc_smoke'
