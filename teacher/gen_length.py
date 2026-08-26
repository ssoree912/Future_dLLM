"""Read a teacher run's generation budget out of the eval task it is training for.

The teacher labels how much of the cache the finished answer needs, so the block
schedule it labels has to be the one the evaluation actually decodes: the task's
``generation_kwargs.max_gen_toks``, or lm-eval's own fallback when the task sets
none. Reading it here keeps the two from drifting apart by hand.
"""

from __future__ import annotations

from pathlib import Path

import yaml


TASKS = Path(__file__).resolve().parent.parent / "eval" / "tasks"

# lm-eval's HFLM.max_gen_toks, used for generative tasks that set no budget.
DEFAULT_MAX_GEN_TOKS = 256

# Which eval task each teacher dataset is training for, matching how
# scripts/run_eval.sh routes the dataset name.
DATASET_TASK = {
    "samsum": "longbench/samsum.yaml",
    "samsum_lb": "longbench/samsum.yaml",
    "trec_lb": "longbench/trec.yaml",
    "wiki2_lb": "longbench/2wikimqa.yaml",
    "musique": "longbench/musique.yaml",
    "qasper": "longbench/qasper.yaml",
    "gov_report": "longbench/gov_report.yaml",
    "multi_news": "longbench/multi_news.yaml",
    "repobench_p": "longbench/repobench-p.yaml",
    "gsm8k": "local/gsm8k.yaml",
    "math": "local/math.yaml",
    "math5s": "local/math.yaml",
    "math_ho_near": "local/math.yaml",
    "math_ho_far": "local/math.yaml",
}

# No task budget to read. MMLU is scored by loglikelihood over the four letters,
# so nothing is generated at all; MBPP has no eval task in this repo. These are
# the budgets their existing labels were built with.
NO_TASK_BUDGET = {
    "mmlu": 64,
    "mbpp": 128,
    "mbpp_full": 256,
}


def _load_yaml(path: Path) -> dict:
    loader = yaml.SafeLoader
    loader.add_constructor("!function", lambda l, n: l.construct_scalar(n))
    return yaml.load(path.read_text(), Loader=loader)


def resolve(dataset: str) -> tuple[int, str]:
    """Return (gen_length, where it came from) for a teacher dataset name."""
    if dataset in NO_TASK_BUDGET:
        return NO_TASK_BUDGET[dataset], "no eval task generates; repo default"

    try:
        rel = DATASET_TASK[dataset]
    except KeyError:
        raise SystemExit(
            f"no eval task known for --dataset {dataset}; add it to "
            f"teacher/gen_length.py or pass --gen-length"
        ) from None

    cfg = _load_yaml(TASKS / rel)
    budget = (cfg.get("generation_kwargs") or {}).get("max_gen_toks")
    if budget is None:
        return DEFAULT_MAX_GEN_TOKS, f"{rel} sets none; lm-eval default"
    return int(budget), f"{rel} max_gen_toks"


if __name__ == "__main__":
    for name in sorted(set(DATASET_TASK) | set(NO_TASK_BUDGET)):
        value, source = resolve(name)
        print(f"{name:14s} {value:5d}   {source}")
