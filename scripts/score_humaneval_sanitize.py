"""Score a HumanEval samples file two ways, to isolate the extraction step.

    python scripts/score_humaneval_sanitize.py <samples_*.jsonl> [...]

``harness``   what ``local_humaneval.yaml`` reports: ``doc["prompt"] + response``
              executed as-is (lm-eval's own build_predictions).
``sanitize``  what dLLM-Cache and Fast-dLLM report: the response is stripped of a
              markdown code fence, glued to the prompt, and passed through
              dLLM-Cache's ``sanitize`` - which keeps the longest syntactically
              valid line span and then only the entry point and what it calls.

Both are executed by the same runner (lm-eval's guarded ``check_correctness``, the
one ``local_humaneval_utils.pass_at_k`` uses), so the only thing that differs is
how the candidate program is extracted from the model's answer.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT.parent
SANITIZER = WORKSPACE / "dLLM-cache" / "metrics" / "humaneval_pass@1.py"

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
sys.path.insert(0, str(REPO_ROOT / "eval" / "tasks"))


def _load_sanitizer():
    """Import dLLM-Cache's scorer by path - '@' makes the name unimportable."""
    if not SANITIZER.is_file():
        raise FileNotFoundError(f"no sanitizer at {SANITIZER}")
    spec = importlib.util.spec_from_file_location("dllmcache_humaneval", SANITIZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sanitize


def _harness_prediction(doc: dict, response: str) -> str:
    return doc["prompt"] + response


def _sanitize_prediction(sanitize, doc: dict, response: str) -> str:
    body = response.split("```python\n", 1)[-1].split("```")[0]
    try:
        return sanitize(doc["prompt"] + "\n" + body, doc["entry_point"])
    except Exception:
        # sanitize parses with ast; an answer with no valid span scores zero either way
        return ""


def score(path: Path) -> None:
    from local_humaneval_utils import pass_at_k

    sanitize = _load_sanitizer()
    docs, responses = [], []
    with open(path) as fh:
        for line in fh:
            record = json.loads(line)
            docs.append(record["doc"])
            responses.append(record["resps"][0][0])

    references = [f"{d['test']}\ncheck({d['entry_point']})" for d in docs]
    variants = {
        "harness ": [[_harness_prediction(d, r)] for d, r in zip(docs, responses)],
        "sanitize": [[_sanitize_prediction(sanitize, d, r)] for d, r in zip(docs, responses)],
    }

    fenced = sum(1 for r in responses if "```" in r)
    print(f"{path.name}  n={len(docs)}  코드펜스 포함 응답 {fenced}/{len(docs)}")
    for name, predictions in variants.items():
        result = pass_at_k(references, predictions, k=[1])
        print(f"    {name}  pass@1 = {result['pass@1'] * 100:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for argument in sys.argv[1:]:
        score(Path(argument))
