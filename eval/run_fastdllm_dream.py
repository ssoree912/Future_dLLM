"""lm-eval entry point for the Dream_fastdllm model.

Kept separate from the LLaDA entry point on purpose: Fast-dLLM's llada and dream
directories each contain a package called `model`, and only one of them can win
an import in a single process.

    python eval/run_fastdllm_dream.py --model Dream_fastdllm --model_args "..." --tasks local_gsm8k ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval.lm_eval_model_fastdllm_dream   # noqa: F401  (registers Dream_fastdllm)
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
