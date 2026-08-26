"""lm-eval entry point that registers future_dllm's model first.

    python eval/run.py --model LLaDA_future --model_args "..." --tasks local_gsm8k ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval.lm_eval_model   # noqa: F401  (registers LLaDA_future)
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
