"""lm-eval entry point that registers the dLLM-Cache model first.

    python eval/run_dllmcache.py --model LLaDA_dllmcache --model_args "..." --tasks local_gsm8k ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval.lm_eval_model_dllmcache   # noqa: F401  (registers LLaDA_dllmcache)
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
