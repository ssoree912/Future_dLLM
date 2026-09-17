"""lm-eval entry point that registers the Fast-dLLM v1 model first.

    python eval/run_fastdllm.py --model LLaDA_fastdllm --model_args "..." --tasks local_gsm8k ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval.lm_eval_model_fastdllm   # noqa: F401  (registers LLaDA_fastdllm)
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
