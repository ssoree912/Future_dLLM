"""lm-eval entry point that registers the models before dispatching.

    python eval/run.py --model Dream_future     --model_args "..." --tasks local_gsm8k ...
    python eval/run.py --model Sparse_dLLM_Dream --model_args "..." --tasks local_gsm8k ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval.lm_eval_model               # noqa: F401  (LLaDA_future, Dream_future)
import eval.lm_eval_model_sparse_dllm   # noqa: F401  (Sparse_dLLM_Dream)
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
