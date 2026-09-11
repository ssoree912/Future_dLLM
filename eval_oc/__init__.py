"""OpenCompass integration for future_dllm.

Kept separate from ``eval/`` (lm-eval) because the two harnesses answer
different questions: lm-eval scores generation here, OpenCompass scores the
multiple-choice suite the way Sparse-dLLM reports it. Neither imports the
other, and neither imports a Sparse-dLLM checkout.
"""
from .model import DreamFutureOC

__all__ = ["DreamFutureOC"]
