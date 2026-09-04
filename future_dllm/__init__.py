"""future_dllm — KV cache eviction for diffusion LLMs, driven by future attention.

Standalone package: lm-eval imports this directly, no OpenCompass dependency.

Two backends share one eviction implementation (``cache.CustomCache``) and one
scorer (``student_cache.PromptUtilityStudent``):

  * LLaDA  — ``modeling_llada`` / ``llada_generate``
  * Dream  — ``modeling_dream`` / ``dream_generate``

``backends.load_model`` picks between them from the checkpoint's ``model_type``,
so the teacher and student scripts never branch on the family themselves.
"""
from .cache import CustomCache, sparse_dllm_current_score
from .modeling_llada import LLaDAModelLM
from .llada_generate import generate, add_gumbel_noise, get_num_transfer_tokens
from .student_cache import (PromptUtilityStudent, StudentConfig,
                            load_prompt_utility_student)
from .backends import Backend, detect_family, load_model

__all__ = ["LLaDAModelLM", "CustomCache", "generate", "add_gumbel_noise",
           "get_num_transfer_tokens", "PromptUtilityStudent", "StudentConfig",
           "load_prompt_utility_student", "sparse_dllm_current_score",
           "Backend", "detect_family", "load_model"]
