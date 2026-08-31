"""future_dllm — KV cache eviction for diffusion LLMs, driven by future attention.

Standalone package: lm-eval imports this directly, no OpenCompass dependency.
"""
from .modeling_llada import (LLaDAModelLM, CustomCache,
                             sparse_dllm_current_score)
from .llada_generate import generate, add_gumbel_noise, get_num_transfer_tokens
from .student_cache import (PromptUtilityStudent, StudentConfig,
                            load_prompt_utility_student)

__all__ = ["LLaDAModelLM", "CustomCache", "generate", "add_gumbel_noise",
           "get_num_transfer_tokens", "PromptUtilityStudent", "StudentConfig",
           "load_prompt_utility_student", "sparse_dllm_current_score"]
