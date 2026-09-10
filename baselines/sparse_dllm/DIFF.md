# Departure from upstream

`dream/` is OpenMOSS/Sparse-dLLM's `opencompass/models/sparse_dllm/dream/`.
Everything is upstream's except six lines in `modeling_dream.py`, reproduced in
`DIFF.patch` and shown by:

    diff -r <upstream>/dream baselines/sparse_dllm/dream

The addition calls `customcache.record_attention(...)` on the block-only
forward. Teacher extraction needs the attention a finished block pays to its
cached columns, and post-RoPE queries and keys exist only inside that function.
Their own `CustomCache` has no `record_attention`, so the branch is not taken
and the baseline runs upstream's code path exactly.

Selection itself is untouched. Our scorer is injected from
`future_dllm/sparse_dllm_student.py`, which subclasses their `CustomCache` and
wraps the decoder layer at import time rather than editing these files.
