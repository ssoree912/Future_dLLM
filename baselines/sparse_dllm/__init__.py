"""Sparse-dLLM's own Dream implementation, vendored unmodified.

`dream/` is copied byte-for-byte from OpenMOSS/Sparse-dLLM
(`opencompass/models/sparse_dllm/dream/`) so the baseline is their code, not a
reimplementation of it. Nothing in this directory is edited; the lm-eval
wrapper that drives it lives in `eval/lm_eval_model_sparse_dllm.py`.

Keeping it separate from `future_dllm/` matters: `future_dllm` also carries a
`selection="sparse_dllm"` path that applies their ranking rule inside our
cache. That one isolates the scorer; this one checks that our port did not
drift from theirs in the first place. The two answer different questions and
must not be collapsed.
"""
