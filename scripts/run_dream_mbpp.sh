#!/usr/bin/env bash
# MBPP on Dream, three rows: stock checkpoint, Fast-dLLM, dLLM-Cache.
#
# The task is dLLM-cache's own `--tasks mbpp --num_fewshot 3
# --apply_chat_template`, ported to local parquet on feat#3 -- 500 items, gen
# 512, and the one task in this suite that takes the chat template.
#
# Cheapest row first: Fast-dLLM's dual cache runs several tokens per step, the
# feature cache one, and the stock row has no cache at all.
set -uo pipefail
cd /workspace/dllm/Future_dLLM
source activate future-dllm
python -c "import torch" || { echo "torch missing - wrong env"; exit 1; }

SUMMARY="logs/eval/dream_mbpp_$(date +%Y%m%d_%H%M%S).txt"
export TEMPERATURE=0.2 TOP_P=0.95
echo "dream mbpp, 500 items, gen 512" | tee "$SUMMARY"

step () { local label="$1"; shift; local t=$SECONDS
  echo "=== $label $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  for attempt in 1 2 3 4; do
    if "$@"; then echo "[ok]   $label $(( (SECONDS-t)/60 ))min" | tee -a "$SUMMARY"; return; fi
    echo "    attempt $attempt failed" | tee -a "$SUMMARY"
  done
  echo "[FAIL] $label" | tee -a "$SUMMARY"; }

step "fast-dllm mbpp"  bash scripts/run_eval_fastdllm_dream.sh  mbpp dual_cache_parallel
step "dllm-cache mbpp" bash scripts/run_eval_dllmcache_dream.sh mbpp dllm_cache
step "origin mbpp"     bash scripts/run_eval_dllmcache_dream.sh mbpp baseline
echo "done $(date +%F\ %T)" | tee -a "$SUMMARY"
