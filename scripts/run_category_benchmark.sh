#!/usr/bin/env bash
# 100-sample throughput/memory/accuracy benchmark across representative task
# categories. All rows consume the same first N examples per task.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIMIT="${LIMIT:-100}"
KEEP="${KEEP:-0.1}"
CKPT="${CKPT:-$REPO/../Future_dLLM/artifacts/ckpts/per_head_epoch6/checkpoint-best}"
GPU="${CUDA_VISIBLE_DEVICES:-3}"
DATASETS=(gsm8k hotpotqa gov_report humaneval)
LOG="$REPO/logs/benchmark/categories_${KEEP}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

source /opt/conda/etc/profile.d/conda.sh
conda activate future-dllm
export CUDA_VISIBLE_DEVICES="$GPU"
export LIMIT

run_one() {
  local label="$1"; shift
  echo "=== $label $(date -Is) ===" | tee -a "$LOG"
  ( set +e; "$@"; rc=$?; echo "=== $label rc=$rc $(date -Is) ==="; exit "$rc" ) 2>&1 | tee -a "$LOG"
}

for ds in "${DATASETS[@]}"; do
  # Ours: per-head Student scorer.
  run_one "ours_${ds}" env FUTURE_DLLM_MODEL_TAG=ours_perhead_keep${KEEP} \
    bash "$REPO/scripts/run_eval.sh" "$ds" "$KEEP" "$CKPT"

  # Sparse-dLLM attention baseline through the same Future-dLLM harness.
  run_one "sparse_${ds}" env EVICTION_METHOD=sparse \
    FUTURE_DLLM_MODEL_TAG=sparse_dllm_keep${KEEP} \
    bash "$REPO/scripts/run_eval.sh" "$ds" "$KEEP"

  # Vanilla origin and Fast-dLLM accelerated decoding.
  run_one "origin_${ds}" env FUTURE_DLLM_MODEL_TAG=origin_no_cache \
    bash "$REPO/scripts/run_eval_fastdllm.sh" "$ds" baseline
  run_one "fastdllm_${ds}" env FUTURE_DLLM_MODEL_TAG=fastdllm \
    bash "$REPO/scripts/run_eval_fastdllm.sh" "$ds" dual_cache_parallel

  # dLLM-Cache published feature-cache configuration.
  run_one "dllmcache_${ds}" env FUTURE_DLLM_MODEL_TAG=dllmcache \
    bash "$REPO/scripts/run_eval_dllmcache.sh" "$ds" dllm_cache
done

echo "DONE $(date -Is)" | tee -a "$LOG"
