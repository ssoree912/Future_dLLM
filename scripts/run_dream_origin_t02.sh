#!/usr/bin/env bash
# Dream-v0-Instruct-7B as it ships, at Dream's own recommended sampling.
#
# temperature 0.2 / top_p 0.95 / alg entropy - the combination in Dream's README
# usage example, and what dLLM-Cache and Sparse-dLLM both use. The greedy run
# this replaces scored GSM8K 40.18 against 71.65 for the same decoder with the
# cache on at 0.2, because greedy commits an eos at position 0 and repeats it.
#
# General table only. MATH last: without a cache it is ~40 GPU-hours on its own.
set -uo pipefail
REPO=/workspace/dllm/Future_dLLM
WT=/workspace/dllm/fdllm_oc
source activate future-dllm
python -c "import torch" || { echo "torch missing - wrong env"; exit 1; }

SUMMARY="$REPO/logs/eval/dream_origin_t02_$(date +%Y%m%d_%H%M%S).txt"
echo "dream origin, temperature=0.2 top_p=0.95" | tee "$SUMMARY"
step () { local label="$1"; shift; local t=$SECONDS
  echo "=== $label $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  if "$@"; then echo "[ok]   $label $(( (SECONDS-t)/60 ))min" | tee -a "$SUMMARY"
  else          echo "[FAIL] $label $(( (SECONDS-t)/60 ))min" | tee -a "$SUMMARY"; fi; }

cd "$REPO"
export TEMPERATURE=0.2 TOP_P=0.95
for ds in ${DATASETS:-gsm8k humaneval}; do
  step "dream-origin-t0.2 $ds" bash scripts/run_eval_dllmcache_dream.sh "$ds" baseline
done

cd "$WT"
export FUTURE_DLLM_MODEL="$WT/model/Dream-v0-Instruct-7B"
step "dream-origin-t0.2 mc smoke" bash scripts/run_oc_mc.sh dream-origin-smoke
step "dream-origin-t0.2 mc"       bash scripts/run_oc_mc.sh dream-origin

cd "$REPO"
export TEMPERATURE=0.2 TOP_P=0.95
step "dream-origin-t0.2 math" bash scripts/run_eval_dllmcache_dream.sh math baseline
echo "done $(date +%F\ %T)" | tee -a "$SUMMARY"
