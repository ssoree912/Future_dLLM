#!/usr/bin/env bash
# Smoke then full, one row at a time, cheapest row first.
#
# Each row is its own config because --debug keeps every loaded model resident;
# two rows in one config put two checkpoints on one card. The smoke for a row
# gates its full run, so a broken wrapper costs minutes instead of half a day.
set -uo pipefail
cd /workspace/dllm/fdllm_oc

export FUTURE_DLLM_LLADA_MODEL="$PWD/model/LLaDA-8B-Instruct"
export FUTURE_DLLM_MODEL="$PWD/model/Dream-v0-Instruct-7B"

SUMMARY="logs/oc/run_all_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p logs/oc
echo "start $(date +%F\ %T)" | tee "$SUMMARY"

# Fast-dLLM rows first: ~3 s/item against dLLM-Cache's ~14 s/item.
for row in llada-fastdllm dream-fastdllm llada-dllmcache dream-dllmcache; do
  echo "=== $row smoke $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  if ! bash scripts/run_oc_mc.sh "${row}-smoke"; then
    echo "[SKIP] $row - smoke failed" | tee -a "$SUMMARY"; continue
  fi
  echo "=== $row full  $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  started=$SECONDS
  if bash scripts/run_oc_mc.sh "$row"; then
    echo "[ok]   $row $(( (SECONDS - started) / 60 ))min" | tee -a "$SUMMARY"
  else
    echo "[FAIL] $row $(( (SECONDS - started) / 60 ))min" | tee -a "$SUMMARY"
  fi
done
echo "done $(date +%F\ %T)" | tee -a "$SUMMARY"
