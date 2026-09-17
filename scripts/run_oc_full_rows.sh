#!/usr/bin/env bash
# The no-eviction ceiling rows, both families, same OpenCompass path as ours.
set -uo pipefail
cd /workspace/dllm/fdllm_oc
export FUTURE_DLLM_LLADA_MODEL="$PWD/model/LLaDA-8B-Instruct"
export FUTURE_DLLM_MODEL="$PWD/model/Dream-v0-Instruct-7B"
SUMMARY="logs/oc/full_rows_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p logs/oc
for row in dream-full llada-full; do
  echo "=== $row smoke $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  if ! bash scripts/run_oc_mc.sh "${row}-smoke"; then
    echo "[SKIP] $row - smoke failed" | tee -a "$SUMMARY"; continue
  fi
  started=$SECONDS
  echo "=== $row full  $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  if bash scripts/run_oc_mc.sh "$row"; then
    echo "[ok]   $row $(( (SECONDS - started) / 60 ))min" | tee -a "$SUMMARY"
  else
    echo "[FAIL] $row $(( (SECONDS - started) / 60 ))min" | tee -a "$SUMMARY"
  fi
done
echo "done" | tee -a "$SUMMARY"
