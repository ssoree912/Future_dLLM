#!/usr/bin/env bash
# llada-fastdllm only: its smoke failed in the main driver (a sys.path clash,
# since fixed) so the driver skipped it. llada-dllmcache is in the driver.
# Bracketed pattern so it cannot match the shell that launches it.
set -uo pipefail
cd /workspace/dllm/fdllm_oc
while pgrep -f "[r]un_oc_all.sh" > /dev/null; do sleep 300; done
export FUTURE_DLLM_LLADA_MODEL="$PWD/model/LLaDA-8B-Instruct"
export FUTURE_DLLM_MODEL="$PWD/model/Dream-v0-Instruct-7B"
SUMMARY="logs/oc/llada_rows_$(date +%Y%m%d_%H%M%S).txt"
for row in llada-fastdllm; do
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
