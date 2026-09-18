#!/usr/bin/env bash
# General-table generative sweep: 100 examples at keep ratios 0.1..0.9.
# MC rows are evaluated by the OpenCompass configs separately; this runner covers
# the generation-based GSM8K, MATH, and HumanEval rows with identical samples.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIMIT="${LIMIT:-100}"
CKPT="${CKPT:-$REPO/../Future_dLLM/artifacts/ckpts/per_head_epoch6/checkpoint-best}"
GPU="${CUDA_VISIBLE_DEVICES:-3}"
LOG="$REPO/logs/benchmark/general_generative_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
source /opt/conda/etc/profile.d/conda.sh
conda activate future-dllm
export CUDA_VISIBLE_DEVICES="$GPU" LIMIT
SAMPLE_ROOT="${BENCH_DATA:-$REPO/.bench_samples_${LIMIT}_seed20260917}"
if [ ! -d "$SAMPLE_ROOT" ]; then
  python "$REPO/scripts/prepare_benchmark_samples.py" --source "$REPO/../Future_dLLM/data" --output "$SAMPLE_ROOT" --limit "$LIMIT" >/dev/null
fi
export FUTURE_DLLM_DATA="$SAMPLE_ROOT"

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  for ds in gsm8k math humaneval; do
    echo "=== ours ds=$ds keep=$ratio $(date -Is) ===" | tee -a "$LOG"
    env FUTURE_DLLM_MODEL_TAG=ours_perhead_keep${ratio} \
      bash "$REPO/scripts/run_eval.sh" "$ds" "$ratio" "$CKPT" 2>&1 | tee -a "$LOG"
    echo "=== sparse ds=$ds keep=$ratio $(date -Is) ===" | tee -a "$LOG"
    env EVICTION_METHOD=sparse FUTURE_DLLM_MODEL_TAG=sparse_dllm_keep${ratio} \
      bash "$REPO/scripts/run_eval.sh" "$ds" "$ratio" 2>&1 | tee -a "$LOG"
  done
done
echo "DONE $(date -Is)" | tee -a "$LOG"
