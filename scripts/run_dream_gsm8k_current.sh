#!/usr/bin/env bash
# Full Dream GSM8K evaluation with teacher-style current-block attention.
# Physical GPU 2 is exposed as the process-local cuda:0.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/opt/conda/envs/future-dllm/bin/python}"
if [ "$#" -eq 0 ]; then
  RATIOS=(0.5 0.1)
else
  RATIOS=("$@")
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
GPU_UUID="$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | awk -F, -v uuid="$GPU_UUID" '{gsub(/ /,"",$1); if ($1 == uuid) found=1} END {exit !found}'; then
  echo "physical GPU 2 already has a compute process; refusing to overlap" >&2
  exit 3
fi
nvidia-smi -i 2 --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

export FUTURE_DLLM_MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
export FUTURE_DLLM_MODEL_TAG="${FUTURE_DLLM_MODEL_TAG:-Dream-v0-Instruct-7B_current_teacher_rowmax_groupmean_perhead_entropy_t0.2_p0.95_s512_seed0}"
export EVICTION_METHOD=current
export CURRENT_ROW_REDUCE=max
export CURRENT_GROUP_REDUCE=mean
export CURRENT_PER_HEAD=True
export DREAM_ALG=entropy
export DREAM_TEMPERATURE=0.2
export DREAM_TOP_P=0.95
export DREAM_STEPS=512
export DREAM_SEED=0
export PY

for keep in "${RATIOS[@]}"; do
  echo "=== Dream GSM8K current-teacher keep=$keep started $(date -Is) ==="
  bash "$REPO/scripts/run_eval.sh" gsm8k "$keep"
  echo "=== Dream GSM8K current-teacher keep=$keep finished $(date -Is) ==="
done

if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | awk -F, -v uuid="$GPU_UUID" '{gsub(/ /,"",$1); if ($1 == uuid) found=1} END {exit !found}'; then
  echo "warning: a compute process remains on physical GPU 2" >&2
  exit 5
fi
echo "GPU 2 cleanup confirmed"
