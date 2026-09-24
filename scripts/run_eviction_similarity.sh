#!/usr/bin/env bash
# Diagnose full-cache vs future-score vs current-attention eviction on one task.
#
#   scripts/run_eviction_similarity.sh <dataset> <keep_ratio> <checkpoint>
#
# Env: LIMIT, LOW_SIM_THRESHOLD=0.70, MIN_SIM_GAIN=0.01,
#      DECISION_METRIC=mean, PY.  Every GPU subprocess is isolated to physical
#      GPU 2; inside the process it is addressed as cuda:0.
set -euo pipefail

DATASET="${1:?usage: run_eviction_similarity.sh <dataset> <keep_ratio> <checkpoint>}"
KEEP="${2:?usage: run_eviction_similarity.sh <dataset> <keep_ratio> <checkpoint>}"
CKPT="${3:?usage: run_eviction_similarity.sh <dataset> <keep_ratio> <checkpoint>}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
if [[ "$(basename "$MODEL")" != *Dream* && "$(basename "$MODEL")" != *dream* ]]; then
  echo "this diagnostic currently targets Dream; got model $MODEL" >&2
  exit 2
fi
if [ ! -d "$CKPT" ]; then
  echo "student checkpoint not found: $CKPT" >&2
  exit 2
fi

# A Dream student is only valid under the decoding schedule that produced its
# teacher labels.  Read that schedule before running the full-cache arm too, so
# all three outputs are generated with the same step count.  An explicit,
# mismatched DREAM_STEPS is rejected early instead of wasting the first run.
DECODING_META="$CKPT/decoding.json"
if [ ! -f "$DECODING_META" ]; then
  echo "Dream checkpoint is missing decoding metadata: $DECODING_META" >&2
  exit 2
fi
CHECKPOINT_STEPS="$("$PY" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["steps"]))' "$DECODING_META")"
if [ -n "${DREAM_STEPS:-}" ] && [ "$DREAM_STEPS" != "$CHECKPOINT_STEPS" ]; then
  echo "DREAM_STEPS=$DREAM_STEPS does not match checkpoint steps=$CHECKPOINT_STEPS" >&2
  exit 2
fi
export DREAM_STEPS="$CHECKPOINT_STEPS"

# The workspace policy reserves physical GPU 2. Do not allow an inherited
# device list to silently redirect a run to GPU 0/1 or to multiple GPUs.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
GPU_UUID="$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
    | awk -F, -v uuid="$GPU_UUID" '{gsub(/ /,"",$1); if ($1 == uuid) found=1} END {exit !found}'; then
  echo "physical GPU 2 already has a compute process; refusing to overlap workloads" >&2
  exit 3
fi

check_gpu_cleanup() {
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
      | awk -F, -v uuid="$GPU_UUID" '{gsub(/ /,"",$1); if ($1 == uuid) found=1} END {exit !found}'; then
    echo "warning: a compute process remains on physical GPU 2" >&2
  else
    echo "GPU 2 cleanup confirmed"
  fi
  return 0
}
trap check_gpu_cleanup EXIT
nvidia-smi -i 2 --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_TAG="eviction_similarity_${DATASET}_${STAMP}"
FULL_TAG="${BASE_TAG}_full"
FUTURE_TAG="${BASE_TAG}_future"
CURRENT_TAG="${BASE_TAG}_current"

run_one() {
  local label="$1" tag="$2" ratio="$3" method="$4" checkpoint="${5:-}"
  echo "=== $label: dataset=$DATASET keep=$ratio method=$method ==="
  env CUDA_VISIBLE_DEVICES=2 LOG_SAMPLES=1 FUTURE_DLLM_MODEL="$MODEL" \
    FUTURE_DLLM_MODEL_TAG="$tag" EVICTION_METHOD="$method" \
    bash "$REPO/scripts/run_eval.sh" "$DATASET" "$ratio" "$checkpoint"
}

latest_samples() {
  local tag="$1" ratio="$2"
  find "$REPO/results/$tag/keep$ratio/$DATASET" -type f -name 'samples_*.jsonl' \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
}

# keep=1.0 is the same Future-dLLM decoder with no entries removed.  It is the
# controlled reference: schedule, prompt, random seed and model are identical,
# so the measured delta isolates eviction rather than a decoder change.
run_one full-cache "$FULL_TAG" 1.0 sparse
run_one future "$FUTURE_TAG" "$KEEP" student "$CKPT"
run_one current "$CURRENT_TAG" "$KEEP" current

FULL_SAMPLES="$(latest_samples "$FULL_TAG" 1.0)"
FUTURE_SAMPLES="$(latest_samples "$FUTURE_TAG" "$KEEP")"
CURRENT_SAMPLES="$(latest_samples "$CURRENT_TAG" "$KEEP")"
for pair in "full:$FULL_SAMPLES" "future:$FUTURE_SAMPLES" "current:$CURRENT_SAMPLES"; do
  if [ -z "${pair#*:}" ] || [ ! -f "${pair#*:}" ]; then
    echo "missing ${pair%%:*} sample file" >&2
    exit 4
  fi
done

OUT="$REPO/results/eviction_similarity/$BASE_TAG/keep$KEEP"
"$PY" "$REPO/scripts/compare_eviction_similarity.py" \
  --full "$FULL_SAMPLES" --future "$FUTURE_SAMPLES" --current "$CURRENT_SAMPLES" \
  --low-threshold "${LOW_SIM_THRESHOLD:-0.70}" \
  --min-gain "${MIN_SIM_GAIN:-0.01}" \
  --decision-metric "${DECISION_METRIC:-mean}" \
  --out "$OUT"

echo "diagnostic -> $OUT"
