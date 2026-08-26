#!/usr/bin/env bash
# Train the default student, using every natural teacher block.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/LLaDA-8B-Instruct}"
TEACHER_ROOT="${TEACHER_ROOT:-$REPO/artifacts/teacher}"
EPOCHS="${EPOCHS:-15}"
LR="${LR:-2e-4}"
SEED="${SEED:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
PROJ_DIM="${PROJ_DIM:-256}"
MLP_DIM="${MLP_DIM:-512}"
PAIRS="${PAIRS:-4096}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-default_5ds_500-371-1600-150-800_e${EPOCHS}_lr${LR}_blk_gov16_${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-$REPO/logs/train_${RUN_NAME}.log}"

ROOTS=(
  "$TEACHER_ROOT/math5s"
  "$TEACHER_ROOT/mbpp_full"
  "$TEACHER_ROOT/musique"
  "$TEACHER_ROOT/gov_report"
  "$TEACHER_ROOT/repobench_p"
)
TEACHER_ROOTS="$(IFS=,; echo "${ROOTS[*]}")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'default student training\nmodel=%s\nteacher=%s\nrun=%s\nlog=%s\n' \
  "$MODEL" "$TEACHER_ROOT" "$RUN_NAME" "$LOG_FILE"

"$PY" "$REPO/student/train_student.py" \
  --model "$MODEL" \
  --teacher-root "$TEACHER_ROOTS" \
  --max-shards "500,371,1600,150,800" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --seed "$SEED" \
  --val-ratio "$VAL_RATIO" \
  --proj-dim "$PROJ_DIM" \
  --mlp-dim "$MLP_DIM" \
  --pairs "$PAIRS" \
  --name "$RUN_NAME"

echo "default student training complete"
