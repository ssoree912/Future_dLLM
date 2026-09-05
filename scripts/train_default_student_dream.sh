#!/usr/bin/env bash
# Train the default student on Dream teacher labels, using every natural block.
#
# --model has to be the same checkpoint the labels came from: the trainer
# replays the selection-time forward to recover the hidden states the scorer
# reads, so a different backend would train on states deployment never sees.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
# Dream runs at 2048 total, not 4096. The checkpoint's config advertises
# max_position_embeddings=131072, but that is inherited from the Qwen2.5-7B
# weights Dream was initialised from, not a trained diffusion context.
# Sparse-dLLM evaluates Dream at max_seq_len=2048 (4096 only for humaneval and
# longbench), so 2048 is what the baseline comparison is against.
# Generation lengths stay per-dataset, so the prompt cap is 2048 - gen_length.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
TEACHER_ROOT="${TEACHER_ROOT:-$REPO/artifacts/teacher_dream_${MAX_SEQ_LEN}}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
SEED="${SEED:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
PROJ_DIM="${PROJ_DIM:-256}"
MLP_DIM="${MLP_DIM:-512}"
PAIRS="${PAIRS:-4096}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-dream_5ds_500-371-150-100-500_e${EPOCHS}_lr${LR}_len${MAX_SEQ_LEN}_${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-$REPO/logs/train/train_${RUN_NAME}.log}"

ROOTS=(
  "$TEACHER_ROOT/math5s"
  "$TEACHER_ROOT/mbpp_full"
  "$TEACHER_ROOT/gov_report"
  "$TEACHER_ROOT/multi_news"
  "$TEACHER_ROOT/musique"
)
TEACHER_ROOTS="$(IFS=,; echo "${ROOTS[*]}")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'default student training (dream)\nmodel=%s\nteacher=%s\nmax_seq_len=%s\ngpu=%s\nrun=%s\nlog=%s\n' \
  "$MODEL" "$TEACHER_ROOT" "$MAX_SEQ_LEN" "$CUDA_VISIBLE_DEVICES" "$RUN_NAME" "$LOG_FILE"

"$PY" "$REPO/student/train_student.py" \
  --model "$MODEL" \
  --teacher-root "$TEACHER_ROOTS" \
  --max-shards "500,371,150,100,500" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --seed "$SEED" \
  --val-ratio "$VAL_RATIO" \
  --proj-dim "$PROJ_DIM" \
  --mlp-dim "$MLP_DIM" \
  --pairs "$PAIRS" \
  --block-length "$BLOCK_LENGTH" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --name "$RUN_NAME"

echo "default student training (dream) complete"
