#!/usr/bin/env bash
# Build prompts and extract the default five-domain teacher labels.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/LLaDA-8B-Instruct}"
DATA_ROOT="${FUTURE_DLLM_DATA:-$REPO/data}"
PROMPT_ROOT="${PROMPT_ROOT:-$REPO/artifacts/prompt_shards}"
TEACHER_ROOT="${TEACHER_ROOT:-$REPO/artifacts/teacher}"
MAX_SEQ_LEN=4096
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$REPO/logs/teacher_extract/extract_default_teacher_${RUN_TAG}.log}"

DATASETS=(math5s mbpp_full gov_report multi_news musique)
LIMITS=(500 371 150 100 500)

export FUTURE_DLLM_DATA="$DATA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$LOG_FILE")" "$PROMPT_ROOT" "$TEACHER_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'default teacher extraction\nmodel=%s\ndata=%s\nprompts=%s\nteacher=%s\nmax_seq_len=%s\nlog=%s\n' \
  "$MODEL" "$DATA_ROOT" "$PROMPT_ROOT" "$TEACHER_ROOT" "$MAX_SEQ_LEN" "$LOG_FILE"

for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$index]}"
  limit="${LIMITS[$index]}"
  "$PY" "$REPO/teacher/build_prompt_shards.py" \
    --dataset "$dataset" \
    --limit "$limit" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --model "$MODEL" \
    --out-root "$PROMPT_ROOT"
done

for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$index]}"
  limit="${LIMITS[$index]}"
  "$PY" "$REPO/teacher/extract_teacher.py" \
    --dataset "$dataset" \
    --n-samples "$limit" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --model "$MODEL" \
    --shard-root "$PROMPT_ROOT" \
    --output-root "$TEACHER_ROOT"
done

echo "default teacher extraction complete"
