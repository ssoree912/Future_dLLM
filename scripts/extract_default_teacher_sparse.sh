#!/usr/bin/env bash
# Extract the default five-domain teacher labels through Sparse-dLLM's decode.
#
# Separate roots from the other runs on purpose: these labels describe blocks
# their loop produced (entropy reveal, temperature 0.2, their schedule), which
# is a different trajectory from the ones under teacher_dream_2048/. Mixing the
# two would train a scorer on states it never sees at deployment.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
DATA_ROOT="${FUTURE_DLLM_DATA:-$REPO/data}"
PROMPT_ROOT="${PROMPT_ROOT:-$REPO/artifacts/prompt_shards_dream_${MAX_SEQ_LEN}}"
TEACHER_ROOT="${TEACHER_ROOT:-$REPO/artifacts/teacher_sparse_${MAX_SEQ_LEN}}"
SEED="${SEED:-2025}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$REPO/logs/teacher_extract/extract_teacher_sparse_${RUN_TAG}.log}"

DATASETS=(math5s mbpp_full gov_report multi_news musique)
LIMITS=(500 371 150 100 500)

export FUTURE_DLLM_DATA="$DATA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$LOG_FILE")" "$TEACHER_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'teacher extraction via Sparse-dLLM decode\nmodel=%s\nprompts=%s\nteacher=%s\nmax_seq_len=%s\nseed=%s\ngpu=%s\nlog=%s\n\n' \
  "$MODEL" "$PROMPT_ROOT" "$TEACHER_ROOT" "$MAX_SEQ_LEN" "$SEED" "$CUDA_VISIBLE_DEVICES" "$LOG_FILE"

for index in "${!DATASETS[@]}"; do
  "$PY" "$REPO/teacher/extract_teacher_sparse.py" \
    --model "$MODEL" \
    --dataset "${DATASETS[$index]}" \
    --n-samples "${LIMITS[$index]}" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --seed "$SEED" \
    --shard-root "$PROMPT_ROOT" \
    --output-root "$TEACHER_ROOT"
done

echo "teacher extraction (sparse decode) complete"
