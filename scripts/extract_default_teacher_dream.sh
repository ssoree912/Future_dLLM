#!/usr/bin/env bash
# Build prompts and extract the default five-domain teacher labels, on Dream.
#
# Separate artifact roots from the LLaDA run on purpose. Dream tokenises with a
# Qwen2 chat template, so its prompt shards have different ids and lengths for
# the same sample, and the resume check in build_prompt_shards only compares
# lengths -- pointed at the LLaDA root it would silently accept LLaDA shards.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
source "$REPO/scripts/dream_decoding_env.sh"
# Dream runs at 2048 total, not 4096. The checkpoint's config advertises
# max_position_embeddings=131072, but that is inherited from the Qwen2.5-7B
# weights Dream was initialised from, not a trained diffusion context.
# Sparse-dLLM evaluates Dream at max_seq_len=2048 (4096 only for humaneval and
# longbench), so 2048 is what the baseline comparison is against.
# Generation lengths stay per-dataset, so the prompt cap is 2048 - gen_length.
#
# DATASETS / LIMITS / PER_HEAD / TEACHER_ROOT are overridable, so a per-head run
# on a subset does not need a second copy of this script:
#
#   PER_HEAD=1 LIMITS="250 185 75 50 250" \
#     TEACHER_ROOT=/workspace/hd/data/ask-dllm/teacher/teacher_dream_perhead \
#     scripts/extract_default_teacher_dream.sh
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
DATA_ROOT="${FUTURE_DLLM_DATA:-$REPO/data}"
PROMPT_ROOT="${PROMPT_ROOT:-$REPO/artifacts/prompt_shards_dream_${MAX_SEQ_LEN}}"
TEACHER_ROOT="${TEACHER_ROOT:-$REPO/artifacts/teacher_dream_${MAX_SEQ_LEN}_${DREAM_DECODER_TAG}}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$REPO/logs/teacher_extract/extract_default_teacher_dream_${RUN_TAG}.log}"

read -r -a DATASETS <<< "${DATASETS:-math5s mbpp_full gov_report multi_news musique}"
read -r -a LIMITS <<< "${LIMITS:-500 371 150 100 500}"
# Head-averaged labels force one kept set per layer; --per-head keeps the axis
# so each head can keep its own. On Dream that axis is the KV head axis (4, not
# the 28 query heads): the cache holds one entry per KV head, so that is the
# finest granularity eviction can act on. 4x the storage, hence its own root.
PER_HEAD_ARGS=()
[ -n "${PER_HEAD:-}" ] && PER_HEAD_ARGS=(--per-head)

export FUTURE_DLLM_DATA="$DATA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$LOG_FILE")" "$PROMPT_ROOT" "$TEACHER_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'default teacher extraction (dream)\nmodel=%s\ndata=%s\nprompts=%s\nteacher=%s\nmax_seq_len=%s\nper_head=%s\ndatasets=%s\nlimits=%s\ngpu=%s\nlog=%s\n' \
  "$MODEL" "$DATA_ROOT" "$PROMPT_ROOT" "$TEACHER_ROOT" "$MAX_SEQ_LEN" \
  "${PER_HEAD:-0}" "${DATASETS[*]}" "${LIMITS[*]}" \
  "$CUDA_VISIBLE_DEVICES" "$LOG_FILE"

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
  "$PY" "$REPO/teacher/extract_teacher_dream.py" \
    --dataset "$dataset" \
    --n-samples "$limit" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --model "$MODEL" \
    --shard-root "$PROMPT_ROOT" \
    --output-root "$TEACHER_ROOT" \
    --seed "$DREAM_SEED" "${DREAM_ARGS[@]}" \
    "${PER_HEAD_ARGS[@]}"
done

echo "default teacher extraction (dream) complete"
