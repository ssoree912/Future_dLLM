#!/usr/bin/env bash
# Build prompts and extract the default five-domain teacher labels.
#
# DATASETS / LIMITS / PER_HEAD / TEACHER_ROOT are overridable, so a per-head run
# on a subset does not need a second copy of this script:
#
#   PER_HEAD=1 LIMITS="250 185 75 50 250" \
#     TEACHER_ROOT=$PWD/artifacts/teacher_perhead \
#     scripts/extract_default_teacher.sh
#
#   PER_HEAD=1 LIMITS="250 185 75 50 250" LABEL_ROW_REDUCE=mean \
#     TEACHER_ROOT=$PWD/artifacts/teacher_perhead_mean \
#     scripts/extract_default_teacher.sh
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

read -r -a DATASETS <<< "${DATASETS:-math5s mbpp_full gov_report multi_news musique}"
read -r -a LIMITS <<< "${LIMITS:-500 371 150 100 500}"
# Head-averaged labels force one kept set per layer; --per-head keeps the axis
# so each head can keep its own. H times the storage, hence its own root.
PER_HEAD_ARGS=()
[ -n "${PER_HEAD:-}" ] && PER_HEAD_ARGS=(--per-head)
# Which reduction turns a block's rows into one number per candidate, and --
# on a GQA backend -- how the query heads sharing a KV entry fold. Same knobs
# as the Dream script; LLaDA is MHA (32 KV heads for 32 query heads), so the
# group setting has no group to fold and only the row one moves. Each writes
# its own teacher_kind, so the resume check refuses to mix them in one root.
REDUCE_ARGS=(--label-row-reduce "${LABEL_ROW_REDUCE:-max}"
             --label-group-reduce "${LABEL_GROUP_REDUCE:-max}")

export FUTURE_DLLM_DATA="$DATA_ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$(dirname "$LOG_FILE")" "$PROMPT_ROOT" "$TEACHER_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

printf 'default teacher extraction\nmodel=%s\ndata=%s\nprompts=%s\nteacher=%s\nmax_seq_len=%s\nper_head=%s\nreduce=row:%s group:%s\ndatasets=%s\nlimits=%s\nlog=%s\n' \
  "$MODEL" "$DATA_ROOT" "$PROMPT_ROOT" "$TEACHER_ROOT" "$MAX_SEQ_LEN" \
  "${PER_HEAD:-0}" "${LABEL_ROW_REDUCE:-max}" "${LABEL_GROUP_REDUCE:-max}" \
  "${DATASETS[*]}" "${LIMITS[*]}" "$LOG_FILE"

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
  "$PY" "$REPO/teacher/extract_teacher_llada.py" \
    --dataset "$dataset" \
    --n-samples "$limit" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --model "$MODEL" \
    --shard-root "$PROMPT_ROOT" \
    --output-root "$TEACHER_ROOT" \
    "${PER_HEAD_ARGS[@]}" \
    "${REDUCE_ARGS[@]}"
done

echo "default teacher extraction complete"
