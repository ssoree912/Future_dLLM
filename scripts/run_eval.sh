#!/usr/bin/env bash
# Run one lm-eval task with the LLaDA future-cache model.
#
#   scripts/run_eval.sh <dataset> <keep_ratio> [checkpoint]
#
# Examples:
#   scripts/run_eval.sh samsum 0.1 artifacts/ckpts/<run>/checkpoint-best
#   scripts/run_eval.sh gsm8k  1.0
#   LIMIT=200 scripts/run_eval.sh math 0.1 artifacts/ckpts/<run>/checkpoint-best
set -euo pipefail

DATASET="${1:?usage: run_eval.sh <dataset> <keep_ratio> [checkpoint]}"
KEEP="${2:?usage: run_eval.sh <dataset> <keep_ratio> [checkpoint]}"
CKPT="${3:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/LLaDA-8B-Instruct}"
DATA_ROOT="${FUTURE_DLLM_DATA:-$REPO/data}"
LONGBENCH_DATA="${LONGBENCH_DATA:-$DATA_ROOT/longbench/data}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$REPO/logs/eval/${DATASET}_keep${KEEP}_${RUN_TAG}.log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

LIKELIHOOD_TASK=0
UNSAFE_TASK=0
case "$DATASET" in
  samsum|trec|triviaqa|2wikimqa|hotpotqa|musique|qasper|narrativeqa|multifieldqa_en|\
  gov_report|qmsum|multi_news|lcc|repobench-p|passage_retrieval_en|passage_count)
              TASK="longbench_$DATASET"; SHOTS="" ;;
  mmlu)       TASK="${MMLU_TASKS:-local_mc_mmlu}";   SHOTS="--num_fewshot 5"; LIKELIHOOD_TASK=1 ;;
  arc_c)      TASK=local_mc_arc_challenge;           SHOTS="--num_fewshot 25"; LIKELIHOOD_TASK=1 ;;
  piqa)       TASK=local_mc_piqa;                    SHOTS=""; LIKELIHOOD_TASK=1 ;;
  gpqa)       TASK=local_mc_gpqa_main_n_shot;        SHOTS="--num_fewshot 5"; LIKELIHOOD_TASK=1 ;;
  gsm8k)      TASK=local_gsm8k;          SHOTS="--num_fewshot 5" ;;
  math)       TASK=local_math;           SHOTS="" ;;
  math500)    TASK=local_math500;        SHOTS="" ;;
  humaneval)  TASK=local_humaneval;      SHOTS=""; UNSAFE_TASK=1 ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

if [[ ! "$KEEP" =~ ^1([.]0+)?$ ]] && [ -z "$CKPT" ]; then
  echo "keep_ratio=$KEEP requires a student checkpoint" >&2
  exit 2
fi

MODEL_NAME=LLaDA_future
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
ARGS="pretrained=$MODEL,block_len=32,keep_ratio=$KEEP,max_seq_len=$MAX_SEQ_LEN"
if [ -n "${MAX_PROMPT_LEN:-}" ]; then
  ARGS="$ARGS,max_prompt_len=$MAX_PROMPT_LEN"
fi
if [ "$LIKELIHOOD_TASK" -eq 1 ]; then
  ARGS="$ARGS,diffusion_steps=${NLL_SAMPLES:-32}"
fi
METHOD=none
if [ -n "$CKPT" ]; then
  ARGS="$ARGS,student_path=$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
  METHOD=$(basename "$(dirname "$CKPT")")
fi
MODEL_TAG="${FUTURE_DLLM_MODEL_TAG:-$(basename "$MODEL")}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT="$REPO/results/${MODEL_TAG}/keep${KEEP}/${DATASET}/${DATASET}_keep${KEEP}_${METHOD}_${STAMP}.json"
TMP="$REPO/results/.run_${DATASET}_${STAMP}"
TASKS_DIR="$TMP/tasks"
mkdir -p "$(dirname "$RESULT")" "$TASKS_DIR"

cp "$REPO"/eval/tasks/metrics.py "$TASKS_DIR/"
cp "$REPO"/eval/tasks/local_*.py "$TASKS_DIR/"
for y in "$REPO"/eval/tasks/longbench/*.yaml; do
  sed "s|LONGBENCH_DATA_DIR|$LONGBENCH_DATA|" "$y" > "$TASKS_DIR/$(basename "$y")"
done
for y in "$REPO"/eval/tasks/local/*.yaml "$REPO"/eval/tasks/local_mc/*; do
  sed "s|DATA_DIR|$DATA_ROOT|" "$y" > "$TASKS_DIR/$(basename "$y")"
done

RESUME_KEY="$(printf '%s\n%s' "$ARGS" "${LIMIT:-all}" | md5sum | cut -c1-8)"
export FUTURE_DLLM_RESUME="$REPO/results/.resume/${MODEL_TAG}_${DATASET}_keep${KEEP}_${METHOD}_${RESUME_KEY}.jsonl"
mkdir -p "$(dirname "$FUTURE_DLLM_RESUME")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Every task reads local files, so nothing here needs the Hub. datasets still rewrites those
# parquet files as arrow the first time it reads them; pin that cache inside the repo so a run
# never writes to a shared or home-directory cache. Deleting it only costs one re-read.
export HF_HOME="$REPO/.hf_cache"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=()
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi
if [ "$LIKELIHOOD_TASK" -eq 1 ]; then
  EXTRA_ARGS+=(--apply_chat_template)
fi
if [ "$UNSAFE_TASK" -eq 1 ]; then
  export HF_ALLOW_CODE_EVAL=1
  EXTRA_ARGS+=(--confirm_run_unsafe_code)
fi

echo "$DATASET keep=$KEEP max_seq_len=$MAX_SEQ_LEN samples=${LIMIT:-all} -> $RESULT"
cd "$REPO"
"$PY" eval/run.py \
  --model "$MODEL_NAME" \
  --model_args "$ARGS" \
  --tasks "$TASK" ${SHOTS} \
  --include_path "$TASKS_DIR" \
  --batch_size 1 \
  "${LIMIT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --output_path "$TMP/out"

mapfile -d '' -t RESULT_FILES < <(find "$TMP/out" -name 'results_*.json' -print0)
if [ "${#RESULT_FILES[@]}" -ne 1 ]; then
  echo "expected one lm-eval result under $TMP/out, found ${#RESULT_FILES[@]}" >&2
  exit 1
fi
mv "${RESULT_FILES[0]}" "$RESULT"
rm -rf "$TMP"
rm -f "$FUTURE_DLLM_RESUME"
echo "wrote $RESULT"
