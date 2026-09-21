#!/usr/bin/env bash
# Run one lm-eval task from Future_dLLM's harness with dLLM-Cache's Dream decoding.
#
#   scripts/run_eval_dllmcache_dream.sh <dataset> [method]
#
#   dllm_cache   adaptive feature cache on
#   baseline     cache off
#
# Generation budgets follow Future_dLLM's own table; total sequence length is 2048,
# which is what the repo's Dream branch (scripts/run_eval_sweep_dream.sh) uses.
# Multiple-choice tasks are not supported: that branch records that Dream's
# diffusion loglikelihood returns near-chance numbers through lm-eval.
#
# Env: LIMIT, GEN_LENGTH, STEPS, MAX_SEQ_LEN, MAX_PROMPT_LEN, LOG_SAMPLES=0,
#      CHAT_TEMPLATE, ADD_BOS, CUDA_VISIBLE_DEVICES.
set -euo pipefail

DATASET="${1:?usage: run_eval_dllmcache_dream.sh <dataset> [method]}"
METHOD="${2:-dllm_cache}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
DATA_ROOT="${FUTURE_DLLM_DATA:-$REPO/data}"
LONGBENCH_DATA="${LONGBENCH_DATA:-$DATA_ROOT/longbench/data}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$REPO/logs/eval/dllmcache_dream_${DATASET}_${METHOD}_${RUN_TAG}.log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

UNSAFE_TASK=0
CHAT_TASK=0
PROMPT_INT=25
GEN_INT=2
case "$DATASET" in
  gov_report|multi_news|qmsum)          TASK="longbench_$DATASET"; SHOTS=""; LEN=512; PROMPT_INT=100; GEN_INT=8 ;;
  samsum|qasper|narrativeqa)            TASK="longbench_$DATASET"; SHOTS=""; LEN=128; PROMPT_INT=100; GEN_INT=8 ;;
  trec|lcc|repobench-p|multifieldqa_en) TASK="longbench_$DATASET"; SHOTS=""; LEN=64;  PROMPT_INT=100; GEN_INT=8 ;;
  triviaqa|2wikimqa|hotpotqa|musique|passage_retrieval_en|passage_count)
                                        TASK="longbench_$DATASET"; SHOTS=""; LEN=32;  PROMPT_INT=100; GEN_INT=8 ;;
  gsm8k)      TASK=local_gsm8k;     SHOTS="--num_fewshot 5"; LEN=256; PROMPT_INT=25; GEN_INT=2 ;;
  math)       TASK=local_math;      SHOTS=""; LEN=256; PROMPT_INT=50; GEN_INT=1 ;;
  math500)    TASK=local_math500;   SHOTS=""; LEN=256; PROMPT_INT=50; GEN_INT=1 ;;
  humaneval)  TASK=local_humaneval; SHOTS=""; LEN=512; PROMPT_INT=50; GEN_INT=1; UNSAFE_TASK=1 ;;
  mbpp)      TASK=local_mbpp;     SHOTS="--num_fewshot 3"; LEN=512; PROMPT_INT=10; GEN_INT=8; UNSAFE_TASK=1; CHAT_TASK=1 ;;
  mmlu|arc_c|piqa|gpqa)
    echo "$DATASET is multiple choice; Dream's diffusion loglikelihood is not usable here" >&2
    exit 1 ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

case "$METHOD" in
  dllm_cache) FEATURE_CACHE=True ;;
  baseline)   FEATURE_CACHE=False ;;
  *) echo "unknown method: $METHOD (dllm_cache | baseline)" >&2; exit 1 ;;
esac

GEN_LENGTH="${GEN_LENGTH:-$LEN}"
STEPS="${STEPS:-$GEN_LENGTH}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"

ARGS="pretrained=$MODEL,max_new_tokens=$GEN_LENGTH,diffusion_steps=$STEPS"
ARGS="$ARGS,max_seq_len=$MAX_SEQ_LEN,is_feature_cache=$FEATURE_CACHE"
ARGS="$ARGS,prompt_interval_steps=${PROMPT_INTERVAL:-$PROMPT_INT}"
ARGS="$ARGS,gen_interval_steps=${GEN_INTERVAL:-$GEN_INT}"
ARGS="$ARGS,transfer_ratio=${TRANSFER_RATIO:-0.25}"
# dLLM-Cache's own Dream scripts sample at 0.2/0.95, unlike their LLaDA ones which
# leave generate() at greedy. That is part of the method, not of the measurement
# setup, so it stays at the author's value even though the task yaml says
# do_sample: false. Greedy here made the model commit an eos at position 0 and
# repeat that deterministically - 96% empty answers on multi_news.
ARGS="$ARGS,temperature=${TEMPERATURE:-0.2},top_p=${TOP_P:-0.95}"
ARGS="$ARGS,chat_template=${CHAT_TEMPLATE:-False},add_bos_token=${ADD_BOS:-True}"
if [ -n "${MAX_PROMPT_LEN:-}" ]; then
  ARGS="$ARGS,max_prompt_len=$MAX_PROMPT_LEN"
fi

MODEL_TAG="${FUTURE_DLLM_MODEL_TAG:-$(basename "$MODEL")}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT="$REPO/results/dllmcache/${MODEL_TAG}/${DATASET}/${DATASET}_${METHOD}_len${GEN_LENGTH}_${STAMP}.json"
TMP="$REPO/results/.run_dllmcache_dream_${DATASET}_${STAMP}"
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
export DLLMCACHE_RESUME="$REPO/results/.resume/dllmcache_dream_${MODEL_TAG}_${DATASET}_${METHOD}_${RESUME_KEY}.jsonl"
mkdir -p "$(dirname "$DLLMCACHE_RESUME")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_HOME="$REPO/.hf_cache"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_ARGS=()
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi
if [ "${LOG_SAMPLES:-1}" != "0" ]; then
  EXTRA_ARGS+=(--log_samples)
fi
if [ "$CHAT_TASK" -eq 1 ]; then
  # dLLM-cache runs MBPP as `--tasks mbpp --num_fewshot 3 --apply_chat_template`,
  # so this one task takes the template even though the rest of the suite does not.
  EXTRA_ARGS+=(--apply_chat_template)
fi
if [ "$UNSAFE_TASK" -eq 1 ]; then
  export HF_ALLOW_CODE_EVAL=1
  EXTRA_ARGS+=(--confirm_run_unsafe_code)
fi

echo "$DATASET method=$METHOD gen_length=$GEN_LENGTH steps=$STEPS max_seq_len=$MAX_SEQ_LEN"
echo "samples=${LIMIT:-all} gpu=$CUDA_VISIBLE_DEVICES -> $RESULT"
cd "$REPO"
"$PY" eval/run_dllmcache_dream.py \
  --model Dream_dllmcache \
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
if [ "${LOG_SAMPLES:-1}" != "0" ]; then
  SAMPLE_DIR="${RESULT%.json}_samples"
  mkdir -p "$SAMPLE_DIR"
  find "$TMP/out" -name 'samples_*.jsonl' -exec mv {} "$SAMPLE_DIR/" \;
fi
rm -rf "$TMP"
rm -f "$DLLMCACHE_RESUME"
echo "wrote $RESULT"
