#!/usr/bin/env bash
# Run one lm-eval task from Future_dLLM's harness with dLLM-Cache's LLaDA decoding.
#
#   scripts/run_eval_dllmcache.sh <dataset> [method]
#
#   dllm_cache   adaptive feature cache on (dLLM-Cache's method)
#   baseline     cache off - the uncached decode their own scripts compare against
#
# Examples:
#   scripts/run_eval_dllmcache.sh gsm8k dllm_cache
#   LIMIT=20 scripts/run_eval_dllmcache.sh gsm8k baseline
#
# Env: LIMIT, GEN_LENGTH, STEPS, BLOCK_LENGTH, PROMPT_INTERVAL, GEN_INTERVAL,
#      TRANSFER_RATIO, MAX_SEQ_LEN, MAX_PROMPT_LEN, MC_NUM, MC_BATCH_SIZE,
#      LOG_SAMPLES=1, CHAT_TEMPLATE, ADD_BOS, CUDA_VISIBLE_DEVICES.
set -euo pipefail

DATASET="${1:?usage: run_eval_dllmcache.sh <dataset> [method]}"
METHOD="${2:-dllm_cache}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/LLaDA-8B-Instruct}"
DLLMCACHE="${DLLMCACHE_ROOT:-$(cd "$REPO/.." && pwd)/dLLM-cache}"
DATA_ROOT="${FUTURE_DLLM_DATA:-$REPO/data}"
LONGBENCH_DATA="${LONGBENCH_DATA:-$DATA_ROOT/longbench/data}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$REPO/logs/eval/dllmcache_${DATASET}_${METHOD}_${RUN_TAG}.log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

# Task, few-shot count and generation budget come from Future_dLLM's own table so
# the row is comparable with the others. The cache intervals come from dLLM-Cache's
# scripts/run_LLaDA_*_Instruct.sh, which tunes them per task family.
LIKELIHOOD_TASK=0
UNSAFE_TASK=0
PROMPT_INT=50
GEN_INT=7
case "$DATASET" in
  gov_report|multi_news|qmsum)          TASK="longbench_$DATASET"; SHOTS=""; LEN=512; PROMPT_INT=100; GEN_INT=8 ;;
  samsum|qasper|narrativeqa)            TASK="longbench_$DATASET"; SHOTS=""; LEN=128; PROMPT_INT=100; GEN_INT=8 ;;
  trec|lcc|repobench-p|multifieldqa_en) TASK="longbench_$DATASET"; SHOTS=""; LEN=64;  PROMPT_INT=100; GEN_INT=8 ;;
  triviaqa|2wikimqa|hotpotqa|musique|passage_retrieval_en|passage_count)
                                        TASK="longbench_$DATASET"; SHOTS=""; LEN=32;  PROMPT_INT=100; GEN_INT=8 ;;
  gsm8k)      TASK=local_gsm8k;     SHOTS="--num_fewshot 5"; LEN=256; PROMPT_INT=50; GEN_INT=7 ;;
  math)       TASK=local_math;      SHOTS=""; LEN=256; PROMPT_INT=50; GEN_INT=1 ;;
  math500)    TASK=local_math500;   SHOTS=""; LEN=256; PROMPT_INT=50; GEN_INT=1 ;;
  humaneval)  TASK=local_humaneval; SHOTS=""; LEN=512; PROMPT_INT=25; GEN_INT=5; UNSAFE_TASK=1 ;;
  mmlu)       TASK="${MMLU_TASKS:-local_mc_mmlu}"; SHOTS="--num_fewshot 5";  LEN=0; LIKELIHOOD_TASK=1 ;;
  arc_c)      TASK=local_mc_arc_challenge;         SHOTS="--num_fewshot 25"; LEN=0; LIKELIHOOD_TASK=1 ;;
  piqa)       TASK=local_mc_piqa;                  SHOTS="";                 LEN=0; LIKELIHOOD_TASK=1 ;;
  gpqa)       TASK=local_mc_gpqa_main_n_shot;      SHOTS="--num_fewshot 5";  LEN=0; LIKELIHOOD_TASK=1 ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

case "$METHOD" in
  dllm_cache) FEATURE_CACHE=True ;;
  baseline)   FEATURE_CACHE=False ;;
  *) echo "unknown method: $METHOD (dllm_cache | baseline)" >&2; exit 1 ;;
esac

GEN_LENGTH="${GEN_LENGTH:-$LEN}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
# One denoising step per token: dLLM-Cache's loop is a fixed range(steps/num_blocks)
# with a per-step quota and no early exit, so this moves exactly one token a step.
STEPS="${STEPS:-$GEN_LENGTH}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

MODEL_NAME=LLaDA_dllmcache
ARGS="pretrained=$MODEL,dllmcache_path=$DLLMCACHE,block_length=$BLOCK_LENGTH"
ARGS="$ARGS,gen_length=$GEN_LENGTH,steps=$STEPS,max_seq_len=$MAX_SEQ_LEN"
if [ "$LIKELIHOOD_TASK" -eq 1 ]; then
  # The feature cache is state carried across denoising steps; a likelihood pass is
  # a single forward, so the hooks have nothing to reuse and their forward trips
  # over cache state that generate() would have set up. Multiple choice therefore
  # runs uncached, which is also what dLLM-Cache does (its loglikelihood raises
  # NotImplementedError).
  FEATURE_CACHE=False
fi
ARGS="$ARGS,is_feature_cache=$FEATURE_CACHE"
ARGS="$ARGS,prompt_interval_steps=${PROMPT_INTERVAL:-$PROMPT_INT}"
ARGS="$ARGS,gen_interval_steps=${GEN_INTERVAL:-$GEN_INT}"
ARGS="$ARGS,transfer_ratio=${TRANSFER_RATIO:-0.25}"
# Off by default so the prompt matches Future_dLLM's own run_eval.sh and the two
# sets of numbers stay comparable. dLLM-Cache's own scripts pass
# --apply_chat_template --fewshot_as_multiturn instead.
ARGS="$ARGS,chat_template=${CHAT_TEMPLATE:-False}"
# dLLM-Cache's own HumanEval script passes add_bos_token=True; off by default here so
# the prompt matches the other rows. ADD_BOS=True switches it on.
if [ -n "${ADD_BOS:-}" ]; then
  ARGS="$ARGS,add_bos_token=$ADD_BOS"
fi
if [ -n "${MAX_PROMPT_LEN:-}" ]; then
  ARGS="$ARGS,max_prompt_len=$MAX_PROMPT_LEN"
fi
if [ "$LIKELIHOOD_TASK" -eq 1 ]; then
  # Multiple choice never generates, so the feature cache does nothing here; the
  # estimator and its 32 samples match the Fast-dLLM row.
  ARGS="$ARGS,mc_num=${MC_NUM:-32},mc_batch_size=${MC_BATCH_SIZE:-4}"
fi

MODEL_TAG="${FUTURE_DLLM_MODEL_TAG:-$(basename "$MODEL")}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT="$REPO/results/dllmcache/${MODEL_TAG}/${DATASET}/${DATASET}_${METHOD}_len${GEN_LENGTH}_${STAMP}.json"
TMP="$REPO/results/.run_dllmcache_${DATASET}_${STAMP}"
TASKS_DIR="$TMP/tasks"
mkdir -p "$(dirname "$RESULT")" "$TASKS_DIR"

# lm-eval resolves the yamls' "!function ..." references next to the yaml file, so
# the task definitions are copied into one scratch directory with the data paths
# substituted - same as scripts/run_eval.sh.
cp "$REPO"/eval/tasks/metrics.py "$TASKS_DIR/"
cp "$REPO"/eval/tasks/local_*.py "$TASKS_DIR/"
for y in "$REPO"/eval/tasks/longbench/*.yaml; do
  sed "s|LONGBENCH_DATA_DIR|$LONGBENCH_DATA|" "$y" > "$TASKS_DIR/$(basename "$y")"
done
for y in "$REPO"/eval/tasks/local/*.yaml "$REPO"/eval/tasks/local_mc/*; do
  sed "s|DATA_DIR|$DATA_ROOT|" "$y" > "$TASKS_DIR/$(basename "$y")"
done

RESUME_KEY="$(printf '%s\n%s' "$ARGS" "${LIMIT:-all}" | md5sum | cut -c1-8)"
export DLLMCACHE_RESUME="$REPO/results/.resume/dllmcache_${MODEL_TAG}_${DATASET}_${METHOD}_${RESUME_KEY}.jsonl"
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
if [ "$LIKELIHOOD_TASK" -eq 1 ]; then
  EXTRA_ARGS+=(--apply_chat_template)
fi
if [ "${LOG_SAMPLES:-1}" != "0" ]; then
  EXTRA_ARGS+=(--log_samples)
fi
if [ "$UNSAFE_TASK" -eq 1 ]; then
  export HF_ALLOW_CODE_EVAL=1
  EXTRA_ARGS+=(--confirm_run_unsafe_code)
fi

echo "$DATASET method=$METHOD gen_length=$GEN_LENGTH steps=$STEPS block=$BLOCK_LENGTH"
echo "cache prompt/${PROMPT_INTERVAL:-$PROMPT_INT} gen/${GEN_INTERVAL:-$GEN_INT} transfer_ratio=${TRANSFER_RATIO:-0.25}"
echo "max_seq_len=$MAX_SEQ_LEN samples=${LIMIT:-all} gpu=$CUDA_VISIBLE_DEVICES -> $RESULT"
cd "$REPO"
"$PY" eval/run_dllmcache.py \
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
if [ "${LOG_SAMPLES:-1}" != "0" ]; then
  SAMPLE_DIR="${RESULT%.json}_samples"
  mkdir -p "$SAMPLE_DIR"
  find "$TMP/out" -name 'samples_*.jsonl' -exec mv {} "$SAMPLE_DIR/" \;
  echo "wrote samples to $SAMPLE_DIR"
fi
rm -rf "$TMP"
rm -f "$DLLMCACHE_RESUME"
echo "wrote $RESULT"
