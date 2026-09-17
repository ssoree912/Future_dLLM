#!/usr/bin/env bash
# Run one lm-eval task from Future_dLLM's harness with Fast-dLLM v1's LLaDA decoding.
#
#   scripts/run_eval_fastdllm.sh <dataset> [method]
#
# method is one of Fast-dLLM's own eval_gsm8k.sh variants:
#
#   baseline                  no cache,     one token per step
#   prefix_cache              prefix cache, one token per step
#   parallel                  no cache,     confidence threshold
#   cache_parallel            prefix cache, confidence threshold
#   dual_cache_parallel       dual cache,   confidence threshold
#   parallel_factor           no cache,     dynamic (factor) threshold
#   cache_parallel_factor     prefix cache, dynamic (factor) threshold
#   dual_cache_parallel_factor dual cache,  dynamic (factor) threshold
#
# Examples:
#   scripts/run_eval_fastdllm.sh gsm8k dual_cache_parallel
#   LIMIT=20 scripts/run_eval_fastdllm.sh gsm8k baseline
#   GEN_LENGTH=512 scripts/run_eval_fastdllm.sh humaneval cache_parallel
#
# Env: LIMIT, GEN_LENGTH, STEPS, BLOCK_LENGTH, THRESHOLD, FACTOR, MAX_SEQ_LEN,
#      MAX_PROMPT_LEN, MC_NUM, MC_BATCH_SIZE, LOG_SAMPLES=1, CHAT_TEMPLATE,
#      CUDA_VISIBLE_DEVICES.
set -euo pipefail

DATASET="${1:?usage: run_eval_fastdllm.sh <dataset> [method]}"
METHOD="${2:-baseline}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/LLaDA-8B-Instruct}"
FASTDLLM="${FASTDLLM_LLADA:-$(cd "$REPO/.." && pwd)/Fast-dLLM/v1/llada}"
DATA_ROOT="${FUTURE_DLLM_DATA:-$REPO/data}"
LONGBENCH_DATA="${LONGBENCH_DATA:-$DATA_ROOT/longbench/data}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$REPO/logs/eval/fastdllm_${DATASET}_${METHOD}_${RUN_TAG}.log}"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

# Task, few-shot count and generation budget, taken from Future_dLLM's own table
# so the two harness runs are directly comparable.
LIKELIHOOD_TASK=0
UNSAFE_TASK=0
case "$DATASET" in
  gov_report|multi_news|qmsum)          TASK="longbench_$DATASET"; SHOTS=""; LEN=512 ;;
  samsum|qasper|narrativeqa)            TASK="longbench_$DATASET"; SHOTS=""; LEN=128 ;;
  trec|lcc|repobench-p|multifieldqa_en) TASK="longbench_$DATASET"; SHOTS=""; LEN=64 ;;
  triviaqa|2wikimqa|hotpotqa|musique|passage_retrieval_en|passage_count)
                                        TASK="longbench_$DATASET"; SHOTS=""; LEN=32 ;;
  gsm8k)      TASK=local_gsm8k;     SHOTS="--num_fewshot 5"; LEN=256 ;;
  math)       TASK=local_math;      SHOTS=""; LEN=256 ;;
  math500)    TASK=local_math500;   SHOTS=""; LEN=256 ;;
  humaneval)  TASK=local_humaneval; SHOTS=""; LEN=512; UNSAFE_TASK=1 ;;
  mmlu)       TASK="${MMLU_TASKS:-local_mc_mmlu}"; SHOTS="--num_fewshot 5";  LEN=0; LIKELIHOOD_TASK=1 ;;
  arc_c)      TASK=local_mc_arc_challenge;         SHOTS="--num_fewshot 25"; LEN=0; LIKELIHOOD_TASK=1 ;;
  piqa)       TASK=local_mc_piqa;                  SHOTS="";                 LEN=0; LIKELIHOOD_TASK=1 ;;
  gpqa)       TASK=local_mc_gpqa_main_n_shot;      SHOTS="--num_fewshot 5";  LEN=0; LIKELIHOOD_TASK=1 ;;
  *) echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac

# Cache variant and parallel schedule.
THRESHOLD_DEFAULT=0.9
FACTOR_DEFAULT=1.0
case "$METHOD" in
  baseline)                   USE_CACHE=False; DUAL=False; SCHEDULE=none ;;
  prefix_cache)               USE_CACHE=True;  DUAL=False; SCHEDULE=none ;;
  parallel)                   USE_CACHE=False; DUAL=False; SCHEDULE=threshold ;;
  cache_parallel)             USE_CACHE=True;  DUAL=False; SCHEDULE=threshold ;;
  dual_cache_parallel)        USE_CACHE=True;  DUAL=True;  SCHEDULE=threshold ;;
  parallel_factor)            USE_CACHE=False; DUAL=False; SCHEDULE=factor ;;
  cache_parallel_factor)      USE_CACHE=True;  DUAL=False; SCHEDULE=factor ;;
  dual_cache_parallel_factor) USE_CACHE=True;  DUAL=True;  SCHEDULE=factor ;;
  *) echo "unknown method: $METHOD" >&2; exit 1 ;;
esac

GEN_LENGTH="${GEN_LENGTH:-$LEN}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
# One denoising step per token. The threshold and factor schedules ignore the
# per-step quota and stop a block as soon as it is filled, so this is only an
# upper bound for them - but dual cache needs it, because its refinement loop is
# bounded by steps/num_blocks rather than by the remaining mask count.
STEPS="${STEPS:-$GEN_LENGTH}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

MODEL_NAME=LLaDA_fastdllm
ARGS="pretrained=$MODEL,fastdllm_path=$FASTDLLM,block_length=$BLOCK_LENGTH"
ARGS="$ARGS,gen_length=$GEN_LENGTH,steps=$STEPS,max_seq_len=$MAX_SEQ_LEN"
ARGS="$ARGS,use_cache=$USE_CACHE,dual_cache=$DUAL"
# Off by default so the prompt is exactly what Future_dLLM's own run_eval.sh
# feeds the model and the two sets of numbers stay comparable. CHAT_TEMPLATE=True
# switches to Fast-dLLM's own convention (one user turn around the whole few-shot
# prompt), which reproduces their published setup but makes the model answer in
# its own style, so gsm8k strict-match and math exact_match go to zero.
ARGS="$ARGS,chat_template=${CHAT_TEMPLATE:-False}"
case "$SCHEDULE" in
  threshold) ARGS="$ARGS,threshold=${THRESHOLD:-$THRESHOLD_DEFAULT}" ;;
  factor)    ARGS="$ARGS,factor=${FACTOR:-$FACTOR_DEFAULT}" ;;
esac
# Fast-dLLM's own HumanEval script passes add_bos_token=True; off by default here so
# the prompt matches the other rows. ADD_BOS=True switches it on.
if [ -n "${ADD_BOS:-}" ]; then
  ARGS="$ARGS,add_bos_token=$ADD_BOS"
fi
if [ -n "${MAX_PROMPT_LEN:-}" ]; then
  ARGS="$ARGS,max_prompt_len=$MAX_PROMPT_LEN"
fi
if [ "$LIKELIHOOD_TASK" -eq 1 ]; then
  # Multiple choice never generates, so the cache knobs above do nothing here.
  # 32 Monte-Carlo samples, matching Future_dLLM's NLL_SAMPLES default rather than
  # Fast-dLLM's 128 - same estimator, same budget, comparable numbers.
  ARGS="$ARGS,mc_num=${MC_NUM:-32},mc_batch_size=${MC_BATCH_SIZE:-4}"
fi

MODEL_TAG="${FUTURE_DLLM_MODEL_TAG:-$(basename "$MODEL")}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT="$REPO/results/fastdllm/${MODEL_TAG}/${DATASET}/${DATASET}_${METHOD}_len${GEN_LENGTH}_${STAMP}.json"
TMP="$REPO/results/.run_fastdllm_${DATASET}_${STAMP}"
TASKS_DIR="$TMP/tasks"
mkdir -p "$(dirname "$RESULT")" "$TASKS_DIR"

# lm-eval resolves the yamls' "!function ..." references next to the yaml file,
# so the task definitions are copied into one scratch directory with the data
# paths substituted - same as scripts/run_eval.sh.
cp "$REPO"/eval/tasks/metrics.py "$TASKS_DIR/"
cp "$REPO"/eval/tasks/local_*.py "$TASKS_DIR/"
for y in "$REPO"/eval/tasks/longbench/*.yaml; do
  sed "s|LONGBENCH_DATA_DIR|$LONGBENCH_DATA|" "$y" > "$TASKS_DIR/$(basename "$y")"
done
for y in "$REPO"/eval/tasks/local/*.yaml "$REPO"/eval/tasks/local_mc/*; do
  sed "s|DATA_DIR|$DATA_ROOT|" "$y" > "$TASKS_DIR/$(basename "$y")"
done

RESUME_KEY="$(printf '%s\n%s' "$ARGS" "${LIMIT:-all}" | md5sum | cut -c1-8)"
export FASTDLLM_RESUME="$REPO/results/.resume/fastdllm_${MODEL_TAG}_${DATASET}_${METHOD}_${RESUME_KEY}.jsonl"
mkdir -p "$(dirname "$FASTDLLM_RESUME")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# Every task reads local files, so nothing here needs the Hub. datasets still
# rewrites those parquet files as arrow the first time it reads them; pin that
# cache inside the repo so a run never writes to a shared or home cache.
export HF_HOME="$REPO/.hf_cache"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
# Fast-dLLM's get_transfer_index takes a float64 softmax over the whole sequence,
# so peak memory swings hard between blocks; this keeps fragmentation out of it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_ARGS=()
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARGS+=(--limit "$LIMIT")
fi
if [ "$LIKELIHOOD_TASK" -eq 1 ]; then
  # Generative tasks get the chat template inside the model wrapper, the way
  # Fast-dLLM applies it; only the likelihood tasks go through lm-eval's flag.
  EXTRA_ARGS+=(--apply_chat_template)
fi
if [ "${LOG_SAMPLES:-1}" != "0" ]; then
  # Per-sample generations next to the result, for inspection or for
  # Fast-dLLM's postprocess_code.py on HumanEval.
  EXTRA_ARGS+=(--log_samples)
fi
if [ "$UNSAFE_TASK" -eq 1 ]; then
  export HF_ALLOW_CODE_EVAL=1
  EXTRA_ARGS+=(--confirm_run_unsafe_code)
fi

echo "$DATASET method=$METHOD gen_length=$GEN_LENGTH steps=$STEPS block=$BLOCK_LENGTH"
echo "max_seq_len=$MAX_SEQ_LEN samples=${LIMIT:-all} gpu=$CUDA_VISIBLE_DEVICES -> $RESULT"
cd "$REPO"
"$PY" eval/run_fastdllm.py \
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
rm -f "$FASTDLLM_RESUME"
echo "wrote $RESULT"
