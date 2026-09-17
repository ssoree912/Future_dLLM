#!/usr/bin/env bash
# Run every Future_dLLM eval dataset except the multiple-choice ones through Fast-dLLM v1's decoding,
# one after another on a single GPU.
#
#   scripts/run_all_fastdllm_dream.sh [method]
#   DATASETS="gsm8k humaneval" scripts/run_all_fastdllm_dream.sh dual_cache_parallel
#
# Each dataset is retried a few times: this box drops CUDA initialisation now and
# then, and run_eval_fastdllm_dream.sh keeps a resume store, so a retry picks up where
# the crash left off instead of starting over.
#
# The multiple-choice tasks are excluded: this repo's Dream branch records that
# diffusion loglikelihood through lm-eval returns near-chance numbers for Dream
# (piqa 0.45 where scoring the same items directly gives 0.825).
set -uo pipefail

METHOD="${1:-dual_cache_parallel}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETRIES="${RETRIES:-4}"

DEFAULT_DATASETS="gsm8k humaneval \
triviaqa 2wikimqa hotpotqa musique passage_retrieval_en passage_count \
trec multifieldqa_en lcc repobench-p \
samsum qasper narrativeqa \
gov_report multi_news qmsum \
math"
DATASETS="${DATASETS:-$DEFAULT_DATASETS}"

SUMMARY="$REPO/logs/eval/fastdllm_dream_all_${METHOD}_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$SUMMARY")"
echo "method=$METHOD datasets=$DATASETS" | tee "$SUMMARY"

for dataset in $DATASETS; do
  run_method="$METHOD"
  started=$(date +%s)
  status=failed
  for attempt in $(seq 1 "$RETRIES"); do
    echo "=== $dataset ($run_method) attempt $attempt/$RETRIES $(date +%H:%M:%S)" | tee -a "$SUMMARY"
    if bash "$REPO/scripts/run_eval_fastdllm_dream.sh" "$dataset" "$run_method"; then
      status=ok
      break
    fi
    echo "    attempt $attempt failed, retrying" | tee -a "$SUMMARY"
  done
  echo "$dataset $run_method $status $(( ($(date +%s) - started) / 60 ))min" | tee -a "$SUMMARY"
done

echo "done -> $SUMMARY" | tee -a "$SUMMARY"
