#!/usr/bin/env bash
# Run every Future_dLLM eval dataset except MMLU through dLLM-Cache's decoding,
# one after another on a single GPU.
#
#   scripts/run_all_dllmcache.sh [method]
#   DATASETS="gsm8k humaneval" scripts/run_all_dllmcache.sh dllm_cache
#
# Each dataset is retried a few times: this box drops CUDA initialisation now and
# then, and run_eval_dllmcache.sh keeps a resume store, so a retry picks up where
# the crash left off instead of starting over.
#
# arc_c / piqa / gpqa are scored by diffusion likelihood, which never generates -
# the feature cache has nothing to reuse across a single forward, so they run under
# the "baseline" name whatever method is asked for.
set -uo pipefail

METHOD="${1:-dllm_cache}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETRIES="${RETRIES:-4}"

DEFAULT_DATASETS="gsm8k humaneval \
samsum qasper narrativeqa trec lcc repobench-p multifieldqa_en \
triviaqa 2wikimqa hotpotqa musique passage_retrieval_en passage_count \
gov_report multi_news qmsum \
math arc_c piqa gpqa"
DATASETS="${DATASETS:-$DEFAULT_DATASETS}"

SUMMARY="$REPO/logs/eval/dllmcache_all_${METHOD}_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$SUMMARY")"
echo "method=$METHOD datasets=$DATASETS" | tee "$SUMMARY"

for dataset in $DATASETS; do
  case "$dataset" in
    arc_c|piqa|gpqa|mmlu) run_method=baseline ;;
    *)                    run_method="$METHOD" ;;
  esac
  started=$(date +%s)
  status=failed
  for attempt in $(seq 1 "$RETRIES"); do
    echo "=== $dataset ($run_method) attempt $attempt/$RETRIES $(date +%H:%M:%S)" | tee -a "$SUMMARY"
    if bash "$REPO/scripts/run_eval_dllmcache.sh" "$dataset" "$run_method"; then
      status=ok
      break
    fi
    echo "    attempt $attempt failed, retrying" | tee -a "$SUMMARY"
  done
  echo "$dataset $run_method $status $(( ($(date +%s) - started) / 60 ))min" | tee -a "$SUMMARY"
done

echo "done -> $SUMMARY" | tee -a "$SUMMARY"
