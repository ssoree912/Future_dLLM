#!/usr/bin/env bash
# Sparse-dLLM baseline on LLaDA at keep_ratio 0.2, the generative suite.
#
# Same datasets and generation budgets as every other row (gen 32/64/128/256/512
# per Future_dLLM's table); only the eviction differs. Multiple choice is not
# here -- those go through OpenCompass.
#
# One failure does not stop the sweep, and a completed dataset is skipped on a
# rerun, so an interrupted run resumes where it stopped.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export FUTURE_DLLM_MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/LLaDA-8B-Instruct}"
export EVICTION_METHOD=sparse
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
KEEP="${KEEP:-0.2}"

DATASETS="${DATASETS:-gsm8k humaneval \
triviaqa 2wikimqa hotpotqa musique passage_retrieval_en passage_count \
trec multifieldqa_en lcc repobench-p \
samsum qasper narrativeqa \
gov_report multi_news qmsum \
math}"

SUMMARY="$REPO/logs/eval/llada_sparse_keep${KEEP}_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$SUMMARY")"
echo "keep=$KEEP eviction=sparse datasets=$DATASETS" | tee "$SUMMARY"

for ds in $DATASETS; do
  outdir="$REPO/results/$(basename "$FUTURE_DLLM_MODEL")/keep${KEEP}/$ds"
  if compgen -G "$outdir/${ds}_keep${KEEP}_sparse_*.json" > /dev/null; then
    echo "[skip] $ds" | tee -a "$SUMMARY"; continue
  fi
  started=$SECONDS
  echo "=== $ds $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  for attempt in 1 2 3 4; do
    if bash "$REPO/scripts/run_eval.sh" "$ds" "$KEEP"; then
      echo "[ok]   $ds $(( (SECONDS - started) / 60 ))min" | tee -a "$SUMMARY"; break
    fi
    echo "    attempt $attempt failed" | tee -a "$SUMMARY"
    [ "$attempt" = 4 ] && echo "[FAIL] $ds" | tee -a "$SUMMARY"
  done
done
echo "done" | tee -a "$SUMMARY"
