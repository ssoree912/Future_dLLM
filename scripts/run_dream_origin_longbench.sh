#!/usr/bin/env bash
# Dream-v0-Instruct-7B as it ships: LongBench, at Dream's own sampling.
#
# temperature 0.2 / top_p 0.95 / alg entropy, no cache (is_feature_cache=False
# registers no hooks, so this is the checkpoint's own diffusion_generate).
# Generation budgets follow this repo's table, gen 32 through 512, so the row
# sits beside the cached ones.
#
# Ordered cheapest first: the gen-32 sets take minutes each, the gen-512
# summarisation three take hours, so stopping early still leaves most of the
# row filled.
set -uo pipefail
cd /workspace/dllm/Future_dLLM
source activate future-dllm
python -c "import torch" || { echo "torch missing - wrong env"; exit 1; }

SUMMARY="logs/eval/dream_origin_longbench_$(date +%Y%m%d_%H%M%S).txt"
export TEMPERATURE=0.2 TOP_P=0.95
echo "dream origin longbench, temperature=$TEMPERATURE top_p=$TOP_P" | tee "$SUMMARY"

for ds in ${DATASETS:-triviaqa 2wikimqa hotpotqa musique passage_retrieval_en \
          passage_count trec multifieldqa_en lcc repobench-p samsum qasper \
          narrativeqa gov_report multi_news qmsum}; do
  t=$SECONDS
  echo "=== $ds $(date +%H:%M:%S)" | tee -a "$SUMMARY"
  for attempt in 1 2 3 4; do
    if bash scripts/run_eval_dllmcache_dream.sh "$ds" baseline; then
      echo "[ok]   $ds $(( (SECONDS-t)/60 ))min" | tee -a "$SUMMARY"; break
    fi
    echo "    attempt $attempt failed" | tee -a "$SUMMARY"
    [ "$attempt" = 4 ] && echo "[FAIL] $ds" | tee -a "$SUMMARY"
  done
done
echo "done $(date +%F\ %T)" | tee -a "$SUMMARY"
