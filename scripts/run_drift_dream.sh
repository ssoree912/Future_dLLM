#!/usr/bin/env bash
# Generation runs behind the Dream current-vs-oracle drift table.
#
# Three arms per task: the un-evicted reference, teacher-style current-block
# attention, and the oracle that evicts with the decoding block's own completed
# attention. current and oracle share their reductions (row=max, group=mean,
# per head), so the only thing separating them is which block's attention the
# score is read from -- the one being decoded now, or the one after it settles.
#
# The decoder matches scripts/run_dream_gsm8k_current.sh exactly, so these runs
# sit beside the current-attention rows already being produced there.
#
# One GPU run at a time; the oracle decodes every block twice and costs about
# double the others.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATASETS="${DATASETS:-gsm8k math humaneval mbpp qasper musique samsum}"
RATIOS="${RATIOS:-0.5 0.1}"
export LIMIT="${LIMIT:-50}"
export FUTURE_DLLM_MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
export DREAM_ALG="${DREAM_ALG:-entropy}"
export DREAM_TEMPERATURE="${DREAM_TEMPERATURE:-0.2}"
export DREAM_TOP_P="${DREAM_TOP_P:-0.95}"
export DREAM_STEPS="${DREAM_STEPS:-512}"
export DREAM_SEED="${DREAM_SEED:-0}"
export CURRENT_ROW_REDUCE="${CURRENT_ROW_REDUCE:-max}"
export CURRENT_GROUP_REDUCE="${CURRENT_GROUP_REDUCE:-mean}"
export CURRENT_PER_HEAD="${CURRENT_PER_HEAD:-True}"
export ORACLE_ROW_REDUCE="${ORACLE_ROW_REDUCE:-max}"
export ORACLE_GROUP_REDUCE="${ORACLE_GROUP_REDUCE:-mean}"
# The generations live in the resume store and nowhere else; run_eval.sh drops
# it otherwise and scripts/origin_drift.py has nothing to join on.
export KEEP_RESUME=1
# This box has one card. run_eval.sh now defaults to physical GPU 2, which the
# workspace policy reserves elsewhere -- set it explicitly rather than inherit.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SUMMARY="$REPO/logs/eval/drift_dream_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$SUMMARY")"
echo "datasets=$DATASETS ratios=$RATIOS limit=$LIMIT steps=$DREAM_STEPS" | tee "$SUMMARY"

run() {   # tag dataset keep method
  local tag="$1" ds="$2" keep="$3" method="$4"
  # A store short of the limit is a crash, not a finished run: leave it where it
  # is so the rerun resumes from it rather than starting the documents over.
  local store
  store="$(compgen -G "$REPO/results/.resume/${tag}_${ds}_keep${keep}_*.jsonl" | head -1)"
  if [ -n "$store" ] && [ "$(wc -l < "$store")" -ge "$LIMIT" ]; then
    echo "[skip] $ds $tag keep=$keep" | tee -a "$SUMMARY"; return 0
  fi
  local started=$SECONDS
  echo "=== $ds $tag keep=$keep $(date -Is)" | tee -a "$SUMMARY"
  for attempt in 1 2 3; do
    if env FUTURE_DLLM_MODEL_TAG="$tag" EVICTION_METHOD="$method" \
         bash "$REPO/scripts/run_eval.sh" "$ds" "$keep"; then
      echo "[ok]   $ds $tag keep=$keep $(( (SECONDS - started) / 60 ))min" | tee -a "$SUMMARY"
      return 0
    fi
    echo "    attempt $attempt failed" | tee -a "$SUMMARY"
    [ "$attempt" = 3 ] && echo "[FAIL] $ds $tag keep=$keep" | tee -a "$SUMMARY"
    sleep 60
  done
  return 1
}

for ds in $DATASETS; do
  # Without the reference the arms mean nothing, so a failure here skips the
  # task rather than producing arms with nothing to compare against.
  run drift_dream_origin "$ds" 1.0 student
  if ! compgen -G "$REPO/results/.resume/drift_dream_origin_${ds}_keep1.0_*.jsonl" > /dev/null; then
    echo "[STOP] $ds: no origin run, skipping its arms" | tee -a "$SUMMARY"; continue
  fi
  for keep in $RATIOS; do
    run drift_dream_current "$ds" "$keep" current
    run drift_dream_oracle  "$ds" "$keep" oracle
  done
done
echo "done $(date -Is)" | tee -a "$SUMMARY"
