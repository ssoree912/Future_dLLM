#!/usr/bin/env bash
# Generation runs behind the origin-drift figure: one origin, then sparse-dLLM
# and the student at each retention budget.
#
# The x axis is prompt-KV retention, and keep_ratio is exactly that -- sink and
# recent are reserved inside the budget, not added to it (docs/eviction_method.md
# section 9), so the ratios below map onto the axis directly.
#
# The low end is where the arms separate; 0.8 is the "barely evicted" anchor and
# 0.05 is where the cache stops being able to hold the answer. Skipping the
# middle would hide the shape, so the grid is denser below 0.3 than above it.
#
# Datasets: gov_report is in the student's training mix and generates 512 tokens
# over an 11.6K-token prompt, which is the most drift resolution available here;
# qasper is held out and generates 128, so the two together separate "learned
# this domain" from "generalises".
#
# One GPU run at a time. Nothing here is parallel-safe: each run wants ~20 GiB.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CKPT="${CKPT:?set CKPT to the student checkpoint-best}"
DATASETS="${DATASETS:-qasper gov_report}"
RATIOS="${RATIOS:-0.5 0.1 0.05}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
# The generations live in the resume store and nowhere else, so every run here
# has to keep its own; run_eval.sh deletes it otherwise.
export KEEP_RESUME=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SUMMARY="$REPO/logs/eval/drift_sweep_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "$SUMMARY")"
echo "datasets=$DATASETS ratios=$RATIOS ckpt=$CKPT limit=${LIMIT:-all}" | tee "$SUMMARY"

run() {   # tag dataset keep [ckpt]
  local tag="$1" ds="$2" keep="$3" ckpt="${4:-}"
  # A finished run is one whose store is still on disk: the store is what the
  # table is built from, and its name carries the sample limit, so a scored
  # result from a different --limit is not this run. A store short of the
  # limit is a crash rather than a finished run -- it stays where it is, so the
  # rerun resumes from it instead of starting the documents over.
  local store
  store="$(compgen -G "$REPO/results/.resume/${tag}_${ds}_keep${keep}_*.jsonl" | head -1)"
  if [ -n "$store" ] && { [ -z "${LIMIT:-}" ] || [ "$(wc -l < "$store")" -ge "$LIMIT" ]; }; then
    echo "[skip] $ds $tag keep=$keep" | tee -a "$SUMMARY"; return
  fi
  local started=$SECONDS
  echo "=== $ds $tag keep=$keep $(date -Is)" | tee -a "$SUMMARY"
  for attempt in 1 2 3; do
    if env FUTURE_DLLM_MODEL_TAG="$tag" ${5:+EVICTION_METHOD=$5} \
         bash "$REPO/scripts/run_eval.sh" "$ds" "$keep" $ckpt; then
      echo "[ok]   $ds $tag keep=$keep $(( (SECONDS - started) / 60 ))min" | tee -a "$SUMMARY"
      return
    fi
    echo "    attempt $attempt failed" | tee -a "$SUMMARY"
    [ "$attempt" = 3 ] && echo "[FAIL] $ds $tag keep=$keep" | tee -a "$SUMMARY"
    sleep 60
  done
}

for ds in $DATASETS; do
  # The reference has to exist before the arms mean anything, so it goes first
  # and a failure here stops that dataset rather than producing arms with
  # nothing to be compared against.
  run drift_origin "$ds" 1.0 || true
  if ! compgen -G "$REPO/results/.resume/drift_origin_${ds}_keep1.0_*.jsonl" > /dev/null; then
    echo "[STOP] $ds: no origin run, skipping its arms" | tee -a "$SUMMARY"; continue
  fi
  for keep in $RATIOS; do
    run drift_sparse "$ds" "$keep" ""      sparse
    run drift_ours   "$ds" "$keep" "$CKPT"
  done
done
echo "done $(date -Is)" | tee -a "$SUMMARY"

cat <<EOF | tee -a "$SUMMARY"

Resume stores now hold the generations. Build the table with:

  for ds in $DATASETS; do
    python scripts/origin_drift.py --dataset \$ds \\
      --origin results/.resume/drift_origin_\${ds}_keep1.0_none_*.jsonl \\
      \$(for k in $RATIOS; do
          echo "--arm sparse:\$k:results/.resume/drift_sparse_\${ds}_keep\${k}_sparse_*.jsonl"
          echo "--arm ours:\$k:results/.resume/drift_ours_\${ds}_keep\${k}_*.jsonl"
        done) \\
      --out results/drift/\$ds
  done
EOF
