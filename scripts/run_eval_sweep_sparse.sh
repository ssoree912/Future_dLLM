#!/usr/bin/env bash
# Evaluate the Sparse-dLLM baseline and our scorer through Sparse-dLLM's code.
#
#   scripts/run_eval_sweep_sparse.sh <checkpoint>
#
# Three rows come out of this:
#
#   Full cache     keep_ratio=1.0, no eviction. The scorer is never reached, so
#                  the baseline and our run are bit-identical -- verified on
#                  five GSM8K prompts -- and one run serves both.
#   Sparse-dLLM    their attention score      (SELECTION=sparse_dllm_orig)
#   Ours           the trained scorer         (SELECTION=sparse_dllm_ours)
#
# Both rows execute the same files; only the ranking function differs. Result
# filenames carry the method, so the three never overwrite each other.
#
# Ordered gsm8k -> math -> longbench, so stopping early still leaves the most
# useful rows finished. `math` is 5,000 items and dominates the budget; it sits
# second because it was asked for there, not because it is cheap.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
CKPT="${1:?usage: run_eval_sweep_sparse.sh <checkpoint>}"
CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"

export FUTURE_DLLM_MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL_TAG="$(basename "$FUTURE_DLLM_MODEL")"
OURS_METHOD="$(basename "$(dirname "$CKPT")")"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="$REPO/logs/eval/sweep_sparse_${RUN_TAG}.log"
mkdir -p "$(dirname "$SWEEP_LOG")"
exec > >(tee -a "$SWEEP_LOG") 2>&1

LONGBENCH="gov_report multi_news musique qmsum samsum qasper narrativeqa trec
           lcc repobench-p multifieldqa_en triviaqa 2wikimqa hotpotqa
           passage_retrieval_en passage_count"
read -r -a DATASETS <<< "${DATASETS:-gsm8k math $LONGBENCH}"
read -r -a KEEPS <<< "${KEEPS:-0.1 0.5}"

printf 'sparse-dLLM eval sweep\nmodel=%s\nckpt=%s\nmax_seq_len=%s\ngpu=%s\nlog=%s\n\n' \
  "$FUTURE_DLLM_MODEL" "$CKPT" "$MAX_SEQ_LEN" "$CUDA_VISIBLE_DEVICES" "$SWEEP_LOG"

done_n=0; skip_n=0; fail_n=0; failed=()

run_one() {   # dataset keep selection method [ckpt]
  local ds=$1 keep=$2 sel=$3 method=$4 ckpt=${5:-}
  local outdir="$REPO/results/$MODEL_TAG/keep$keep/$ds"
  # Only a full run counts. A LIMIT smoke writes to the same path and would
  # otherwise stand in for the benchmark.
  if compgen -G "$outdir/${ds}_keep${keep}_${method}_*.json" > /dev/null \
     && "$PY" - "$outdir" "$ds" "$keep" "$method" <<'PYEOF'
import glob, json, sys
outdir, ds, keep, method = sys.argv[1:5]
full = [f for f in glob.glob(f"{outdir}/{ds}_keep{keep}_{method}_*.json")
        if json.load(open(f)).get("config", {}).get("limit") is None]
sys.exit(0 if full else 1)
PYEOF
  then
    echo "[skip] $ds keep=$keep $method"
    skip_n=$((skip_n + 1)); return
  fi
  echo "=== $ds keep=$keep $method  $(date +%H:%M:%S) ==="
  local started=$SECONDS
  if SELECTION="$sel" PY="$PY" "$REPO/scripts/run_eval.sh" "$ds" "$keep" ${ckpt:+"$ckpt"}; then
    echo "[ok]   $ds keep=$keep $method  $(( (SECONDS - started) / 60 ))min"
    done_n=$((done_n + 1))
  else
    echo "[FAIL] $ds keep=$keep $method  $(( (SECONDS - started) / 60 ))min"
    fail_n=$((fail_n + 1)); failed+=("$ds:$keep:$method")
  fi
}

for ds in "${DATASETS[@]}"; do
  # The shared no-eviction reference, once.
  run_one "$ds" 1.0 sparse_dllm_orig sparse_dllm_orig
  for keep in "${KEEPS[@]}"; do
    run_one "$ds" "$keep" sparse_dllm_orig sparse_dllm_orig
    run_one "$ds" "$keep" sparse_dllm_ours "$OURS_METHOD" "$CKPT"
  done
done

echo
echo "sweep complete: $done_n ok, $skip_n skipped, $fail_n failed"
[ "$fail_n" -gt 0 ] && printf 'failed: %s\n' "${failed[*]}"
exit 0
