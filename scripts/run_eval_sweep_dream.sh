#!/usr/bin/env bash
# Run every generative eval task on Dream, at both keep ratios.
#
#   scripts/run_eval_sweep_dream.sh [checkpoint]
#
# Ordered so that stopping early still leaves the most useful rows finished:
# gsm8k first, then the three LongBench domains the scorer was trained on, then
# the shorter benchmarks, then the rest of LongBench, and `math` (5,000 items,
# roughly half the total budget) last.
#
# One failing task does not stop the sweep -- the failure is recorded and the
# next task starts. Completed (dataset, keep) pairs are skipped on a rerun, so
# an interrupted sweep resumes where it stopped.
#
# Multiple-choice tasks (mmlu / arc_c / piqa / gpqa) are deliberately excluded.
# They score by diffusion loglikelihood and currently return near-chance numbers
# through lm-eval (piqa 0.45 at keep_ratio=1.0, where scoring the same items
# directly gives 0.825). Until that is understood they would only produce
# invalid rows.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
CKPT="${1:?usage: run_eval_sweep_dream.sh <checkpoint>}"
CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"

export FUTURE_DLLM_MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

MODEL_TAG="$(basename "$FUTURE_DLLM_MODEL")"
METHOD="$(basename "$(dirname "$CKPT")")"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="$REPO/logs/eval/sweep_dream_${RUN_TAG}.log"
mkdir -p "$(dirname "$SWEEP_LOG")"
exec > >(tee -a "$SWEEP_LOG") 2>&1

# keep_ratio=1.0 is the no-eviction reference every 0.1 row is read against, so
# each dataset runs both back to back rather than doing all of one ratio first.
DATASETS=(
  gsm8k                                    # 5-shot reasoning, gen 256
  gov_report multi_news musique            # LongBench, the scorer's own domains
  math500 humaneval                        # short generative benchmarks
  samsum qasper narrativeqa                # LongBench gen 128
  trec lcc repobench-p multifieldqa_en     # LongBench gen 64
  triviaqa 2wikimqa hotpotqa               # LongBench gen 32
  passage_retrieval_en passage_count qmsum
  math                                     # 5,000 items -- longest, so last
)
KEEPS=(0.1 1.0)

total=$(( ${#DATASETS[@]} * ${#KEEPS[@]} ))
printf 'dream eval sweep\nmodel=%s\nckpt=%s\nmax_seq_len=%s\ngpu=%s\nruns=%d\nlog=%s\n\n' \
  "$FUTURE_DLLM_MODEL" "$CKPT" "$MAX_SEQ_LEN" "$CUDA_VISIBLE_DEVICES" "$total" "$SWEEP_LOG"

done_n=0; skip_n=0; fail_n=0; failed=()
for ds in "${DATASETS[@]}"; do
  for keep in "${KEEPS[@]}"; do
    # keep=1.0 needs no scorer; passing one anyway would be ignored, but leaving
    # it off keeps the results filename honest about what produced the row.
    if [ "$keep" = "1.0" ]; then args=("$ds" "$keep"); tag=none
    else                        args=("$ds" "$keep" "$CKPT"); tag="$METHOD"; fi

    # A result only counts as done if it covers the whole task. A LIMIT run
    # writes to the same path, and treating one as finished would silently
    # leave a 20-item smoke number standing in for a 1,319-item benchmark.
    outdir="$REPO/results/$MODEL_TAG/keep$keep/$ds"
    if compgen -G "$outdir/${ds}_keep${keep}_${tag}_*.json" > /dev/null \
       && "$PY" - "$outdir" "$ds" "$keep" "$tag" <<'PYEOF'
import glob, json, sys
outdir, ds, keep, tag = sys.argv[1:5]
full = [f for f in glob.glob(f"{outdir}/{ds}_keep{keep}_{tag}_*.json")
        if json.load(open(f)).get("config", {}).get("limit") is None]
sys.exit(0 if full else 1)
PYEOF
    then
      echo "[skip] $ds keep=$keep (already has a full result)"
      skip_n=$((skip_n + 1)); continue
    fi

    echo "=== [$((done_n + skip_n + fail_n + 1))/$total] $ds keep=$keep  $(date +%H:%M:%S) ==="
    started=$SECONDS
    if PY="$PY" "$REPO/scripts/run_eval.sh" "${args[@]}"; then
      echo "[ok]   $ds keep=$keep  $(( (SECONDS - started) / 60 ))min"
      done_n=$((done_n + 1))
    else
      echo "[FAIL] $ds keep=$keep  $(( (SECONDS - started) / 60 ))min"
      fail_n=$((fail_n + 1)); failed+=("$ds:$keep")
    fi
  done
done

echo
echo "sweep complete: $done_n ok, $skip_n skipped, $fail_n failed"
[ "$fail_n" -gt 0 ] && printf 'failed: %s\n' "${failed[*]}"
exit 0
