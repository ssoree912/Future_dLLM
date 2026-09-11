#!/usr/bin/env bash
# Start the MMLU sweep once the GPQA/ARC-C/PIQA sweep has finished.
#
# Both want the same GPU, and running them together roughly halves each one's
# throughput without finishing either sooner. Waiting on the process rather
# than on a fixed delay means a sweep that dies early does not leave the GPU
# idle for hours.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

while pgrep -f "eval_oc/configs/eval_dream_mc.py" >/dev/null; do sleep 120; done
echo "$(date -Is) multiple-choice sweep finished; starting MMLU"
exec ./scripts/run_oc_mc.sh mmlu
