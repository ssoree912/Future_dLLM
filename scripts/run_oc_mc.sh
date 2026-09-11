#!/usr/bin/env bash
# OpenCompass multiple-choice sweep for Dream: Full / Sparse-dLLM / Ours.
#
# Separate from scripts/run_eval.sh (lm-eval) on purpose. Dream's attention is
# bidirectional, so lm-eval's multiple_choice loglikelihood is not defined for
# it; OpenCompass scores these tasks generatively, which is also how Sparse-dLLM
# reports them. Same model, same decoding, different harness.
#
#   scripts/run_oc_mc.sh smoke     # 2 items per dataset, student row only
#   scripts/run_oc_mc.sh           # the full suite, all three rows
#
# Env: FUTURE_DLLM_MODEL, FUTURE_DLLM_STUDENT, CUDA_VISIBLE_DEVICES.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# OpenCompass lives in its own venv, created with --system-site-packages from
# future-dllm so it inherits the pinned torch/transformers. The tree there is
# vanilla 0.4.2; everything of ours is under eval_oc/.
OC_PYTHON="${OC_PYTHON:-/workspace/dllm/oc/ocenv/bin/python}"

export FUTURE_DLLM_MODEL="${FUTURE_DLLM_MODEL:-$REPO/model/Dream-v0-Instruct-7B}"
export FUTURE_DLLM_STUDENT="${FUTURE_DLLM_STUDENT:?set FUTURE_DLLM_STUDENT to a checkpoint-best directory}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# The config does `from eval_oc.model import DreamFutureOC`, and each task runs
# in its own subprocess, so the repo has to be importable from the environment.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == "smoke" ]]; then
  CONFIG="eval_oc/configs/eval_dream_mc_smoke.py"
  shift
else
  CONFIG="eval_oc/configs/eval_dream_mc.py"
fi

mkdir -p logs/oc
LOG="logs/oc/$(basename "${CONFIG%.py}")_$(date +%Y%m%d_%H%M%S).log"

echo "config    $CONFIG"
echo "model     $FUTURE_DLLM_MODEL"
echo "student   $FUTURE_DLLM_STUDENT"
echo "gpu       $CUDA_VISIBLE_DEVICES"
echo "log       $LOG"

# --debug keeps tasks in-process and streams their output, so a failure shows
# its traceback instead of a swallowed exit code.
"$OC_PYTHON" -m opencompass.cli.main "$CONFIG" --debug "$@" 2>&1 | tee -a "$LOG"
