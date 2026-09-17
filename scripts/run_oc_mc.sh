#!/usr/bin/env bash
# OpenCompass multiple-choice sweep for Dream: Full / Sparse-dLLM / Ours.
#
# Separate from scripts/run_eval.sh (lm-eval) on purpose. Dream's attention is
# bidirectional, so lm-eval's multiple_choice loglikelihood is not defined for
# it; OpenCompass scores these tasks generatively, which is also how Sparse-dLLM
# reports them. Same model, same decoding, different harness.
#
#   scripts/run_oc_mc.sh smoke     # 2 items per dataset, student row only
#   scripts/run_oc_mc.sh           # GPQA + ARC-C + PIQA, all three rows
#   scripts/run_oc_mc.sh mmlu      # MMLU alone -- four fifths of the cost
#   scripts/run_oc_mc.sh llada     # the LLaDA rows (needs a LLaDA checkpoint)
#   scripts/run_oc_mc.sh llada-mmlu
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
if [[ "${1:-}" == *llada* ]]; then
  # The LLaDA wrapper reads its own pair of variables, so a Dream checkpoint
  # can never be handed to a LLaDA run by accident.
  export FUTURE_DLLM_LLADA_MODEL="${FUTURE_DLLM_LLADA_MODEL:?set FUTURE_DLLM_LLADA_MODEL}"
  export FUTURE_DLLM_LLADA_STUDENT="${FUTURE_DLLM_LLADA_STUDENT:-}"
else
  export FUTURE_DLLM_STUDENT="${FUTURE_DLLM_STUDENT:-}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
# The config does `from eval_oc.model import DreamFutureOC`, and each task runs
# in its own subprocess, so the repo has to be importable from the environment.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# opencompass.datasets pulls in nltk, which imports sqlite3; conda's
# libicui18n.so.78 wants CXXABI_1.3.15 and the system libstdc++ is older, so the
# import dies unless conda's own libstdc++ is found first.
export LD_LIBRARY_PATH="/opt/conda/envs/future-dllm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"


case "${1:-}" in
  smoke) CONFIG="eval_oc/configs/eval_dream_mc_smoke.py"; shift ;;
  mmlu)  CONFIG="eval_oc/configs/eval_dream_mmlu.py";     shift ;;
  llada) CONFIG="eval_oc/configs/eval_llada_mc.py";       shift ;;
  llada-smoke) CONFIG="eval_oc/configs/eval_llada_mc_smoke.py"; shift ;;
  llada-fastdllm) CONFIG="eval_oc/configs/eval_llada_fastdllm.py"; shift ;;
  llada-fastdllm-smoke) CONFIG="eval_oc/configs/eval_llada_fastdllm_smoke.py"; shift ;;
  llada-dllmcache) CONFIG="eval_oc/configs/eval_llada_dllmcache.py"; shift ;;
  llada-dllmcache-smoke) CONFIG="eval_oc/configs/eval_llada_dllmcache_smoke.py"; shift ;;
  dream-fastdllm) CONFIG="eval_oc/configs/eval_dream_fastdllm.py"; shift ;;
  dream-fastdllm-smoke) CONFIG="eval_oc/configs/eval_dream_fastdllm_smoke.py"; shift ;;
  dream-dllmcache) CONFIG="eval_oc/configs/eval_dream_dllmcache.py"; shift ;;
  dream-dllmcache-smoke) CONFIG="eval_oc/configs/eval_dream_dllmcache_smoke.py"; shift ;;
  llada-full) CONFIG="eval_oc/configs/eval_llada_full.py"; shift ;;
  llada-full-smoke) CONFIG="eval_oc/configs/eval_llada_full_smoke.py"; shift ;;
  dream-full) CONFIG="eval_oc/configs/eval_dream_full.py"; shift ;;
  dream-full-smoke) CONFIG="eval_oc/configs/eval_dream_full_smoke.py"; shift ;;
  llada-sparse-k02) CONFIG="eval_oc/configs/eval_llada_sparse_k02.py"; shift ;;
  llada-sparse-k02-smoke) CONFIG="eval_oc/configs/eval_llada_sparse_k02_smoke.py"; shift ;;
  llada-sparse-k01) CONFIG="eval_oc/configs/eval_llada_sparse_k01.py"; shift ;;
  llada-sparse-k01-smoke) CONFIG="eval_oc/configs/eval_llada_sparse_k01_smoke.py"; shift ;;
  dream-sparse-k01) CONFIG="eval_oc/configs/eval_dream_sparse_k01.py"; shift ;;
  dream-sparse-k01-smoke) CONFIG="eval_oc/configs/eval_dream_sparse_k01_smoke.py"; shift ;;
  dream-origin) CONFIG="eval_oc/configs/eval_dream_origin.py"; shift ;;
  dream-origin-smoke) CONFIG="eval_oc/configs/eval_dream_origin_smoke.py"; shift ;;
  cache-llada) CONFIG="eval_oc/configs/eval_llada_cache_methods.py"; shift ;;
  cache-llada-smoke) CONFIG="eval_oc/configs/eval_llada_cache_methods_smoke.py"; shift ;;
  cache-dream-smoke) CONFIG="eval_oc/configs/eval_dream_cache_methods_smoke.py"; shift ;;
  cache-dream) CONFIG="eval_oc/configs/eval_dream_cache_methods.py"; shift ;;
  llada-mmlu) CONFIG="eval_oc/configs/eval_llada_mmlu.py"; shift ;;
  *)     CONFIG="eval_oc/configs/eval_dream_mc.py" ;;
esac

mkdir -p logs/oc
LOG="logs/oc/$(basename "${CONFIG%.py}")_$(date +%Y%m%d_%H%M%S).log"

echo "config    $CONFIG"
echo "model     $FUTURE_DLLM_MODEL"
echo "student   ${FUTURE_DLLM_STUDENT:-${FUTURE_DLLM_LLADA_STUDENT:-(none)}}"
echo "gpu       $CUDA_VISIBLE_DEVICES"
echo "log       $LOG"

# --debug keeps tasks in-process and streams their output, so a failure shows
# its traceback instead of a swallowed exit code.
"$OC_PYTHON" -m opencompass.cli.main "$CONFIG" --debug "$@" 2>&1 | tee -a "$LOG"
