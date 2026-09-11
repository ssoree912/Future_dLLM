#!/usr/bin/env bash
# Fetch the OpenCompass datasets that its own auto-download cannot resolve.
#
# OpenCompass downloads most sets on first use, but two in this suite fall
# through: `opencompass/piqa` has no entry in DATASETS_URL (the archive exists
# on their mirror, it is just not in the table), and GPQA is gated on the Hub so
# it has to come from an authenticated download. Fetching them here keeps the
# installed OpenCompass tree vanilla -- no edits to datasets_info.py.
#
#   scripts/fetch_oc_data.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
CACHE="${COMPASS_DATA_CACHE:-$HOME/.cache/opencompass}/data"
MIRROR="http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data"

mkdir -p "$CACHE"
if [[ -f "$CACHE/piqa/dev.jsonl" ]]; then
  echo "piqa   already present"
else
  echo "piqa   downloading"
  curl -fsSL "$MIRROR/piqa.zip" -o "$CACHE/piqa.zip"
  ( cd "$CACHE" && unzip -oq piqa.zip && rm -f piqa.zip )
fi

# GPQA's config reads ./data/gpqa/ relative to the working directory, so this
# one lives in the repo rather than the shared cache.
if [[ -f "$REPO/data/gpqa/gpqa_diamond.csv" ]]; then
  echo "gpqa   already present"
else
  echo "gpqa   downloading from the Hub (gated: needs an accepted licence)"
  mkdir -p "$REPO/data/gpqa"
  "${OC_PYTHON:-/workspace/dllm/oc/ocenv/bin/python}" - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
dst = os.path.join(os.environ["REPO"], "data", "gpqa")
for name in ("gpqa_diamond.csv", "gpqa_main.csv"):
    shutil.copy(hf_hub_download("Idavidrein/gpqa", name, repo_type="dataset"),
                os.path.join(dst, name))
PY
fi

echo "piqa   $(ls "$CACHE/piqa" 2>/dev/null | tr '\n' ' ')"
echo "gpqa   $(ls "$REPO/data/gpqa" 2>/dev/null | tr '\n' ' ')"
