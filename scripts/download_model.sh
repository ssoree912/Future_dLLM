#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
MODEL_REPO="${MODEL_REPO:-GSAI-ML/LLaDA-8B-Instruct}"
MODEL_DIR="${MODEL_DIR:-$REPO/model/LLaDA-8B-Instruct}"
export MODEL_REPO MODEL_DIR

mkdir -p "$MODEL_DIR"
"$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    local_dir=os.environ["MODEL_DIR"],
    token=os.environ.get("HF_TOKEN") or None,
)
print(f"model downloaded to {os.environ['MODEL_DIR']}")
PY
