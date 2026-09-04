#!/usr/bin/env python
"""Extract LLaDA teacher labels.

Thin entry point: the label definition and the block loop live in
``extract_teacher``, shared with the Dream extractor so the two cannot
drift. This file only pins the family.

    python teacher/extract_teacher_llada.py \
        --model model/LLaDA-8B-Instruct --dataset math5s --n-samples 500

``--model`` is required and is checked against the checkpoint's config: a Dream
path here is refused rather than quietly labelled.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_teacher import run

if __name__ == "__main__":
    raise SystemExit(run("llada", __doc__))
