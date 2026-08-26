#!/usr/bin/env python
"""Restore the local LongBench dump to the official schema.

``data/longbench/`` holds two shapes. Five files are already the official
release -- ``input`` plus a raw ``context``. The other eleven are a repackaged
copy that bakes the prompt into the fields: ``context`` carries the instructions
and the passages, ``question`` the query, ``answer_prefix`` the closing cue.
Feeding that copy to the task files in ``eval/tasks/longbench/`` would render
the instruction twice, because those files hold the prompt themselves.

Concatenating the three fields reproduces the rendered prompt exactly, so the
official fields can be recovered by stripping the template back off:

    prompt = PREFIX + context + MIDDLE + input + SUFFIX

Every row is checked by re-rendering the template against the recovered fields
and comparing with the prompt the repackaged row produces. A row that does not
round-trip is an error, not a warning -- nothing is written unless all of them
do.

    python eval/tasks/prepare_longbench_data.py

Writes ``data/longbench/data/<task>.jsonl``, where the task files expect it.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

yaml.SafeLoader.add_constructor("!function", lambda l, n: l.construct_scalar(n))

HERE = Path(__file__).resolve().parent
TASKS = HERE / "longbench"
SRC = Path("/mnt/srv/home/dlpcg.325/dllm/data/longbench")
DST = SRC / "data"


def template_parts(doc_to_text: str) -> tuple[str, str, str, bool]:
    """Split a task template into the text around ``{{context}}``/``{{input}}``."""
    prefix, rest = doc_to_text.split("{{context}}", 1)
    has_input = "{{input}}" in rest
    if has_input:
        middle, suffix = rest.split("{{input}}", 1)
    else:
        middle, suffix = rest, ""
    return prefix, middle, suffix, has_input


def recover(row: dict, parts: tuple[str, str, str, bool]) -> dict:
    """Turn one repackaged row back into ``context`` + ``input``."""
    prefix, middle, suffix, has_input = parts
    prompt = row["context"] + row.get("question", "") + row.get("answer_prefix", "")

    if not prompt.startswith(prefix):
        raise ValueError("prompt does not start with the template prefix")
    if suffix and not prompt.endswith(suffix):
        raise ValueError("prompt does not end with the template suffix")
    core = prompt[len(prefix): len(prompt) - len(suffix) if suffix else None]

    if has_input:
        # `middle` is a long instruction block; the query is what follows the
        # last occurrence of it, so search from the right in case the passages
        # happen to quote it.
        cut = core.rfind(middle)
        if cut < 0:
            raise ValueError("template middle not found")
        context, text_input = core[:cut], core[cut + len(middle):]
    else:
        if not core.endswith(middle):
            raise ValueError("core does not end with the template middle")
        context, text_input = core[:len(core) - len(middle)] if middle else core, ""

    if prefix + context + middle + text_input + suffix != prompt:
        raise ValueError("round-trip mismatch")

    out = {k: v for k, v in row.items()
           if k not in {"question", "answer_prefix", "task", "max_new_tokens"}}
    out["context"] = context
    out["input"] = text_input
    return out


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    converted = copied = 0

    for task_yaml in sorted(TASKS.glob("*.yaml")):
        name = task_yaml.stem
        src = SRC / f"{name}.jsonl"
        if not src.is_file():
            print(f"  {name:<22} 원본 없음, 건너뜀")
            continue

        rows = [json.loads(line) for line in src.open(encoding="utf-8")]
        if "input" in rows[0]:
            shutil.copyfile(src, DST / f"{name}.jsonl")
            print(f"  {name:<22} 공식 스키마 그대로 복사  ({len(rows)} rows)")
            copied += 1
            continue

        parts = template_parts(yaml.safe_load(task_yaml.open())["doc_to_text"])
        out = []
        for i, row in enumerate(rows):
            try:
                out.append(recover(row, parts))
            except ValueError as exc:
                sys.exit(f"{name} row {i}: {exc}")

        with (DST / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in out:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {name:<22} 복원 완료, {len(out)} rows 전부 왕복 검증 통과")
        converted += 1

    print(f"\n{DST} 에 기록: 복원 {converted}개 · 복사 {copied}개")


if __name__ == "__main__":
    main()
