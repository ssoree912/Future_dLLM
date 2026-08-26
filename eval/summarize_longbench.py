#!/usr/bin/env python
"""Collect the per-task lm-eval results under a run directory into one table.

LongBench reports one number per task -- ``score``, the task's own metric scaled
to 0-100 -- and an unweighted mean over tasks. lm-eval writes each task's score
into its own ``results_*.json``; this reads them back and prints the table.

    python eval/summarize_longbench.py <run-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ORDER = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
         "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
         "samsum", "passage_count", "passage_retrieval_en", "lcc",
         "repobench-p"]


def latest_results(task_dir: Path) -> dict | None:
    files = sorted(task_dir.rglob("results_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    rows, missing = [], []
    for name in ORDER:
        payload = latest_results(root / name) if (root / name).is_dir() else None
        if payload is None:
            missing.append(name)
            continue
        res = payload["results"][f"longbench_{name}"]
        n = payload.get("n-samples", {}).get(f"longbench_{name}", {}).get("effective")
        other = next((k.split(",")[0] for k in res
                      if k.endswith(",none") and not k.startswith("score")), "")
        rows.append((name, res["score,none"], other, res.get(f"{other},none"), n))

    width = max((len(r[0]) for r in rows), default=4)
    print(f"{'task':<{width}}  {'score':>7}  {'n':>5}  metric")
    for name, score, other, value, n in rows:
        extra = f"{other}={value:.4f}" if value is not None else ""
        print(f"{name:<{width}}  {score:>7.2f}  {str(n):>5}  {extra}")
    if rows:
        mean = sum(r[1] for r in rows) / len(rows)
        print(f"{'-' * (width + 18)}")
        print(f"{'mean':<{width}}  {mean:>7.2f}  over {len(rows)} task(s)")
    if missing:
        print(f"\nno results yet: {', '.join(missing)}")


if __name__ == "__main__":
    main()
