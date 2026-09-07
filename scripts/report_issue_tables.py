#!/usr/bin/env python
"""Render the issue #5 tables from whatever is finished under results/.

Regenerating both tables from the result files beats patching the posted HTML:
the numbers can only ever be what a full run actually wrote, a rerun overwrites
rather than accumulates, and there is no string surgery to get wrong on the
fortieth update.

    python scripts/report_issue_tables.py            # print both tables
    python scripts/report_issue_tables.py --post     # ... and update issue #5

A result only counts when lm-eval recorded no ``limit``; LIMIT smoke runs write
to the same path and must never stand in for a benchmark.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GH_REPO = "ssoree912/Future_dLLM"
ISSUE = 5
MODEL_TAG = "Dream-v0-Instruct-7B"

HEADER_NOTE = (
    "**총 길이 2048 기준.** Dream 의 `config.json` 은 `max_position_embeddings=131072` 을 "
    "광고하지만 이는 Qwen2.5-7B 가중치에서 상속된 값이고 학습된 diffusion 컨텍스트가 아니다. "
    "Sparse-dLLM 도 이 벤치들에 `max_seq_len=2048` 을 쓴다 (humaneval / longbench 만 4096)."
)

# (column label, results/ dataset dir, lm-eval task name)
MAIN = [("MMLU", "mmlu", "local_mc_mmlu"), ("ARC-C", "arc_c", "local_mc_arc_challenge"),
        ("PIQA", "piqa", "local_mc_piqa"), ("GPQA", "gpqa", "local_mc_gpqa_main_n_shot"),
        ("GSM8K", "gsm8k", "local_gsm8k"), ("MATH", "math", "local_math"),
        ("HEval", "humaneval", "local_humaneval")]

LONGBENCH_GROUPS = [
    ("Single-Doc QA", [("NrtvQA", "narrativeqa"), ("Qasper", "qasper"), ("MF-en", "multifieldqa_en")]),
    ("Multi-Doc QA", [("HotpotQA", "hotpotqa"), ("2WikiMQA", "2wikimqa"), ("MuSiQue", "musique")]),
    ("Summarization", [("GovReport", "gov_report"), ("QMSum", "qmsum"), ("MultiNews", "multi_news")]),
    ("Few-shot", [("TREC", "trec"), ("TriviaQA", "triviaqa"), ("SAMSum", "samsum")]),
    ("Synthetic", [("PCount", "passage_count"), ("PRe", "passage_retrieval_en")]),
    ("Code", [("LCC", "lcc"), ("RB-P", "repobench-p")]),
]

# lm-eval reports one of these per task; the first present is the headline
# number. math_verify comes before exact_match because the MATH tasks report
# both and exact_match alone marks a symbolically equal answer wrong (math500:
# 23.4 vs 39.6 on the same run).
METRICS = ("score,none", "math_verify,none", "exact_match,flexible-extract",
           "pass@1,create_test", "exact_match,none", "acc_norm,none", "acc,none")


def read_results() -> dict[tuple[str, str], float]:
    """{(dataset, keep): metric} for every completed unlimited run."""
    out: dict[tuple[str, str], float] = {}
    for path in sorted(glob.glob(str(REPO_ROOT / "results" / MODEL_TAG / "keep*" / "*" / "*.json"))):
        data = json.loads(Path(path).read_text())
        if data.get("config", {}).get("limit") is not None:
            continue                      # a LIMIT smoke, not a benchmark
        keep = re.search(r"keep([\d.]+)", path).group(1)
        dataset = Path(path).parent.name
        for task_metrics in data["results"].values():
            for metric in METRICS:
                if metric in task_metrics:
                    out[(dataset, keep)] = float(task_metrics[metric]) * 100
                    break
    return out


# Not in the sweep, so "대기" would promise a run that is not queued. The
# multiple-choice four are held back pending the loglikelihood-harness question.
NOT_QUEUED = {"mmlu", "arc_c", "piqa", "gpqa"}


def cell(results, dataset, keep, running: set[tuple[str, str]]) -> str:
    if (dataset, keep) in results:
        return f"{results[(dataset, keep)]:.2f}"
    if dataset in NOT_QUEUED:
        return "—"
    return "실행 중" if (dataset, keep) in running else "대기"


def main_table(results, running) -> str:
    # Only the rows this repo produces carry numbers; the other methods are
    # placeholders kept so the table matches issue #3's shape.
    rows = {"1.0": ["Dream-v0-Instruct-7B", "dLLM-Cache", "Fast-dLLM", "Full cache"],
            "0.5": ["Oracle", "H<sub>2</sub>O", "Sparse-dLLM", "Ours"],
            "0.1": ["Oracle", "H<sub>2</sub>O", "Sparse-dLLM", "Ours"]}
    ours = {"1.0": "Full cache", "0.5": "Ours", "0.1": "Ours"}
    L = ["<table>", "  <thead>", "    <tr>", "      <th>Method</th>"]
    L += [f"      <th>{label}</th>" for label, _, _ in MAIN]
    L += ["      <th><strong>Avg.</strong></th>", "    </tr>", "  </thead>", "  <tbody>"]
    for keep, methods in rows.items():
        L.append(f'    <tr><th colspan="{len(MAIN) + 2}"><em>Keep ratio = {keep}</em></th></tr>')
        for method in methods:
            if method == ours[keep] and keep in ("1.0", "0.1"):
                vals = [cell(results, ds, keep, running) for _, ds, _ in MAIN]
            else:
                vals = ["—"] * len(MAIN)
            L.append(f"    <tr><td>{method}</td>" + "".join(f"<td>{v}</td>" for v in vals)
                     + "<td>—</td></tr>")
    L += ["  </tbody>", "</table>"]
    return "\n".join(L)


def longbench_table(results, running) -> str:
    tasks = [t for _, group in LONGBENCH_GROUPS for t in group]
    L = ["<table>", "  <thead>", "    <tr>", '      <th rowspan="2">Method</th>']
    L += [f'      <th colspan="{len(g)}">{name}</th>' for name, g in LONGBENCH_GROUPS]
    L += ['      <th rowspan="2"><strong>Avg.</strong></th>', "    </tr>", "    <tr>"]
    L += [f"      <th>{label}</th>" for label, _ in tasks]
    L += ["    </tr>", "  </thead>", "  <tbody>"]
    rows = {"1.0": ["Dream-v0-Instruct-7B", "dLLM-Cache", "Fast-dLLM", "Full cache"],
            "0.5": ["Oracle", "H<sub>2</sub>O", "Sparse-dLLM", "Ours"],
            "0.1": ["Oracle", "H<sub>2</sub>O", "Sparse-dLLM", "Ours"]}
    ours = {"1.0": "Full cache", "0.5": "Ours", "0.1": "Ours"}
    span = len(tasks) + 2
    for keep, methods in rows.items():
        L.append(f'    <tr><th colspan="{span}"><em>Keep ratio = {keep}</em></th></tr>')
        for method in methods:
            if method == ours[keep] and keep in ("1.0", "0.1"):
                vals = [cell(results, ds, keep, running) for _, ds in tasks]
            else:
                vals = ["—"] * len(tasks)
            L.append(f"    <tr><td>{method}</td>" + "".join(f"<td>{v}</td>" for v in vals)
                     + "<td>—</td></tr>")
    L += ["  </tbody>", "</table>"]
    return "\n".join(L)


def running_now() -> set[tuple[str, str]]:
    """What the sweep log says is in flight, so a cell reads 실행 중 not 대기."""
    logs = sorted(glob.glob(str(REPO_ROOT / "logs" / "eval" / "sweep_dream_*.log")))
    if not logs:
        return set()
    text = Path(logs[-1]).read_text(errors="ignore")
    starts = re.findall(r"^=== \[\d+/\d+\] (\S+) keep=([\d.]+)", text, re.M)
    finished = set(re.findall(r"^\[(?:ok|FAIL)\]\s+(\S+) keep=([\d.]+)", text, re.M))
    return {p for p in starts if p not in finished}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--post", action="store_true", help="update issue #5 in place")
    args = ap.parse_args()

    results, running = read_results(), running_now()
    body = f"{HEADER_NOTE}\n\n{main_table(results, running)}"
    comment = longbench_table(results, running)

    done = sorted(f"{d} keep={k} = {v:.2f}" for (d, k), v in results.items())
    print(f"완료 {len(results)}건" + ("".join(f"\n  {d}" for d in done) or " (없음)"), file=sys.stderr)
    print(f"실행 중: {sorted(running) or '없음'}", file=sys.stderr)

    if not args.post:
        print(body)
        print()
        print(comment)
        return 0

    subprocess.run(["gh", "issue", "edit", str(ISSUE), "--repo", GH_REPO, "--body-file", "-"],
                   input=body, text=True, check=True)
    ids = subprocess.run(["gh", "api", f"repos/{GH_REPO}/issues/{ISSUE}/comments", "--jq", ".[].id"],
                         capture_output=True, text=True, check=True).stdout.split()
    if not ids:
        subprocess.run(["gh", "issue", "comment", str(ISSUE), "--repo", GH_REPO, "--body-file", "-"],
                       input=comment, text=True, check=True)
    else:
        subprocess.run(["gh", "api", "-X", "PATCH",
                        f"repos/{GH_REPO}/issues/comments/{ids[0]}", "-F", "body=@-"],
                       input=comment, text=True, check=True, stdout=subprocess.DEVNULL)
    print(f"updated https://github.com/{GH_REPO}/issues/{ISSUE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
