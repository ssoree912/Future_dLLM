#!/usr/bin/env python
"""Compare future- and current-attention eviction with a full-cache run.

The three inputs are lm-eval ``samples_*.jsonl`` files produced from the same
task and sample order.  Records are aligned by document/prompt hash rather than
line number.  Two dependency-free output-similarity measures are reported:

* ``sequence``: :class:`difflib.SequenceMatcher` over normalized tokens.
* ``jaccard``: set Jaccard over the same normalized tokens.

This is an *offline diagnostic*.  Looking at the full-cache answer is legitimate
for deciding which method is closer in an experiment, but it must not be used as
an online eviction oracle.  Deployment should route a task using held-out
diagnostics, never choose per example after seeing the reference answer.

Example::

    python scripts/compare_eviction_similarity.py \
      --full results/.../full_samples \
      --future results/.../future_samples \
      --current results/.../current_samples \
      --out results/eviction_similarity/gsm8k
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def normalized_tokens(text: str) -> list[str]:
    """Case-fold text into word/punctuation tokens for both similarity metrics."""
    return TOKEN_RE.findall(str(text).casefold())


def sequence_similarity(left: str, right: str) -> float:
    """Order-sensitive token similarity in ``[0, 1]``."""
    return difflib.SequenceMatcher(
        None, normalized_tokens(left), normalized_tokens(right), autojunk=False
    ).ratio()


def jaccard_similarity(left: str, right: str) -> float:
    """Token-set Jaccard in ``[0, 1]``; two empty outputs are identical."""
    a, b = set(normalized_tokens(left)), set(normalized_tokens(right))
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _sample_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("samples_*.jsonl"))
        if not files:
            files = sorted(path.rglob("*.jsonl"))
        if files:
            return files
    raise FileNotFoundError(f"no sample jsonl found at {path}")


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    """Stable cross-run key for one generated continuation.

    lm-eval writes one sample record per task filter.  GSM8K therefore emits
    the same raw generation twice (strict-match and flexible-extract).  Filter
    names must not turn those into two diagnostic samples.
    """
    for field in ("doc_hash", "prompt_hash"):
        value = record.get(field)
        if value not in (None, ""):
            return str(value), "raw-generation"
    if "doc_id" in record:
        return f"doc_id:{record['doc_id']}", "raw-generation"
    raise ValueError("sample has none of doc_hash, prompt_hash or doc_id")


def _response(record: dict[str, Any]) -> str:
    # Compare the generated continuation, not task-filtered output. GSM8K's
    # strict filter collapses a chain of thought to one number, while
    # HumanEval's create_test filter prepends the shared prompt; either would
    # conceal the eviction-induced text drift this diagnostic is meant to see.
    raw: Any = record.get("resps")
    while isinstance(raw, list) and raw:
        raw = raw[0]
    if raw is not None:
        return str(raw)
    filtered: Any = record.get("filtered_resps")
    while isinstance(filtered, list) and filtered:
        filtered = filtered[0]
    return "" if filtered is None else str(filtered)


def load_samples(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for sample_file in _sample_files(path):
        with sample_file.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {sample_file}:{line_number}: {exc}"
                    ) from exc
                key = _record_key(record)
                if key in records:
                    if _response(records[key]) == _response(record):
                        continue
                    raise ValueError(
                        f"duplicate sample key {key!r} has different raw responses under {path}"
                    )
                records[key] = record
    if not records:
        raise ValueError(f"no sample records found at {path}")
    return records


def _score(candidate: str, full: str) -> dict[str, float]:
    sequence = sequence_similarity(candidate, full)
    jaccard = jaccard_similarity(candidate, full)
    return {"sequence": sequence, "jaccard": jaccard,
            "mean": (sequence + jaccard) / 2.0}


def compare_runs(
    full: dict[tuple[str, str], dict[str, Any]],
    future: dict[tuple[str, str], dict[str, Any]],
    current: dict[tuple[str, str], dict[str, Any]],
    *,
    low_threshold: float = 0.70,
    min_gain: float = 0.01,
    decision_metric: str = "mean",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return aggregate summary and per-sample comparisons."""
    key_sets = {"full": set(full), "future": set(future), "current": set(current)}
    common = set.intersection(*key_sets.values())
    if not common:
        raise ValueError("the three runs have no aligned samples")
    missing = {name: len(keys - common) for name, keys in key_sets.items()}

    rows: list[dict[str, Any]] = []
    for key in sorted(common):
        full_text = _response(full[key])
        future_text = _response(future[key])
        current_text = _response(current[key])
        future_score = _score(future_text, full_text)
        current_score = _score(current_text, full_text)
        future_decision = future_score[decision_metric]
        current_decision = current_score[decision_metric]
        low_future = future_decision < low_threshold
        use_current = low_future and current_decision >= future_decision + min_gain
        source = full[key]
        rows.append({
            "key": key[0],
            "response_scope": key[1],
            "doc_id": source.get("doc_id"),
            "future": future_score,
            "current": current_score,
            "delta_current_minus_future": {
                metric: current_score[metric] - future_score[metric]
                for metric in ("sequence", "jaccard", "mean")
            },
            "future_is_low": low_future,
            "diagnostic_choice": "current" if use_current else "future",
        })

    metrics = ("sequence", "jaccard", "mean")
    means = {
        method: {
            metric: statistics.fmean(row[method][metric] for row in rows)
            for metric in metrics
        }
        for method in ("future", "current")
    }
    wins = Counter()
    for row in rows:
        delta = row["delta_current_minus_future"][decision_metric]
        wins["current" if delta > min_gain else "future" if delta < -min_gain else "tie"] += 1
    mean_delta = means["current"][decision_metric] - means["future"][decision_metric]
    dataset_choice = (
        "current" if mean_delta > min_gain
        else "future" if mean_delta < -min_gain
        else "tie"
    )
    summary = {
        "aligned_samples": len(rows),
        "unaligned_samples": missing,
        "decision_metric": decision_metric,
        "low_threshold": low_threshold,
        "min_gain": min_gain,
        "mean_similarity_to_full": means,
        "mean_delta_current_minus_future": {
            metric: means["current"][metric] - means["future"][metric]
            for metric in metrics
        },
        "per_sample_wins": {name: wins[name] for name in ("future", "current", "tie")},
        "low_future_samples": sum(row["future_is_low"] for row in rows),
        "low_future_switched_to_current": sum(
            row["diagnostic_choice"] == "current" for row in rows
        ),
        "held_out_dataset_recommendation": dataset_choice,
        "warning": (
            "Recommendation is diagnostic only. Route future/current using a held-out "
            "dataset; do not inspect the full-cache answer at deployment time."
        ),
    }
    return summary, rows


def _print_summary(summary: dict[str, Any]) -> None:
    means = summary["mean_similarity_to_full"]
    delta = summary["mean_delta_current_minus_future"]
    print(f"aligned samples: {summary['aligned_samples']}")
    print("method    sequence  jaccard   mean")
    for method in ("future", "current"):
        print(f"{method:8s}  {means[method]['sequence']:.4f}    "
              f"{means[method]['jaccard']:.4f}   {means[method]['mean']:.4f}")
    print(f"current-future: sequence {delta['sequence']:+.4f}, "
          f"jaccard {delta['jaccard']:+.4f}, mean {delta['mean']:+.4f}")
    print(f"wins ({summary['decision_metric']}): {summary['per_sample_wins']}")
    print(f"future below {summary['low_threshold']:.2f}: "
          f"{summary['low_future_samples']} "
          f"(current better by >= {summary['min_gain']:.3f}: "
          f"{summary['low_future_switched_to_current']})")
    print("held-out dataset recommendation: "
          f"{summary['held_out_dataset_recommendation']}")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True,
                        help="full-cache samples JSONL or directory")
    parser.add_argument("--future", required=True,
                        help="future-scorer samples JSONL or directory")
    parser.add_argument("--current", required=True,
                        help="current-attention samples JSONL or directory")
    parser.add_argument("--out", default="",
                        help="optional directory for summary.json and samples.jsonl")
    parser.add_argument("--low-threshold", type=float, default=0.70)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--decision-metric", choices=("sequence", "jaccard", "mean"),
                        default="mean")
    args = parser.parse_args(argv)
    if not 0.0 <= args.low_threshold <= 1.0:
        parser.error("--low-threshold must be in [0, 1]")
    if args.min_gain < 0.0:
        parser.error("--min-gain must be non-negative")

    summary, rows = compare_runs(
        load_samples(args.full), load_samples(args.future), load_samples(args.current),
        low_threshold=args.low_threshold, min_gain=args.min_gain,
        decision_metric=args.decision_metric,
    )
    _print_summary(summary)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _write_jsonl(out / "samples.jsonl", rows)
        print(f"wrote {out / 'summary.json'} and {out / 'samples.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
