#!/usr/bin/env python
"""Assemble the one-time adaptation cost table from a pipeline log.

Reads the ``cost:`` lines extract_teacher and train_student print and writes
results/adaptation_cost.json, keyed by backbone. Parsing the log rather than
re-timing means the table cannot drift from what the run actually did, and a
rerun on another backbone appends a row without touching this one.

    python scripts/adaptation_cost.py --log logs/... --backbone Dream-v0-Instruct-7B

A dataset whose cost line is missing -- extracted before the reporting existed,
say -- can be supplied as --reconstruct name=hours, and is recorded as such so
the table says which numbers were measured and which were recovered.
"""

from __future__ import annotations

import argparse, json, re, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COST = re.compile(r"^cost: (.+)$", re.M)


def fields(line):
    out = {}
    for token in line.split():
        if "=" in token:
            k, v = token.split("=", 1)
            try:
                out[k] = float(v) if "." in v else int(v)
            except ValueError:
                out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, help="pipeline log holding the cost lines")
    p.add_argument("--backbone", required=True, help="row key, e.g. Dream-v0-Instruct-7B")
    p.add_argument("--gpu", default="NVIDIA RTX A6000")
    p.add_argument("--reconstruct", action="append", default=[], metavar="NAME=HOURS",
                   help="a dataset's generate hours recovered from the log's "
                        "s/sample rather than measured in-process; repeatable")
    p.add_argument("--note", default="")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "adaptation_cost.json"))
    return p.parse_args()


def main():
    args = parse_args()
    text = Path(args.log).read_text(errors="replace")
    datasets, training = {}, None
    for line in COST.findall(text):
        f = fields(line)
        if "dataset" in f:
            datasets[f["dataset"]] = {"samples": f.get("generated"),
                                      "generate_h": f.get("generate_h"),
                                      "peak_alloc_gib": f.get("peak_alloc_gib"),
                                      "measured": True}
        elif "shards" in f:
            training = f
    if training is None:
        raise SystemExit(f"{args.log}: no training cost line; did the run finish?")

    for entry in args.reconstruct:
        name, _, hours = entry.partition("=")
        datasets[name] = {"samples": None, "generate_h": float(hours),
                          "peak_alloc_gib": None, "measured": False}

    teacher_h = sum(d["generate_h"] for d in datasets.values())
    peaks = [d["peak_alloc_gib"] for d in datasets.values() if d["peak_alloc_gib"]]
    row = {
        "gpu": args.gpu,
        "teacher_generation_gpu_h": round(teacher_h, 3),
        "student_training_gpu_h": round(training["train_h"], 3),
        "total_offline_gpu_h": round(teacher_h + training["train_h"], 3),
        "peak_vram_gib": round(max(peaks + [training["peak_alloc_gib"]]), 2),
        "teacher_peak_vram_gib": round(max(peaks), 2) if peaks else None,
        "training_peak_vram_gib": round(training["peak_alloc_gib"], 2),
        "teacher_samples": sum(d["samples"] for d in datasets.values() if d["samples"]),
        "teacher_per_dataset": datasets,
        "training_shards": training["shards"], "epochs": training["epochs"],
        "log": str(args.log), "note": args.note,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = json.loads(out.read_text()) if out.is_file() else {}
    table[args.backbone] = row
    out.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")

    print(f"{'Backbone':<24}{'Teacher':>10}{'Student':>10}{'Total':>10}{'Peak VRAM':>12}")
    print(f"{args.backbone:<24}{row['teacher_generation_gpu_h']:>9.3f}h"
          f"{row['student_training_gpu_h']:>9.3f}h{row['total_offline_gpu_h']:>9.3f}h"
          f"{row['peak_vram_gib']:>10.2f} GiB")
    for name, d in row["teacher_per_dataset"].items():
        flag = "" if d["measured"] else "   (reconstructed)"
        print(f"    {name:<16}{d['generate_h']:>8.4f}h{flag}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
