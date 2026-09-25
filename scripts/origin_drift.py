"""How far does an evicted run drift from the un-evicted one?

The task tables say a keep ratio costs so many accuracy points. They do not say
whether the model is still answering the same way. This measures that directly:
for every document, the keep=1.0 generation is the reference, and each evicted
arm is scored on how much of that reference survives.

    origin        keep_ratio 1.0, no eviction -- the reference, never an arm
    sparse-dLLM   EVICTION_METHOD=sparse
    ours          the student scorer

LLaDA's eval path pins temperature to 0 (``eval/lm_eval_model.py``), remasking
is deterministic, and gen_length is a fixed per-task budget rounded up to a
block multiple. Two runs over the same documents therefore produce sequences of
the same length whose positions correspond, and every difference between them is
caused by eviction rather than by sampling. That is what makes a position-wise
comparison meaningful here; it would not be on a sampled decoder.

Inputs are the resume stores the eval driver already writes -- no rerun and no
``--log_samples`` needed. Records carry ``ctx``, a hash of the request alone, so
arms join per document even though their ``key`` differs by construction.

    python scripts/origin_drift.py \
        --dataset qasper \
        --tokenizer model/LLaDA-8B-Instruct \
        --origin results/.resume/<keep1.0 run>.jsonl \
        --arm sparse:0.2:results/.resume/<...>.jsonl \
        --arm ours:0.2:results/.resume/<...>.jsonl \
        --out results/drift/qasper

Stores written before ``ctx`` existed cannot be joined and are refused with the
rerun they need, rather than silently lining up the wrong documents.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_store(path, field):
    """ctx -> generation, from one resume jsonl.

    A store is append-only and a resumed run re-appends nothing, but a document
    retried after a crash can appear twice; the last write wins, matching what
    the driver itself replays.
    """
    out = {}
    no_ctx = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue              # a line a crash cut in half
        if "ctx" not in rec:
            no_ctx += 1
            continue
        out[rec["ctx"]] = rec.get(field, rec.get("text", ""))
    if no_ctx:
        raise SystemExit(
            f"{path}: {no_ctx} records predate the ctx field and cannot be joined.\n"
            f"Rerun that arm -- the store is keyed per (keep_ratio, checkpoint), so "
            f"nothing else on disk can supply it.")
    return out


def drift(ref_ids, arm_ids):
    """One document's agreement with the reference, position by position.

    Positions past the shorter sequence count as mismatches: a run that stops
    early has not preserved the response, and scoring only the overlap would
    read truncation as perfect agreement.

    ``token_f1`` and ``token_jaccard`` drop the positions and compare the
    tokens as a bag. The same words in a different order score 1.0 there and
    near 0 on ``token_agreement``, which is why both are carried: the
    position-wise view says "same answer", the bag view says "same vocabulary".
    Deterministic decoding is what makes the position-wise view meaningful (see
    the module docstring); the bag view would be the only honest one without it.
    """
    span = max(len(ref_ids), len(arm_ids))
    if span == 0:
        return {"token_agreement": 1.0, "first_divergence": 1.0,
                "token_f1": 1.0, "token_jaccard": 1.0,
                "exact_match": 1, "ref_len": 0, "arm_len": 0}
    same = sum(1 for a, b in zip(ref_ids, arm_ids) if a == b)
    # F1 over multisets, so a token repeated twice in origin and once here is
    # half credit rather than full; Jaccard over sets, which ignores counts.
    overlap = sum((Counter(ref_ids) & Counter(arm_ids)).values())
    union = len(set(ref_ids) | set(arm_ids))
    first = next((i for i, (a, b) in enumerate(zip(ref_ids, arm_ids)) if a != b),
                 min(len(ref_ids), len(arm_ids)))
    return {
        "token_agreement": same / span,
        # normalised by the reference, so "held together for 90% of the answer
        # origin gave" reads the same across documents of different lengths
        "first_divergence": min(first / max(1, len(ref_ids)), 1.0),
        "token_f1": (2 * overlap / (len(ref_ids) + len(arm_ids))) if overlap else 0.0,
        "token_jaccard": (len(set(ref_ids) & set(arm_ids)) / union) if union else 1.0,
        "exact_match": int(ref_ids == arm_ids),
        "ref_len": len(ref_ids),
        "arm_len": len(arm_ids),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="label for the output only")
    ap.add_argument("--origin", required=True,
                    help="resume store of the keep_ratio=1.0 run")
    ap.add_argument("--arm", action="append", default=[], metavar="METHOD:RATIO:PATH",
                    help="repeatable, e.g. ours:0.2:results/.resume/foo.jsonl")
    ap.add_argument("--tokenizer", default=str(REPO / "model/LLaDA-8B-Instruct"))
    ap.add_argument("--field", default="raw", choices=("raw", "text"),
                    help="raw is the fixed-length decode, so positions align across "
                         "arms; text is cut at a stop string, which moves with the "
                         "content (default: raw)")
    ap.add_argument("--out", default="", help="directory for summary.json and per_sample.jsonl")
    args = ap.parse_args()

    if not args.arm:
        raise SystemExit("no --arm given; there is nothing to compare against origin")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def encode(text):
        return tok.encode(text, add_special_tokens=False)

    origin = load_store(args.origin, args.field)
    print(f"origin: {len(origin)} documents", flush=True)
    ref_ids = {c: encode(t) for c, t in origin.items()}

    rows, per_sample = [], []
    for spec in args.arm:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"--arm wants METHOD:RATIO:PATH, got {spec!r}")
        method, ratio, path = parts[0], float(parts[1]), parts[2]
        arm = load_store(path, args.field)
        shared = sorted(set(ref_ids) & set(arm))
        if not shared:
            raise SystemExit(
                f"{method}@{ratio}: no documents in common with origin. Both runs "
                f"have to cover the same task and the same --limit.")
        stats = defaultdict(list)
        for c in shared:
            d = drift(ref_ids[c], encode(arm[c]))
            for k, v in d.items():
                stats[k].append(v)
            per_sample.append({"method": method, "keep_ratio": ratio, "ctx": c, **d})
        row = {"method": method, "keep_ratio": ratio, "n": len(shared),
               "missing_vs_origin": len(ref_ids) - len(shared)}
        for k in ("token_agreement", "first_divergence", "token_f1",
                  "token_jaccard", "exact_match", "ref_len", "arm_len"):
            row[k] = statistics.fmean(stats[k])
        rows.append(row)
        print(f"  {method:8s} keep={ratio:<5} n={row['n']:4d}  "
              f"agree={row['token_agreement']:.4f}  "
              f"firstdiv={row['first_divergence']:.4f}  "
              f"f1={row['token_f1']:.4f}  "
              f"jacc={row['token_jaccard']:.4f}  "
              f"exact={row['exact_match']:.4f}", flush=True)

    rows.sort(key=lambda r: (r["method"], r["keep_ratio"]))
    print(f"\n### {args.dataset} -- agreement with origin (keep_ratio 1.0)\n")
    print("| Method | Keep | n | Token agreement | First divergence | "
          "Token F1 | Jaccard | Exact match |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['method']} | {r['keep_ratio']:.2f} | {r['n']} | "
              f"{r['token_agreement']:.4f} | {r['first_divergence']:.4f} | "
              f"{r['token_f1']:.4f} | {r['token_jaccard']:.4f} | "
              f"{r['exact_match']:.4f} |")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(
            {"dataset": args.dataset, "field": args.field,
             "origin": str(args.origin), "rows": rows}, indent=2) + "\n")
        with open(out / "per_sample.jsonl", "w") as fh:
            for rec in per_sample:
                fh.write(json.dumps(rec) + "\n")
        print(f"\nwrote {out}/summary.json and {out}/per_sample.jsonl")


if __name__ == "__main__":
    sys.exit(main())
