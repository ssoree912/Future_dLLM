#!/usr/bin/env python
"""End-to-end decoding speed: our scorer against the Sparse-dLLM baseline.

Answers the question the selection microbenchmark cannot: once the cache is
pruned, does decoding itself differ? Both rules keep ``int(candidates *
keep_ratio)`` entries and gather to the same ``[1, kv heads, k, head_dim]``, so
the attention after selection should cost the same and only the selection should
separate the two -- an argument worth checking rather than asserting.

Shaped after Sparse-dLLM's own speed harness (``myeval/eval_speed``): batch 1,
fixed prompt and generation lengths, tokens per second and peak memory, no
scoring. Lengths are fixed because the selection cost is a function of them --
``(gen_length / block_length)`` selections over ``prompt + gen - block``
candidates -- so a task benchmark would vary the thing being measured.

Prompts come from the real tokenised shards, truncated to the target length, so
the sequence the model sees is the one it sees in evaluation.

    python scripts/decode_tps.py --student <checkpoint-best>
"""

from __future__ import annotations

import argparse, glob, json, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student", required=True)
    p.add_argument("--model", default=str(REPO_ROOT / "model" / "Dream-v0-Instruct-7B"))
    p.add_argument("--shard-root",
                   default=str(REPO_ROOT / "artifacts" / "prompt_shards_dream_2048"))
    p.add_argument("--shard-dataset", default="gov_report",
                   help="a domain whose prompts are long enough to truncate from")
    p.add_argument("--points", default="512x512,1536x512,2016x32",
                   help="prompt x generation pairs; prompt+gen must fit max_seq_len")
    p.add_argument("--samples", type=int, default=5, help="timed generations per point")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--keep-ratio", type=float, default=0.1)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--dream-steps", type=int, default=512)
    p.add_argument("--dream-temperature", type=float, default=0.2)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "decode_tps.json"))
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    from future_dllm import load_model, load_prompt_utility_student
    from future_dllm.dream_decoding import DreamDecoding
    from future_dllm.dream_generate import generate

    model, backend = load_model(args.model, max_seq_len=args.max_seq_len,
                                block_length=args.block_length,
                                keep_ratio=args.keep_ratio)
    student = load_prompt_utility_student(args.student, model.device)
    decoding = DreamDecoding(alg="entropy", temperature=args.dream_temperature,
                             top_p=0.95, steps=args.dream_steps)
    print(f"keep_ratio={args.keep_ratio} decoding={decoding.metadata()}", flush=True)

    shards = sorted(glob.glob(f"{args.shard_root}/{args.shard_dataset}/*.pt"))
    if not shards:
        raise SystemExit(f"no prompt shards under {args.shard_root}/{args.shard_dataset}")

    def prompt_of(length, index):
        ids = torch.load(shards[index % len(shards)], map_location="cpu",
                         weights_only=False)["prompt_input_ids"].to(torch.long)
        if ids.numel() < length:
            ids = ids.repeat((length // ids.numel()) + 1)
        return ids[-length:].unsqueeze(0).to(model.device)

    rows = []
    for point in args.points.split(","):
        P, G = (int(x) for x in point.lower().split("x"))
        if P + G > args.max_seq_len:
            raise SystemExit(f"{point}: prompt+gen exceeds max_seq_len")
        kwargs = decoding.generation_kwargs(G)
        entry = {"prompt_len": P, "gen_len": G, "steps": kwargs["steps"],
                 "selections_per_sample": G // args.block_length,
                 "candidates": P + G - args.block_length, "arms": {}}
        for arm, scorer, method in (("ours", student, "student"),
                                    ("sparse", None, "sparse")):
            with torch.no_grad():
                for i in range(args.warmup):
                    generate(model, prompt_of(P, i), gen_length=G,
                             block_length=args.block_length, mask_id=backend.mask_id,
                             cache_scorer=scorer, eviction_method=method, **kwargs)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for i in range(args.samples):
                    generate(model, prompt_of(P, i), gen_length=G,
                             block_length=args.block_length, mask_id=backend.mask_id,
                             cache_scorer=scorer, eviction_method=method, **kwargs)
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) / args.samples
            entry["arms"][arm] = {
                "s_per_sample": round(elapsed, 4),
                "tps": round(G / elapsed, 2),
                "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            }
            print(f"  P={P} G={G} {arm:<7} {elapsed:7.3f} s/sample  "
                  f"{G/elapsed:7.2f} tok/s  peak {entry['arms'][arm]['peak_mib']:.0f} MiB",
                  flush=True)
        o, s = entry["arms"]["ours"], entry["arms"]["sparse"]
        entry["overhead_pct"] = round((o["s_per_sample"] / s["s_per_sample"] - 1) * 100, 2)
        entry["overhead_s"] = round(o["s_per_sample"] - s["s_per_sample"], 4)
        print(f"  P={P} G={G} -> ours is {entry['overhead_pct']:+.2f}% "
              f"({entry['overhead_s']:+.3f} s/sample)", flush=True)
        rows.append(entry)

    out = {"gpu": torch.cuda.get_device_name(0), "backbone": Path(args.model).name,
           "student": str(args.student), "keep_ratio": args.keep_ratio,
           "block_length": args.block_length, "decoding": decoding.metadata(),
           "samples_per_point": args.samples, "rows": rows,
           "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
