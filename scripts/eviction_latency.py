#!/usr/bin/env python
"""Time one cache selection: our scorer against the Sparse-dLLM baseline.

Measures ``CustomCache.filter_cache`` directly rather than a task benchmark.
The selection is a fixed amount of work -- ``(gen_length / block_length) x
n_layers`` calls, each over ``prompt + gen - block`` candidates -- so it is a
function of two lengths, and a task benchmark scrambles both: LongBench alone
varies the candidate pool by 30% across samples and the call count 16-fold
across datasets (musique 1 block, gov_report 16), on top of truncation,
scoring and whatever else shares the card. A sub-1% effect does not survive
that. Here the lengths are the sweep axis instead.

Both arms run the same tensors, the same keep_ratio and the same gather, so the
difference is the scoring rule and nothing else:

    student   28 layers of [cand ; block ; cand*block] -> MLP -> top-k per KV head
    sparse    mean-query . K, head-averaged, max-pooled -> one top-k, broadcast

    python scripts/eviction_latency.py --student <checkpoint-best>

Writes results/eviction_latency.json.
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student", required=True, help="checkpoint-best of the scorer")
    p.add_argument("--model", default=str(REPO_ROOT / "model" / "Dream-v0-Instruct-7B"),
                   help="read for layer count, width and KV heads; weights not loaded")
    p.add_argument("--candidates", default="256,1024,2016",
                   help="candidate pool sizes to sweep")
    p.add_argument("--keep-ratio", type=float, default=0.1)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "eviction_latency.json"))
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    from future_dllm import load_prompt_utility_student
    from future_dllm.cache import CustomCache

    raw = json.loads((Path(args.model) / "config.json").read_text())
    L = int(raw["num_hidden_layers"])
    H = int(raw["hidden_size"])
    KV = int(raw["num_key_value_heads"])
    Q = int(raw["num_attention_heads"])
    head_dim = H // Q
    device = torch.device("cuda")
    student = load_prompt_utility_student(args.student, device)

    def timed(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / args.iters * 1000

    rows = []
    for C in [int(x) for x in args.candidates.split(",")]:
        B = args.block_length
        seq = C + B
        cur = C // 2                      # block sits mid-sequence: prompt before, suffix after
        k = torch.randn(1, KV, seq, head_dim, device=device, dtype=torch.float32)
        v = torch.randn_like(k)
        q_block = torch.randn(1, Q, B, head_dim, device=device, dtype=torch.float32)
        hidden = torch.randn(1, seq, H, device=device, dtype=torch.float32)

        def one(method):
            def run():
                cache = CustomCache(n_layers=1, device=device, keep_ratio=args.keep_ratio,
                                    cache_scorer=student if method == "student" else None,
                                    eviction_method=method)
                cache.update_cache(0, k, v)
                if method == "student":
                    cache.layer_hidden_states[0] = hidden
                cache.filter_cache(0, q_block, cur, B)
            return run

        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        ours_ms = timed(one("student"))
        ours_peak = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        sparse_ms = timed(one("sparse"))
        sparse_peak = torch.cuda.max_memory_allocated() / 2**20

        row = {
            "candidates": C, "keep_ratio": args.keep_ratio,
            "per_layer_ms": {"ours": round(ours_ms, 4), "sparse": round(sparse_ms, 4)},
            # One selection is every layer scoring the same pool, which is what a
            # block costs; the model has L of them per block.
            "per_selection_ms": {"ours": round(ours_ms * L, 3),
                                 "sparse": round(sparse_ms * L, 3)},
            "delta_ms": round((ours_ms - sparse_ms) * L, 3),
            "ratio": round(ours_ms / sparse_ms, 2),
            "peak_mib": {"ours": round(ours_peak, 1), "sparse": round(sparse_peak, 1)},
        }
        rows.append(row)
        print(f"C={C:<6} per-layer  ours {ours_ms:7.4f} ms  sparse {sparse_ms:7.4f} ms"
              f"   selection({L} layers)  ours {ours_ms*L:7.2f}  sparse {sparse_ms*L:7.2f}"
              f"   x{ours_ms/sparse_ms:.2f}")

    out = {"gpu": torch.cuda.get_device_name(0), "backbone": Path(args.model).name,
           "layers": L, "kv_heads": KV, "query_heads": Q, "head_dim": head_dim,
           "student": str(args.student), "block_length": args.block_length,
           "iters": args.iters, "rows": rows,
           "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
