#!/usr/bin/env python
"""Measure one scorer size's offline-free cost: params, latency, added VRAM.

The three columns a size-vs-performance table needs that do not come from an
eval run. Accuracy columns stay with the eval results; this only measures what
the scorer costs to carry.

    python scripts/scorer_cost.py --size base --model model/Dream-v0-Instruct-7B

Rows accumulate in results/scorer_cost.json keyed by size, so Small and Large
can be added later with the same command and compared against Base without
re-measuring it.

The backbone is not loaded. "Added VRAM" is what the scorer adds *on top of* a
backbone that is already resident, which is exactly the scorer's own weights
plus the activations of one selection -- measuring it beside a loaded backbone
would fold the backbone's allocator caching into the number.
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT))

import torch

# proj_dim / mlp_dim. Base is what this repo trains today.
SIZES = {"small": (128, 256), "base": (256, 512), "large": (512, 1024)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", required=True, choices=sorted(SIZES),
                   help="scorer width preset; the row's name in the table")
    p.add_argument("--model", required=True,
                   help="backbone whose layer count, width and KV heads the "
                        "scorer is built for. Read from its config only -- the "
                        "weights are never loaded")
    p.add_argument("--candidates", type=int, default=2016,
                   help="candidate pool one selection scores. Default is the "
                        "2048-context worst case: prompt+generation minus the "
                        "32-token block")
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "scorer_cost.json"))
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU: latency and VRAM are the point")
    from future_dllm.backends import detect_family
    from future_dllm import PromptUtilityStudent, StudentConfig

    cfg_path = Path(args.model) / "config.json"
    raw = json.loads(cfg_path.read_text())
    family = detect_family(args.model)
    if family == "dream":
        layers, hidden = int(raw["num_hidden_layers"]), int(raw["hidden_size"])
        kv_heads = int(raw["num_key_value_heads"])
    else:
        layers, hidden = int(raw["n_layers"]), int(raw["d_model"])
        kv_heads = int(raw.get("n_kv_heads") or raw["n_heads"])

    proj_dim, mlp_dim = SIZES[args.size]
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    student = PromptUtilityStudent(StudentConfig(
        layer_count=layers, hidden_dim=hidden, proj_dim=proj_dim,
        mlp_dim=mlp_dim, heads=("score",), attn_heads=kv_heads)).to(device).float().eval()
    weights_mib = (torch.cuda.memory_allocated() - before) / 2**20
    params = sum(p.numel() for p in student.parameters())

    # One selection: every layer scores the same pool, which is what
    # filter_cache does once per block per layer.
    seq = args.candidates + args.block_length
    hidden_states = torch.randn(1, seq, hidden, device=device)
    cand = torch.arange(args.candidates, device=device)
    blk = torch.arange(args.candidates, seq, device=device)

    @torch.no_grad()
    def selection():
        for l in range(layers):
            student.forward_layer(l, hidden_states, cand, head="score",
                                  block_indices=blk)

    for _ in range(args.warmup):
        selection()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    peak_before = torch.cuda.memory_allocated()
    started = time.perf_counter()
    for _ in range(args.iters):
        selection()
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) / args.iters * 1000
    # What the scorer adds while it runs: its weights plus the transient
    # activations of one selection, over the hidden states the backbone already
    # has resident.
    activation_mib = (torch.cuda.max_memory_allocated() - peak_before) / 2**20

    row = {
        "size": args.size, "proj_dim": proj_dim, "mlp_dim": mlp_dim,
        "backbone": Path(args.model).name, "layers": layers,
        "hidden_dim": hidden, "kv_heads": kv_heads,
        "params_m": round(params / 1e6, 2),
        "selection_latency_ms": round(latency_ms, 3),
        "per_layer_latency_ms": round(latency_ms / layers, 4),
        "added_vram_mib": round(weights_mib + activation_mib, 1),
        "weights_mib": round(weights_mib, 1),
        "activation_mib": round(activation_mib, 1),
        "candidates": args.candidates, "block_length": args.block_length,
        "iters": args.iters,
        "gpu": torch.cuda.get_device_name(0),
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = json.loads(out.read_text()) if out.is_file() else {}
    table[args.size] = row
    out.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")

    width = max(len(k) for k in row)
    for k, v in row.items():
        print(f"  {k:<{width}}  {v}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
