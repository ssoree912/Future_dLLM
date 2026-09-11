"""How well does each scorer reproduce the teacher's future-attention ranking?

The premise of this work is that predicting the attention a block *will* pay
beats Sparse-dLLM's proxy of the attention it pays *right now*. Training only
ever measured one side of that: the student reaches val recall 0.7345 against
the teacher, and nothing said what the baseline scores on the same labels. A
number with no comparator cannot support the claim either way.

This runs both scorers over the same validation shards the training run held
out, against the same labels, with the same recall metric, and adds a random
ranker so the scale is readable.

    student > baseline   the premise holds; look downstream for why it does
                         not show up in the task numbers
    student ~ baseline   the scorer is not the differentiator
    student < baseline   the learned scorer is worse than the heuristic it
                         replaces, and that is the whole gap

Nothing here trains or writes a checkpoint; it only reads teacher shards and
runs the same selection-time forward that train_student.features() does.

    python scripts/baseline_recall.py --student <checkpoint-best> [--limit 40]
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from future_dllm import CustomCache, load_model, load_prompt_utility_student

RATIOS = (0.05, 0.1, 0.2, 0.3, 0.5)


def recall_at(pred, target, ratios=RATIOS):
    """Overlap of the two rankings' top-k, per keep ratio.

    Same definition as train_student.recall_grid, kept per-ratio here because
    the deployed budget is 0.1 and an average over five ratios can hide it.
    """
    out = []
    for r in ratios:
        k = max(1, int(target.numel() * r))
        a = set(torch.topk(pred, k).indices.tolist())
        b = set(torch.topk(target, k).indices.tolist())
        out.append(len(a & b) / k)
    return out


def val_shards(roots, caps, val_ratio):
    """The same held-out split train_student.py made: sorted, capped, head slice."""
    out = []
    for index, root in enumerate(roots):
        found = sorted(glob.glob(f"{root}/*.pt"))
        if not found:
            raise SystemExit(f"no teacher shards under {root}")
        if caps:
            found = found[:caps[index]]
        split = max(1, int(len(found) * val_ratio))
        out += [(Path(root).name, p) for p in found[:split]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--model", default=str(REPO / "model/Dream-v0-Instruct-7B"))
    ap.add_argument("--limit", type=int, default=0,
                    help="shards per domain; 0 uses the whole val split")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    meta = json.load(open(Path(args.student).parent / "meta.json"))
    roots = meta["teacher_roots"]
    caps = [meta["max_shards"][Path(r).name] for r in roots] if meta.get("max_shards") else []
    shards = val_shards(roots, caps, meta["val_ratio"])
    if args.limit:
        per = {}
        kept = []
        for name, p in shards:
            if per.get(name, 0) < args.limit:
                per[name] = per.get(name, 0) + 1
                kept.append((name, p))
        shards = kept
    print(f"val shards: {len(shards)} over {len(roots)} domain(s)", flush=True)

    model, backend = load_model(args.model, max_seq_len=args.max_seq_len,
                                block_length=args.block_length, keep_ratio=1.0)
    model.eval()
    L = backend.n_layers
    student = load_prompt_utility_student(args.student, model.device)
    # train_student patches this on so the scorer sees deployment's hidden
    # states; the same patch has to be here or forward_layer has no input.
    CustomCache.capture_layer_hidden_states = (
        lambda self, layer_id, hidden: self.layer_hidden_states.__setitem__(layer_id, hidden))

    totals = {}          # domain -> {scorer: [sum per ratio], "n": count}
    with torch.no_grad():
        for si, (name, path) in enumerate(shards):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            for record in shard["blocks"]:
                x = record["x_at_block_start"].unsqueeze(0).to(model.device)
                ws = int(record.get("window_start", record["block_start"]))
                wl = int(record.get("window_length", record["block_length"]))
                cache = CustomCache(n_layers=L, device=model.device, keep_ratio=1.0,
                                    capture_current_scores=True)
                cache.layer_hidden_states = {}
                model(x, ws, 1, cache)

                cand = record["candidate_indices"].to(model.device)
                blk = torch.arange(ws, ws + wl, device=model.device)
                label = record["label_final_rowmax"].float().to(model.device)
                bucket = totals.setdefault(
                    name, {"student": [0.0] * len(RATIOS),
                           "baseline": [0.0] * len(RATIOS),
                           "random": [0.0] * len(RATIOS), "n": 0})
                for l in range(L):
                    tgt = label[l]
                    if not torch.isfinite(tgt).all() or tgt.sum() <= 0:
                        continue
                    # Both scorers are read in natural candidate order:
                    # current_scores is computed on keep_k before filter_cache
                    # permutes anything, and record_attention un-permutes the
                    # label, so the columns line up with each other.
                    base = cache.current_scores[l].squeeze(0).float()
                    stu = student.forward_layer(l, cache.layer_hidden_states[l].float(),
                                                cand, head="score",
                                                block_indices=blk).squeeze(0).float()
                    rnd = torch.rand_like(tgt)
                    if base.numel() != tgt.numel():
                        raise RuntimeError(
                            f"baseline scores {base.numel()} vs label {tgt.numel()}")
                    for key, pred in (("student", stu), ("baseline", base), ("random", rnd)):
                        for i, v in enumerate(recall_at(pred, tgt)):
                            bucket[key][i] += v
                    bucket["n"] += 1
            if (si + 1) % 5 == 0:
                print(f"  {si+1}/{len(shards)} shards", flush=True)

    header = "  ".join(f"@{int(r*100):02d}%" for r in RATIOS)
    print(f"\n{'domain':14s} {'scorer':9s} {header}   mean")
    macro = {k: [0.0] * len(RATIOS) for k in ("student", "baseline", "random")}
    for name in sorted(totals):
        b = totals[name]
        for key in ("student", "baseline", "random"):
            vals = [v / max(1, b["n"]) for v in b[key]]
            for i, v in enumerate(vals):
                macro[key][i] += v / len(totals)
            print(f"{name:14s} {key:9s} " +
                  "  ".join(f"{v:.3f}" for v in vals) +
                  f"   {sum(vals)/len(vals):.4f}")
    print()
    for key in ("student", "baseline", "random"):
        print(f"{'MACRO':14s} {key:9s} " +
              "  ".join(f"{v:.3f}" for v in macro[key]) +
              f"   {sum(macro[key])/len(macro[key]):.4f}")
    gap = macro["student"][1] - macro["baseline"][1]
    print(f"\nat the deployed keep_ratio 0.1: student - baseline = {gap:+.4f}")
    if args.out:
        json.dump({"per_domain": totals, "macro": macro, "ratios": list(RATIOS)},
                  open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
