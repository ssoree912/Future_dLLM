"""How well does each scorer reproduce the teacher's future-attention ranking?

The premise of this work is that predicting the attention a block *will* pay
beats Sparse-dLLM's proxy of the attention it pays *right now*. Training only
ever measured one side of that: the student reaches val recall 0.7345 against
the teacher, and nothing said what the baseline scores on the same labels. A
number with no comparator cannot support the claim either way.

This runs every scorer over the same validation shards the training run held
out, against the same labels, with the same recall metric, and adds a random
ranker so the scale is readable. The scorers are the student, Sparse-dLLM's
pooled current-attention proxy, and the teacher-style current-block score that
``eviction_method=current`` deploys -- the last one matters because the task
tables cannot separate it from the future label once the budget is loose.

Four views of the same ranking, because top-k overlap alone is a blunt
instrument: recall@k and Jaccard@k over the kept set, the share of the future
block's attention mass that set retains (weighted by how much the block will
actually look at each entry, and normalised by the best any selection could do
at that budget), and Spearman over the whole candidate axis, which sees a
ranking that is right everywhere except at the cut.

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
    Leading dimensions (for example KV heads) are averaged; Top-K is always
    taken over the candidate axis.
    """
    out = []
    for r in ratios:
        k = max(1, int(target.shape[-1] * r))
        a = torch.topk(pred, k, dim=-1).indices
        b = torch.topk(target, k, dim=-1).indices
        intersection = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
        out.append(float((intersection.float() / k).mean()))
    return out


def jaccard_at(pred, target, ratios=RATIOS):
    """Jaccard of equal-budget top-k sets, alongside the deployed recall view."""
    out = []
    for r in ratios:
        k = max(1, int(target.shape[-1] * r))
        a = torch.topk(pred, k, dim=-1).indices
        b = torch.topk(target, k, dim=-1).indices
        intersection = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
        out.append(float((intersection.float() / (2 * k - intersection)).mean()))
    return out


def mass_at(pred, target, ratios=RATIOS):
    """Future-attention mass the kept set retains, against the best possible.

    Set overlap counts every candidate the same. The label does not: a block
    pays most of its attention to a few entries, and dropping one of those
    costs more than dropping a dozen it barely looks at. This scores the mass
    the selection keeps, normalised by the mass the target's own top-k keeps,
    so 1.0 means "kept as much of the attention as anything could at this
    budget" rather than "kept all of it".
    """
    out = []
    for r in ratios:
        k = max(1, int(target.shape[-1] * r))
        pred_idx = torch.topk(pred, k, dim=-1).indices
        best = torch.topk(target, k, dim=-1).values.sum(dim=-1)
        kept = target.gather(-1, pred_idx).sum(dim=-1)
        out.append(float((kept / best.clamp_min(1e-12)).mean()))
    return out


def spearman(pred, target):
    """Rank correlation over the whole candidate axis, averaged over heads.

    Ties are broken by argsort order rather than averaged, which is what the
    label's exact zeros would need; they sit at the bottom of both rankings, so
    the effect on the correlation is small and the same for every scorer.
    """
    n = target.shape[-1]
    if n < 2:
        return 0.0
    rank = lambda t: t.argsort(dim=-1).argsort(dim=-1).float()
    a, b = rank(pred), rank(target)
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    denom = (a.pow(2).sum(-1) * b.pow(2).sum(-1)).sqrt().clamp_min(1e-12)
    return float(((a * b).sum(-1) / denom).mean())


SCORERS = ("student", "current", "baseline", "random")


def new_bucket():
    b = {"n": 0}
    for key in SCORERS:
        for metric in ("recall", "jaccard", "mass"):
            b[f"{key}_{metric}"] = [0.0] * len(RATIOS)
        b[f"{key}_spearman"] = 0.0
    return b


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
    ap.add_argument("--domains", default="",
                    help="comma-separated domain folders under --teacher-root to "
                         "score instead of the checkpoint's own val split. Use it "
                         "for a domain the student never trained on: there is no "
                         "split to respect there, so every shard is scored.")
    ap.add_argument("--teacher-root", default="",
                    help="directory holding the per-domain shard folders, when "
                         "meta.json records the path from another machine")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    meta = json.load(open(Path(args.student).parent / "meta.json"))
    if args.domains:
        if not args.teacher_root:
            raise SystemExit("--domains needs --teacher-root to resolve against")
        shards = []
        for name in [d for d in args.domains.split(",") if d]:
            found = sorted(glob.glob(str(Path(args.teacher_root) / name / "*.pt")))
            if not found:
                raise SystemExit(
                    f"no teacher shards under {Path(args.teacher_root) / name}")
            shards += [(name, p) for p in found]
    else:
        roots = meta["teacher_roots"]
        if args.teacher_root:
            # meta records absolute paths from the box that extracted the shards;
            # keep the domain names it recorded and re-root them here.
            roots = [str(Path(args.teacher_root) / Path(r).name) for r in roots]
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
                bucket = totals.setdefault(name, new_bucket())
                for l in range(L):
                    tgt = label[l]
                    if not torch.isfinite(tgt).all() or tgt.sum() <= 0:
                        continue
                    # Both scorers are read in natural candidate order:
                    # current_scores is computed on keep_k before filter_cache
                    # permutes anything, and record_attention un-permutes the
                    # label, so the columns line up with each other.
                    base = cache.current_scores[l].squeeze(0).float()
                    cur = cache.current_teacher_scores[l].squeeze(0).float()
                    stu = student.forward_layer(l, cache.layer_hidden_states[l].float(),
                                                cand, head="score",
                                                block_indices=blk).squeeze(0).float()
                    rnd = torch.rand_like(tgt)
                    # Sparse-dLLM produces one head-averaged ranking shared by
                    # all KV heads. Against a per-head teacher, compare that
                    # same deployed ranking with each head, then macro-average.
                    if base.ndim == 1 and tgt.ndim > 1:
                        base = base.expand_as(tgt)
                    if cur.ndim == 1 and tgt.ndim > 1:
                        cur = cur.expand_as(tgt)
                    for label_name, pred in (("student", stu), ("current", cur),
                                             ("baseline", base)):
                        if pred.shape != tgt.shape:
                            raise RuntimeError(
                                f"score/label shape mismatch at layer {l}: "
                                f"{label_name}={tuple(pred.shape)} "
                                f"label={tuple(tgt.shape)}")
                    for key, pred in (("student", stu), ("current", cur),
                                      ("baseline", base), ("random", rnd)):
                        for i, v in enumerate(recall_at(pred, tgt)):
                            bucket[f"{key}_recall"][i] += v
                        for i, v in enumerate(jaccard_at(pred, tgt)):
                            bucket[f"{key}_jaccard"][i] += v
                        for i, v in enumerate(mass_at(pred, tgt)):
                            bucket[f"{key}_mass"][i] += v
                        bucket[f"{key}_spearman"] += spearman(pred, tgt)
                    bucket["n"] += 1
            if (si + 1) % 5 == 0:
                print(f"  {si+1}/{len(shards)} shards", flush=True)

    header = "  ".join(f"@{int(r*100):02d}%" for r in RATIOS)
    macros = {}
    for metric in ("recall", "jaccard", "mass"):
        title = {"recall": "RECALL of the teacher's future-attention Top-K",
                 "jaccard": "JACCARD of the kept sets",
                 "mass": "FUTURE-ATTENTION MASS retained (1.0 = the best any "
                         "selection could keep)"}[metric]
        print(f"\n{title}")
        print(f"{'domain':14s} {'scorer':9s} {header}   mean")
        macro = {k: [0.0] * len(RATIOS) for k in SCORERS}
        for name in sorted(totals):
            b = totals[name]
            for key in SCORERS:
                vals = [v / max(1, b["n"]) for v in b[f"{key}_{metric}"]]
                for i, v in enumerate(vals):
                    macro[key][i] += v / len(totals)
                print(f"{name:14s} {key:9s} " +
                      "  ".join(f"{v:.3f}" for v in vals) +
                      f"   {sum(vals)/len(vals):.4f}")
        print()
        for key in SCORERS:
            print(f"{'MACRO':14s} {key:9s} " +
                  "  ".join(f"{v:.3f}" for v in macro[key]) +
                  f"   {sum(macro[key])/len(macro[key]):.4f}")
        macros[metric] = macro

    # Budget-free view: the whole ranking, not the set the cut happens to make.
    print("\nSPEARMAN against the future-attention label (whole candidate axis)")
    print(f"{'domain':14s} " + "  ".join(f"{k:>9s}" for k in SCORERS))
    spear = {k: 0.0 for k in SCORERS}
    for name in sorted(totals):
        b = totals[name]
        vals = {k: b[f"{k}_spearman"] / max(1, b["n"]) for k in SCORERS}
        for k in SCORERS:
            spear[k] += vals[k] / len(totals)
        print(f"{name:14s} " + "  ".join(f"{vals[k]:9.4f}" for k in SCORERS))
    print(f"{'MACRO':14s} " + "  ".join(f"{spear[k]:9.4f}" for k in SCORERS))
    macros["spearman"] = spear

    print("\nat the deployed keep_ratio 0.1, against the future label:")
    for metric in ("recall", "jaccard", "mass"):
        m = macros[metric]
        print(f"  {metric:8s} student {m['student'][1]:.4f}   "
              f"current {m['current'][1]:.4f}   "
              f"student - current = {m['student'][1] - m['current'][1]:+.4f}")
    sp = macros["spearman"]
    print(f"  spearman student {sp['student']:.4f}   current {sp['current']:.4f}   "
          f"student - current = {sp['student'] - sp['current']:+.4f}")
    if args.out:
        json.dump({"per_domain": totals, "macro": macros,
                   "scorers": list(SCORERS), "ratios": list(RATIOS)},
                  open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
