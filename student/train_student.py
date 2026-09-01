"""Train a scorer to predict the final x row-max label.

Features are replayed rather than stored: each record keeps the exact block input
(`x_at_block_start`), so the forward that produced the student's deployment-time
hidden states can be reproduced here. That is the one thing the previous student
got wrong — it trained on a prompt-only forward and was deployed on a full
prompt+generation forward.

Targets are ranking targets, so the loss is listwise (KL against the normalised
label, weighted by --lambda-list) plus a pairwise term sampled across the whole
range, and checkpoints are selected on mean recall over a k-grid rather than any
single budget.
"""

from __future__ import annotations

import argparse, glob, hashlib, json, random, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(REPO_ROOT / "model" / "LLaDA-8B-Instruct"))
    p.add_argument("--teacher-root", default=str(REPO_ROOT / "artifacts/teacher/samsum"),
                   help="comma-separated for mixed-domain training: val is split "
                        "per domain and the checkpoint is chosen on the domain "
                        "macro average, so a block-heavy domain cannot own it")
    p.add_argument("--output-dir", default="",
                   help="default: artifacts/ckpts/<auto name>, see checkpoint_name()")
    p.add_argument("--name", default="",
                   help="override just the directory name under artifacts/ckpts")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--mlp-dim", type=int, default=512)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="reject teacher records longer than this total sequence length")
    p.add_argument("--lambda-list", type=float, default=1.0,
                   help="weight on the listwise KL against the pairwise term")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-shards", default="",
                   help="comma list aligned with --teacher-root: use only the first "
                        "N prompts of each domain, 0 = all. The cut is taken before "
                        "the val split, "
                        "so val stays the same fraction of what is used.")
    return p.parse_args()


def recall_grid(pred, target, ratios=(0.05, 0.1, 0.2, 0.3, 0.5)):
    out = []
    for r in ratios:
        k = max(1, int(target.numel() * r))
        a = set(torch.topk(pred, k).indices.tolist())
        b = set(torch.topk(target, k).indices.tolist())
        out.append(len(a & b) / k)
    return sum(out) / len(out)


def checkpoint_name(datasets, counts, epochs, lr):
    """What separates one scorer from another: which domains, how many samples of
    each, and the two training knobs. The domain names themselves do not fit in a
    directory name once there are four of them, so they go in through a hash and
    are written out in full in meta.json."""
    tag = hashlib.sha1(",".join(sorted(datasets)).encode()).hexdigest()[:6]
    lr_text = f"{lr:g}"
    if "e" not in lr_text and lr < 0.01:      # 0.0002 reads worse than 2e-4
        lr_text = f"{lr:.1e}".replace(".0e", "e")
    lr_text = lr_text.replace("e-0", "e-")
    return (f"{len(datasets)}ds_{'-'.join(str(c) for c in counts)}_e{epochs}"
            f"_lr{lr_text}_blk_{tag}")


def load_shard(path, attempts=3):
    """The NAS the shards live on throws transient EIO under load; one of those
    five hours into a run must not kill it."""
    for i in range(attempts):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))


def main():
    args = parse_args()
    if args.max_seq_len < 1:
        raise SystemExit("--max-seq-len must be positive")
    torch.manual_seed(args.seed); random.seed(args.seed)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from transformers import AutoConfig
    from future_dllm import LLaDAModelLM, CustomCache

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    native_limit = int(getattr(cfg, "max_sequence_length", args.max_seq_len))
    if args.max_seq_len > native_limit:
        print(f"warning: max_seq_len={args.max_seq_len} exceeds the checkpoint's "
              f"trained context {native_limit}", flush=True)
    cfg.max_sequence_length = args.max_seq_len
    cfg.block_len, cfg.keep_ratio = 32, 1.0
    model = LLaDAModelLM.from_pretrained(args.model, config=cfg, device_map="auto",
                                         torch_dtype=torch.bfloat16,
                                         trust_remote_code=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device, L, H = model.device, model.config.n_layers, model.config.d_model

    # Capture has to stay on so training sees the same hidden states deployment
    # will hand the scorer.
    CustomCache.capture_layer_hidden_states = (
        lambda self, layer_id, hidden: self.layer_hidden_states.__setitem__(layer_id, hidden))

    # Split val per domain: mmlu carries 2 blocks per sample against 4 elsewhere,
    # so a sample-balanced val would let the block-heavy domains own the choice.
    roots = [r for r in args.teacher_root.split(",") if r]
    shard_caps = [int(c) for c in args.max_shards.split(",")] if args.max_shards else []
    if shard_caps and len(shard_caps) != len(roots):
        raise SystemExit("--max-shards needs one entry per teacher root")
    train_shards, val_shards, datasets = [], [], []
    for index, root in enumerate(roots):
        name = Path(root).name
        found = sorted(glob.glob(f"{root}/*.pt"))
        if not found:
            raise SystemExit(f"no teacher shards under {root}")
        cap = shard_caps[index] if shard_caps else 0
        if cap:
            if cap > len(found):
                raise SystemExit(f"{name}: asked for {cap} prompts, only {len(found)} exist")
            found = found[:cap]
        split = max(1, int(len(found) * args.val_ratio))
        val_shards += [(name, p) for p in found[:split]]
        train_shards += [(name, p) for p in found[split:]]
        datasets.append(name)
        print(f"  {name}: train {len(found)-split} / val {split} shards", flush=True)
    print(f"train {len(train_shards)} / val {len(val_shards)} shards "
          f"over {len(datasets)} domain(s)", flush=True)

    counts = [sum(1 for n, _ in train_shards + val_shards if n == d) for d in datasets]
    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "artifacts" / "ckpts" /
        (args.name or checkpoint_name(datasets, counts, args.epochs, args.lr)))
    print(f"checkpoint -> {out_dir}", flush=True)

    # Same class the deployment path loads, so the checkpoint drops straight in.
    from future_dllm import PromptUtilityStudent, StudentConfig
    student_cfg = StudentConfig(layer_count=L, hidden_dim=H, proj_dim=args.proj_dim,
                                mlp_dim=args.mlp_dim, heads=("score",))
    student = PromptUtilityStudent(student_cfg).to(device).float()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"datasets": datasets, "samples": dict(zip(datasets, counts)),
               "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
               "proj_dim": args.proj_dim, "mlp_dim": args.mlp_dim,
               "val_ratio": args.val_ratio, "pairs": args.pairs,
               "max_seq_len": args.max_seq_len,
               "cache_selection_step": 0,
               "lambda_list": args.lambda_list,
               "teacher_roots": roots,
               "max_shards": dict(zip(datasets, shard_caps)) if shard_caps else {}},
              open(out_dir / "meta.json", "w"), indent=2)

    @torch.no_grad()
    def features(record):
        selection_step = record.get(
            "cache_selection_step", record.get("cache_delay_steps")
        )
        if selection_step != 0:
            raise RuntimeError(
                "teacher record was not collected at the first block step"
            )
        sequence_length = int(record["x_at_block_start"].numel())
        if sequence_length > args.max_seq_len:
            raise RuntimeError(
                f"teacher record total length {sequence_length} exceeds "
                f"--max-seq-len {args.max_seq_len}; use a matching student limit"
            )
        x = record["x_at_block_start"].unsqueeze(0).to(device)
        cache = CustomCache(n_layers=L, device=device, keep_ratio=1.0)
        cache.layer_hidden_states = {}
        model(x, int(record["block_start"]), 1, cache)
        return cache.layer_hidden_states

    def step(record, train: bool):
        hidden = features(record)
        cand = record["candidate_indices"].to(device)
        bs = int(record["block_start"])
        blk = torch.arange(bs, bs + int(record["block_length"]), device=device)
        label = record["label_final_rowmax"].float().to(device)
        total, recalls = 0.0, []
        for l in range(L):
            h = hidden[l].float()
            pred = student.forward_layer(l, h, cand, head="score",
                                         block_indices=blk).squeeze(0)
            tgt = label[l]
            if not torch.isfinite(tgt).all() or tgt.sum() <= 0:
                continue
            # listwise: KL against the normalised label distribution. reduction has to
            # be "sum" — pred is 1-D, so "batchmean" would divide the KL by the
            # candidate count and shrink the term by 132x (mmlu) to 2528x
            # (gov_report), silently weighting domains by their prompt length.
            loss = args.lambda_list * F.kl_div(
                F.log_softmax(pred, -1), tgt / tgt.sum(), reduction="sum")
            # pairwise: random pairs anywhere in the range, to fix the ordering
            i = torch.randint(0, tgt.numel(), (args.pairs,), device=device)
            j = torch.randint(0, tgt.numel(), (args.pairs,), device=device)
            sign = torch.sign(tgt[i] - tgt[j])
            keep = sign != 0
            if keep.any():
                loss = loss + F.softplus(-sign[keep] * (pred[i][keep] - pred[j][keep])).mean()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            total += float(loss.detach())
            recalls.append(recall_grid(pred.detach(), tgt))
        return total / max(1, L), sum(recalls) / max(1, len(recalls))

    best = -1.0
    for epoch in range(args.epochs):
        student.train(); random.shuffle(train_shards)
        started, losses = time.time(), []
        for n, (_, path) in enumerate(train_shards):
            for record in load_shard(path)["blocks"]:
                losses.append(step(record, True)[0])
            if (n + 1) % 30 == 0:
                print(f"  epoch {epoch} {n+1}/{len(train_shards)} "
                      f"loss {sum(losses[-120:])/max(1,len(losses[-120:])):.4f}", flush=True)
        student.eval()
        per_ds = {}
        with torch.no_grad():
            for name, p in val_shards:
                for r in load_shard(p)["blocks"]:
                    per_ds.setdefault(name, []).append(step(r, False)[1])
        means = {k: sum(v) / max(1, len(v)) for k, v in per_ds.items()}
        score = sum(means.values()) / max(1, len(means))   # domain macro average
        detail = "  ".join(f"{k} {v:.3f}" for k, v in sorted(means.items()))
        print(f"epoch {epoch}: loss {sum(losses)/len(losses):.4f} | "
              f"val recall macro {score:.4f} [{detail}] | "
              f"{(time.time()-started)/60:.1f}min", flush=True)
        if score > best:
            best = score
            ckpt = out_dir / "checkpoint-best"
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in student.state_dict().items()},
                       ckpt / "pytorch_model.bin")
            json.dump({"layer_count": L, "hidden_dim": H, "proj_dim": args.proj_dim,
                       "mlp_dim": args.mlp_dim, "heads": ["score"],
                       "cache_selection_step": 0},
                      open(ckpt / "config.json", "w"))
            json.dump({"blk": student.block_proj_norms()},
                      open(out_dir / "block_proj_norms.json", "w"), indent=2)
            json.dump({"val_recall": score, "val_recall_per_dataset": means,
                       "epoch": epoch, "datasets": datasets},
                      open(out_dir / "best.json", "w"))
            print(f"  saved (best {best:.4f})", flush=True)
    print(f"done. best val recall {best:.4f} -> {out_dir}/checkpoint-best", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
