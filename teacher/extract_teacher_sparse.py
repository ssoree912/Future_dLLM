#!/usr/bin/env python
"""Teacher labels from a Sparse-dLLM decode: final x row-max.

The label is unchanged from `extract_teacher.py` -- for each block, run one
extra forward on the completed block and take the per-candidate maximum over
that block's attention rows,

    I_j = max_r a_rj      a_rj = softmax_j(q_r . k_j / sqrt d), head-averaged

so a candidate survives if *any* finished token needed it strongly. What
changes is where the finished block comes from: their `diffusion_generate`,
with their block schedule, their reveal rule and their decoding, rather than
our loop. Training on our trajectory and deploying in theirs would be the same
train/deploy mismatch the original extractor was written to avoid.

Reconstructing the selection-time input needs no history. Entering step 1 of
block b the sequence is: the finished blocks before it (immutable once
written), the single token step 0 confirmed at `block_start` (immutable too),
and mask everywhere after. All of that is readable off the final sequence.

    python teacher/extract_teacher_sparse.py --model model/Dream-v0-Instruct-7B \
        --dataset math5s --n-samples 500 --max-seq-len 2048
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_length import resolve as resolve_gen_length  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Dream checkpoint; labels are only valid for the model "
                        "that produced them")
    p.add_argument("--dataset", required=True)
    p.add_argument("--shard-root", default=str(REPO_ROOT / "artifacts" / "prompt_shards"))
    p.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "teacher"))
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--gen-length", type=int, default=None)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--max-prompt-len", type=int, default=None)
    # Dream's own recommended generation settings, from HKUNLP/Dream's README
    # (temperature 0.2, top_p 0.95); Sparse-dLLM follows them, so matching them
    # keeps the baseline and us on identical decoding. Sampling is stochastic,
    # so --seed is what makes a rerun reproduce the same labels.
    p.add_argument("--alg", default="entropy")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--alg-temp", type=float, default=0.0)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--seed", type=int, default=2025)
    args = p.parse_args()

    if args.gen_length is None:
        args.gen_length, source = resolve_gen_length(args.dataset)
        print(f"gen_length {args.gen_length} from {source}", flush=True)
    if args.gen_length % args.block_length:
        raise SystemExit(f"gen_length {args.gen_length} is not a multiple of "
                         f"block_length {args.block_length}")
    available = args.max_seq_len - args.gen_length
    if available < 1:
        raise SystemExit(f"generation length {args.gen_length} leaves no prompt "
                         f"space within --max-seq-len {args.max_seq_len}")
    args.prompt_limit = min(available, args.max_prompt_len or available)
    return args


@torch.no_grad()
def collect(model, prompt_ids, args, mask_id: int, n_layers: int):
    from future_dllm.sparse_dllm_student import StudentScorerCache, set_scorer

    device = model.device
    prompt_ids = prompt_ids[-args.prompt_limit:].to(device).unsqueeze(0)
    P = prompt_ids.shape[1]
    G, B = args.gen_length, args.block_length
    n_blocks = G // B

    # 1. Their decode, start to finish. collect_pool keeps every candidate in
    #    candidate order so recorded columns line up with cache entries.
    set_scorer(None, collect_pool=True)
    out = model.diffusion_generate(
        prompt_ids, max_new_tokens=G, steps=G, block_length=B,
        temperature=args.temperature, top_p=args.top_p, alg=args.alg,
        alg_temp=args.alg_temp, return_dict_in_generate=True,
        output_history=False)
    final = out.sequences[0]                       # [P + G]

    records = []
    for block in range(n_blocks):
        bs, be = P + block * B, P + (block + 1) * B

        # 2. The state the selection was made from: everything before the block
        #    is finished, step 0's single reveal sits at bs, the rest is mask.
        x_at_block_start = torch.full((1, P + G), mask_id, dtype=torch.long,
                                      device=device)
        x_at_block_start[0, :bs + 1] = final[:bs + 1]

        cache = StudentScorerCache(
            n_layers=n_layers, device=device,
            kernel_size=args.kernel_size, keep_ratio=1.0)
        cache.collect_pool = True

        # 3. Build and cut the cache exactly as their step 1 does.
        model(position_offset=bs, cache_state=1, customcache=cache,
              input_ids=x_at_block_start)

        # 4. One forward on the *completed* block, recording what its rows
        #    attend to across the cached columns.
        cache.capture_rows = True
        model(position_offset=bs, cache_state=2, customcache=cache,
              input_ids=final[bs:be].unsqueeze(0))
        label = torch.stack([cache.pending_rows[layer].max(dim=0).values
                             for layer in range(n_layers)])
        cache.pending_rows.clear()
        cache.capture_rows = False

        candidates = torch.cat([torch.arange(bs, device=device),
                                torch.arange(be, P + G, device=device)])
        records.append({
            "block_index": block,
            "block_start": int(bs),
            "block_length": B,
            "window_start": int(bs),
            "window_length": int(B),
            "prompt_length": int(P),
            "gen_length": G,
            "steps_per_block": B,
            "backend": "sparse_dllm_dream",
            "decode": {"alg": args.alg, "temperature": args.temperature,
                       "top_p": args.top_p, "seed": args.seed},
            "x_at_block_start": x_at_block_start[0].cpu(),
            "candidate_indices": candidates.cpu(),
            "label_final_rowmax": label.to(torch.float16).cpu(),
        })
    return records


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from future_dllm.sparse_dllm_student import load_model

    model = load_model(args.model, block_length=args.block_length,
                       keep_ratio=1.0, kernel_size=args.kernel_size)
    n_layers = int(model.config.num_hidden_layers)
    mask_id = int(model.config.mask_token_id)
    print(f"backend=sparse_dllm_dream layers={n_layers} mask_id={mask_id} "
          f"alg={args.alg} temperature={args.temperature} top_p={args.top_p} "
          f"seed={args.seed}", flush=True)

    out = Path(args.output_root) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(f"{args.shard_root}/{args.dataset}/*.pt"))[: args.n_samples]
    if not shards:
        raise SystemExit(f"no prompt shards under {args.shard_root}/{args.dataset}")

    started, added = time.time(), 0
    for i, path in enumerate(shards):
        target = out / Path(path).name
        src = torch.load(path, map_location="cpu", weights_only=False)
        prompt_ids = src["prompt_input_ids"].to(torch.long)
        expected_prompt_len = min(prompt_ids.numel(), args.prompt_limit)

        if target.exists():
            saved = torch.load(target, map_location="cpu", weights_only=False)
            blocks = saved.get("blocks") or []
            if (blocks
                    and saved.get("backend") == "sparse_dllm_dream"
                    and all(int(r.get("prompt_length", -1)) == expected_prompt_len
                            and int(r.get("gen_length", -1)) == args.gen_length
                            for r in blocks)):
                continue
            print(f"rebuilding mismatched teacher shard: {target.name}", flush=True)
        added += 1

        records = collect(model, prompt_ids, args, mask_id, n_layers)
        payload = {"sample_id": src.get("sample_id"), "dataset": args.dataset,
                   "backend": "sparse_dllm_dream", "model": str(args.model),
                   "prompt_input_ids": prompt_ids,
                   "prompt_limit": args.prompt_limit,
                   "gen_length": args.gen_length,
                   "max_seq_len": args.max_seq_len,
                   "teacher_kind": "final_rowmax",
                   "blocks": records}
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(shards)}  {(time.time() - started) / (i + 1):.1f}s/sample",
                  flush=True)
    print(f"done: {len(list(out.glob('*.pt')))} shards total, {added} new -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
