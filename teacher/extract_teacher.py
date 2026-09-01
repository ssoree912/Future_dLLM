"""Teacher labels from the finished answer: final x row-max.

For each block the extractor lets the block fill in with the full cache, then
runs one extra forward on the completed block and reads what those answer tokens
attend to. The label is the per-candidate maximum over the block's rows

    I_j = max_r a_rj          a_rj = softmax_j (q_r . k_j / sqrt(d)), head-averaged

so a candidate survives if *any* finished token needed it strongly. Taking the
maximum rather than the sum is what makes the label usable: summing averages away
the one token that depended on a given cache entry.

Stored per (sample, block): the label [n_layers, n_candidates], the exact model
input at the block's step-1 so the scorer's features can be replayed at training
time without keeping hidden states, and the candidate index set.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_length import resolve as resolve_gen_length

REPO_ROOT = Path(__file__).resolve().parent.parent
MASK_ID = 126336


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(REPO_ROOT / "model" / "LLaDA-8B-Instruct"))
    p.add_argument("--dataset", required=True,
                   help="prompt shard directory name, e.g. samsum / gsm8k / mmlu")
    p.add_argument("--shard-root", default=str(REPO_ROOT / "artifacts" / "prompt_shards"))
    p.add_argument("--output-root", default=str(REPO_ROOT / "artifacts" / "teacher"))
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--gen-length", type=int, default=None,
                   help="default: the eval task's max_gen_toks, see teacher/gen_length.py")
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="maximum prompt + generation length (default: 4096)")
    p.add_argument("--max-prompt-len", type=int, default=None,
                   help="optional stricter prompt-only cap")
    p.add_argument(
        "--save-attention-rows",
        action="store_true",
        help="analysis only: also save completed-block attention before row-max",
    )
    args = p.parse_args()
    if args.gen_length is None:
        args.gen_length, source = resolve_gen_length(args.dataset)
        print(f"gen_length {args.gen_length} from {source}", flush=True)
    if args.gen_length < 1:
        raise SystemExit("--gen-length must be positive")
    if args.block_length < 1:
        raise SystemExit("--block-length must be positive")
    if args.gen_length % args.block_length:
        raise SystemExit(f"gen_length {args.gen_length} is not a multiple of "
                         f"block_length {args.block_length}")
    if args.max_seq_len < 1:
        raise SystemExit("--max-seq-len must be positive")
    available = args.max_seq_len - args.gen_length
    if available < 1:
        raise SystemExit(
            f"generation length {args.gen_length} leaves no prompt space within "
            f"--max-seq-len {args.max_seq_len}"
        )
    if args.max_prompt_len is not None and args.max_prompt_len < 1:
        raise SystemExit("--max-prompt-len must be positive")
    args.prompt_limit = min(available, args.max_prompt_len or available)
    return args


@torch.no_grad()
def collect(model, prompt_ids, args):
    from future_dllm import CustomCache, add_gumbel_noise, get_num_transfer_tokens

    device = model.device
    # 평가와 같은 left-truncate: 총 길이 안에서 생성 공간을 먼저 확보한다.
    prompt_ids = prompt_ids[-args.prompt_limit:].to(device).unsqueeze(0)
    P = prompt_ids.shape[1]
    G, B = args.gen_length, args.block_length
    n_blocks = G // B
    S = args.gen_length // n_blocks          # steps per block == block length
    L = model.config.n_layers

    x = torch.full((1, P + G), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    records = []

    for block in range(n_blocks):
        cache = CustomCache(n_layers=L, device=device, keep_ratio=1.0)
        cache.collect_pool = True             # keep the whole pool, in candidate order
        bs, be = P + block * B, P + (block + 1) * B
        ntt = get_num_transfer_tokens(x[:, bs:be] == MASK_ID, S)

        def step(i):
            state = 2 if i > 1 else i
            inp = x if state != 2 else x[:, bs:be]
            m = (inp == MASK_ID)
            logits = model(inp, bs, state, cache).logits
            x0 = torch.argmax(add_gumbel_noise(logits, 0.0), dim=-1)
            conf = torch.squeeze(torch.gather(F.softmax(logits, -1), -1,
                                              x0.unsqueeze(-1)), -1)
            tgt = x if state != 2 else x[:, bs:be]
            if state != 2:
                conf[:, be:] = -float("inf")
            x0 = torch.where(m, x0, tgt)
            conf = torch.where(m, conf, torch.full_like(conf, -float("inf")))
            keep = torch.topk(conf[0], k=ntt[0, i]).indices
            tgt[0, keep] = x0[0, keep]

        # Step 1 runs forward, prunes, and only then reveals, so the state the
        # scorer sees at selection time is the one *entering* step 1 - after
        # step 0's reveal, before step 1's. Cloning after step(1) would hand
        # training one revealed token more than deployment ever has.
        step(0)
        x_at_block_start = x.clone()          # the scorer's input at selection time
        step(1)
        for i in range(2, S):
            step(i)

        # One more forward on the completed block: all rows are real tokens now.
        cache.capture_rows = True
        step(S - 1)
        save_attention_rows = getattr(args, "save_attention_rows", False)
        if save_attention_rows:
            future_attention_rows = torch.stack(
                [cache.pending_rows[l].clone() for l in range(L)]
            )
            label = future_attention_rows.max(dim=1).values
        else:
            # Keep the ordinary teacher path at its original memory footprint.
            label = torch.stack([
                cache.pending_rows[layer].max(dim=0).values
                for layer in range(L)
            ])
        cache.pending_rows.clear()
        cache.capture_rows = False

        candidates = torch.cat([torch.arange(bs, device=device),
                                torch.arange(be, x.shape[1], device=device)])
        record = {
            "block_index": block,
            "block_start": int(bs),
            "block_length": B,
            "prompt_length": int(P),
            "gen_length": G,
            "steps_per_block": S,
            "x_at_block_start": x_at_block_start[0].cpu(),
            "candidate_indices": candidates.cpu(),
            "label_final_rowmax": label.to(torch.float16).cpu(),
        }
        if save_attention_rows:
            # Deliberately analysis-only: [layer, answer token, candidate] is
            # much larger than the row-max target used to train the student.
            record["future_attention_rows"] = (
                future_attention_rows.to(torch.float16).cpu()
            )
            record["completed_block_ids"] = x[0, bs:be].cpu()
        records.append(record)
    return records


def main():
    args = parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    from transformers import AutoConfig
    from future_dllm import LLaDAModelLM

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    native_limit = int(getattr(cfg, "max_sequence_length", args.max_seq_len))
    if args.max_seq_len > native_limit:
        print(f"warning: max_seq_len={args.max_seq_len} exceeds the checkpoint's "
              f"trained context {native_limit}", flush=True)
    cfg.max_sequence_length = args.max_seq_len
    cfg.block_len, cfg.keep_ratio = args.block_length, 1.0
    model = LLaDAModelLM.from_pretrained(args.model, config=cfg, device_map="auto",
                                         torch_dtype=torch.bfloat16,
                                         trust_remote_code=True).eval()

    out = Path(args.output_root) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(f"{args.shard_root}/{args.dataset}/*.pt"))[: args.n_samples]
    if not shards:
        raise SystemExit(f"no prompt shards under {args.shard_root}/{args.dataset} "
                         f"- run teacher/build_prompt_shards.py first")
    started, added = time.time(), 0
    for i, path in enumerate(shards):
        target = out / Path(path).name
        src = torch.load(path, map_location="cpu", weights_only=False)
        prompt_ids = src["prompt_input_ids"].to(torch.long)
        expected_prompt_len = min(prompt_ids.numel(), args.prompt_limit)

        # Resume only when the saved labels match the requested sequence shape.
        # Old 2048 labels are therefore rebuilt after their prompt shards grow.
        if target.exists():
            saved = torch.load(target, map_location="cpu", weights_only=False)
            blocks = saved.get("blocks") or []
            if (blocks
                    and all(int(r.get("prompt_length", -1)) == expected_prompt_len
                            and int(r.get("gen_length", -1)) == args.gen_length
                            and r["x_at_block_start"].numel()
                            == expected_prompt_len + args.gen_length
                            and (not args.save_attention_rows
                                 or "future_attention_rows" in r)
                            for r in blocks)):
                continue
            print(f"rebuilding mismatched teacher shard: {target.name}", flush=True)
        added += 1
        records = collect(model, prompt_ids, args)
        payload = {"sample_id": src.get("sample_id"),
                   "dataset": args.dataset,
                   "prompt_input_ids": prompt_ids,
                   "prompt_limit": args.prompt_limit,
                   "gen_length": args.gen_length,
                   "max_seq_len": args.max_seq_len,
                   "teacher_kind": "final_rowmax",
                   "attention_rows_saved": args.save_attention_rows,
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
