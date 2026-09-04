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

Shared implementation -- not runnable on its own. Use the family entry points:

    teacher/extract_teacher_llada.py --model model/LLaDA-8B-Instruct --dataset ...
    teacher/extract_teacher_dream.py --model model/Dream-v0-Instruct-7B --dataset ...

Each pins one family and refuses a checkpoint from the other, so a run cannot
silently label with the wrong model. The label itself lives here and is shared:
one extra forward on the completed block, per-candidate row-max over the block's
rows, over a candidate pool that is the block's complement for both families
(and so the same pool the Sparse-dLLM baseline scores). LLaDA and Dream differ
only in the mask token, Dream's autoregressive logit shift, and whether step 0
seeds the block's first token -- see future_dllm/backends.py.
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


def parse_args(family, description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", required=True,
                   help=f"path to the {family} checkpoint to label with. Required "
                        f"on purpose: teacher labels are only meaningful for the "
                        f"model that produced them, so the run always says which "
                        f"one it used")
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

    # Refuse a checkpoint from the other family outright. Without this a Dream
    # path handed to the LLaDA entry point would just load as Dream and write
    # labels that look fine and belong to the wrong model.
    from future_dllm import detect_family
    found = detect_family(args.model)
    if found != family:
        raise SystemExit(
            f"this script extracts {family} labels, but --model {args.model} is "
            f"a {found} checkpoint. Use teacher/extract_teacher_{found}.py instead."
        )
    args.family = family

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
def collect(model, prompt_ids, args, backend):
    from future_dllm import CustomCache, add_gumbel_noise, get_num_transfer_tokens

    device = model.device
    # 평가와 같은 left-truncate: 총 길이 안에서 생성 공간을 먼저 확보한다.
    prompt_ids = prompt_ids[-args.prompt_limit:].to(device).unsqueeze(0)
    P = prompt_ids.shape[1]
    G, B = args.gen_length, args.block_length
    n_blocks = G // B
    S = args.gen_length // n_blocks          # steps per block == block length
    L, MASK_ID = backend.n_layers, backend.mask_id

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
            if backend.logit_shift:
                # Dream row r predicts token r+1; move each row onto the position
                # it describes, exactly as Dream's own generation_utils does.
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            x0 = torch.argmax(add_gumbel_noise(logits, 0.0), dim=-1)
            conf = torch.squeeze(torch.gather(F.softmax(logits, -1), -1,
                                              x0.unsqueeze(-1)), -1)
            tgt = x if state != 2 else x[:, bs:be]
            if state != 2:
                conf[:, be:] = -float("inf")
            x0 = torch.where(m, x0, tgt)
            conf = torch.where(m, conf, torch.full_like(conf, -float("inf")))
            if state == 0 and backend.seed_block_start:
                # The block's first token is only readable on a full forward
                # under the shift, so it is confirmed here. +inf heads this
                # step's top-k rather than adding a reveal: the budget is
                # unchanged, only which token it spends its first pick on.
                conf[:, bs] = float("inf")
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

        # Candidates are everything outside the block, in cache order -- exactly
        # what CustomCache.filter_cache keeps, so the label columns line up with
        # the entries the student will score.
        candidates = torch.cat([torch.arange(bs, device=device),
                                torch.arange(be, x.shape[1], device=device)])
        record = {
            "block_index": block,
            "block_start": int(bs),
            "block_length": B,
            # The window the cache was cut around; the student conditions on it.
            # Identical to the block for both backends today, and recorded
            # explicitly so a future backend that widens it stays readable.
            "window_start": int(bs),
            "window_length": int(B),
            "seed_block_start": bool(backend.seed_block_start),
            "backend": backend.name,
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


def run(family, description):
    """Entry point body, called by the two family scripts."""
    sys.path.insert(0, str(REPO_ROOT))
    args = parse_args(family, description)
    from future_dllm import load_model

    # keep_ratio=1.0: the teacher labels the whole candidate pool, so nothing is
    # evicted while it runs. load_model injects block_len and keep_ratio onto
    # the config, which is what the patched attention reads at selection time.
    model, backend = load_model(args.model, max_seq_len=args.max_seq_len,
                                block_length=args.block_length, keep_ratio=1.0)
    if args.max_seq_len > backend.native_max_seq_len:
        print(f"warning: max_seq_len={args.max_seq_len} exceeds the checkpoint's "
              f"trained context {backend.native_max_seq_len}", flush=True)
    print(f"backend={backend.name} layers={backend.n_layers} "
          f"mask_id={backend.mask_id} logit_shift={backend.logit_shift} "
          f"seed_block_start={backend.seed_block_start}", flush=True)

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
            # The backend check matters as much as the lengths: a Dream and a
            # LLaDA shard for the same sample can agree on every length and
            # still hold labels from different models over different vocabs.
            if (blocks
                    and saved.get("backend", backend.name) == backend.name
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
        records = collect(model, prompt_ids, args, backend)
        payload = {"sample_id": src.get("sample_id"),
                   "dataset": args.dataset,
                   "backend": backend.name,
                   "model": str(args.model),
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
    raise SystemExit(
        "extract_teacher.py holds the shared implementation and is not runnable "
        "on its own -- the family decides the mask token, the logit shift and "
        "the step-0 seed. Use teacher/extract_teacher_llada.py or "
        "teacher/extract_teacher_dream.py."
    )
