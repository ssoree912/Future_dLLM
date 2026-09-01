"""How well a trained scorer reproduces the teacher label, per domain.

Cheap check before spending a training run: recall@k against the stored labels,
averaged over k in {5,10,20,30,50}% - the same definition train_student.py uses
for checkpoint selection. A random baseline and a pure-recency baseline are
reported alongside, so the numbers have a floor to be read against.
"""
import argparse, glob, random, sys, torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def recall_grid(pred, target, ratios=(0.05, 0.1, 0.2, 0.3, 0.5)):
    out = []
    for r in ratios:
        k = max(1, int(target.numel() * r))
        a = set(torch.topk(pred, k).indices.tolist())
        b = set(torch.topk(target, k).indices.tolist())
        out.append(len(a & b) / k)
    return sum(out) / len(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(REPO_ROOT / "model" / "LLaDA-8B-Instruct"))
    p.add_argument("--student",
                   default=str(REPO_ROOT / "artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best"))
    p.add_argument("--root", default=str(REPO_ROOT / "artifacts"))
    p.add_argument("--datasets", default="samsum,gsm8k,mmlu,mbpp")
    p.add_argument("--shards", type=int, default=25)
    args = p.parse_args()

    torch.set_grad_enabled(False)
    random.seed(0)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from transformers import AutoConfig
    from future_dllm import LLaDAModelLM, CustomCache
    from future_dllm import load_prompt_utility_student

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cfg.block_len, cfg.keep_ratio = 32, 1.0
    model = LLaDAModelLM.from_pretrained(args.model, config=cfg, device_map="auto",
                                         torch_dtype=torch.bfloat16,
                                         trust_remote_code=True).eval()
    CustomCache.capture_layer_hidden_states = (
        lambda self, layer_id, hidden: self.layer_hidden_states.__setitem__(layer_id, hidden))
    device, L = model.device, model.config.n_layers

    student = load_prompt_utility_student(args.student, device).float().eval()

    def features(record):
        x = record["x_at_block_start"].unsqueeze(0).to(device)
        cache = CustomCache(n_layers=L, device=device, keep_ratio=1.0)
        cache.layer_hidden_states = {}
        model(x, int(record["block_start"]), 1, cache)
        return cache.layer_hidden_states

    print(f"{'dataset':10s} {'blocks':>7s} {'scorer':>8s} {'random':>8s} {'recency':>8s}")
    for ds in args.datasets.split(","):
        shards = sorted(glob.glob(f"{args.root}/teacher/{ds}/*.pt"))[:args.shards]
        if not shards:
            print(f"{ds:10s} (no shards)")
            continue
        stu, rnd, rec, n = [], [], [], 0
        for path in shards:
            for record in torch.load(path, map_location="cpu", weights_only=False)["blocks"]:
                hidden = features(record)
                cand = record["candidate_indices"].to(device)
                bs = int(record["block_start"])
                blk = torch.arange(bs, bs + int(record["block_length"]), device=device)
                label = record["label_final_rowmax"].float().to(device)
                C = label.size(-1)
                for l in range(L):
                    tgt = label[l]
                    if not torch.isfinite(tgt).all() or tgt.sum() <= 0:
                        continue
                    pred = student.forward_layer(l, hidden[l].float(), cand,
                                                 head="score",
                                                 block_indices=blk).squeeze(0)
                    stu.append(recall_grid(pred, tgt))
                    rnd.append(recall_grid(torch.randn(C, device=device), tgt))
                    rec.append(recall_grid(torch.arange(C, device=device).float(), tgt))
                n += 1
        m = lambda v: sum(v) / max(1, len(v))
        print(f"{ds:10s} {n:7d} {m(stu):8.3f} {m(rnd):8.3f} {m(rec):8.3f}", flush=True)


if __name__ == "__main__":
    main()
