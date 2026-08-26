# artifacts

Not tracked by git — a checkpoint is 305 MB.

```
prompt_shards/<dataset>/   prompts the teacher reads, built from non-test splits
teacher/<dataset>/         final × row-max labels, one shard per sample
ckpts/<name>/              trained scorers; checkpoint-best/ is what deployment loads
```

**Labels accumulate.** Both `build_prompt_shards.py` and `extract_teacher.py` skip
a sample whose shard already exists, so raising `--limit` / `--n-samples` extracts
only what is missing and an interrupted run resumes where it stopped. Each prints
how many are new.

The 300 samsum prompt shards predate the builder and carry the ids of their source
rows; `--dataset samsum` draws from `a100_source_train.jsonl` in file order, so it
adds different samples rather than reproducing these. Every other dataset rebuilds
identically.

**Checkpoint names** say what separates one scorer from another — how many domains,
how many samples of each, epochs, learning rate — with a hash of the domain names,
since four names do not fit in a directory name:

```
1ds_300_e6_lr2e-4_6a5fc6                  samsum ×300
4ds_300-300-300-100_e6_lr2e-4_c10fe8      samsum, gsm8k, mmlu ×300, mbpp ×100
```

`meta.json` inside each checkpoint lists the domains in full, plus seed, model
dimensions and the teacher roots it was trained from.

Everything produced before the step-1 snapshot fix lives in
`../v1_results/sparse_future/`.
