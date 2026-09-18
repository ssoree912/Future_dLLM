#!/usr/bin/env python
"""Create one deterministic random evaluation snapshot shared by all methods."""
import argparse, json, random, shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

def sample_parquet(srcs, dst, n, rng):
    tables = [pq.read_table(p) for p in srcs]
    table = pa.concat_tables(tables, promote_options='default') if len(tables) > 1 else tables[0]
    n = min(n, table.num_rows)
    idx = rng.sample(range(table.num_rows), n)
    pq.write_table(table.take(pa.array(idx)), dst)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--output', required=True); ap.add_argument('--limit', type=int, default=100); ap.add_argument('--seed', type=int, default=20260917)
    a = ap.parse_args(); src, out = Path(a.source), Path(a.output)
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(src, out)
    rng = random.Random(a.seed)
    specs = {
      'gsm8k': ('eval/gsm8k', ['test.parquet']),
      'humaneval': ('eval/humaneval', ['test.parquet']),
      'math': ('eval/hendrycks_math', [f for f in []]),
    }
    for name, (rel, _) in specs.items():
        d = out / rel
        if name == 'math':
            files = sorted(d.glob('*-test.parquet'))
        else:
            files = [d / 'test.parquet']
        tmp = d / '.sampled.parquet'
        sample_parquet(files, tmp, a.limit, rng)
        for f in files: f.unlink()
        tmp.rename(d / 'sampled-test.parquet' if name == 'math' else d / 'test.parquet')
    lb = out / 'longbench' / 'data'
    for f in lb.glob('*.jsonl'):
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if len(rows) > a.limit:
            rows = rng.sample(rows, a.limit)
        f.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in rows) + '\n')
    print(out)

if __name__ == '__main__': main()
