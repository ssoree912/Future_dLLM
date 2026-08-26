#!/usr/bin/env python
"""Download datasets into the local layout used by training and evaluation.

The runtime scripts default to offline mode. Run this once in an online
environment, then share the resulting data directory together with the repo.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, load_dataset
from huggingface_hub import hf_hub_download


MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

LONGBENCH_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

# PIQA is still published on the Hub as a loader script, which datasets 5.0 refuses to run
# ("Dataset scripts are no longer supported"). It is read from its official release instead
# and rebuilt into the schema the loader script produced so downstream readers see the same
# columns.
PIQA_URL = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"

PIQA_FEATURES = Features({
    "goal": Value("string"),
    "sol1": Value("string"),
    "sol2": Value("string"),
    "label": ClassLabel(names=["0", "1"]),
})

MULTI_NEWS_REPO = "alexfabbri/multi_news"

HF_CACHE: Path | None = None


def _load(repo: str, config: str | None = None, **kwargs) -> DatasetDict | Dataset:
    if HF_CACHE is not None:
        kwargs.setdefault("cache_dir", str(HF_CACHE))
    return load_dataset(repo, config, **kwargs) if config else load_dataset(repo, **kwargs)


def _fetch(url: str, name: str) -> Path:
    """Download an upstream archive into the throwaway cache, once."""
    root = (HF_CACHE or Path(".hf_cache").resolve()) / "upstream"
    root.mkdir(parents=True, exist_ok=True)
    dest = root / name
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as src, tmp.open("wb") as fh:
            shutil.copyfileobj(src, fh)
        tmp.rename(dest)
    return dest


def _hf_download(repo: str, filename: str, repo_type: str = "dataset") -> str:
    """hf_hub_download, but fall back to curl when it fails.

    huggingface_hub streams over requests/HTTP1.1; on some networks the HF CDN resets
    the chunked transfer for large LFS files mid-stream (IncompleteRead at 0 bytes).
    curl negotiates HTTP/2 and resumes, so retry the resolved URL through it."""
    try:
        return hf_hub_download(repo, filename, repo_type=repo_type)
    except Exception as exc:  # network/protocol failure, not a missing file
        from huggingface_hub import hf_hub_url
        url = hf_hub_url(repo, filename, repo_type=repo_type)
        root = (HF_CACHE or Path(".hf_cache").resolve()) / "upstream"
        root.mkdir(parents=True, exist_ok=True)
        dest = root / filename.replace("/", "_")
        print(f"  hf_hub_download failed ({type(exc).__name__}); retrying via curl: {url}",
              flush=True)
        subprocess.run(
            ["curl", "-fSL", "-C", "-", "--retry", "5", "--retry-delay", "3",
             "--max-time", "1800", "-o", str(dest), url],
            check=True,
        )
        return str(dest)


def _piqa(split: str) -> Dataset:
    """PIQA from the official release: one jsonl of questions, one file of labels."""
    archive = _fetch(PIQA_URL, "physicaliqa-train-dev.zip")
    with zipfile.ZipFile(archive) as zf:
        rows = [json.loads(line) for line
                in zf.read(f"physicaliqa-train-dev/{split}.jsonl").decode("utf-8").splitlines()]
        labels = zf.read(f"physicaliqa-train-dev/{split}-labels.lst").decode("utf-8").split()
    if len(rows) != len(labels):
        raise ValueError(f"piqa {split}: {len(rows)} questions but {len(labels)} labels")
    return Dataset.from_dict(
        {
            "goal": [r["goal"] for r in rows],
            "sol1": [r["sol1"] for r in rows],
            "sol2": [r["sol2"] for r in rows],
            "label": [int(x) for x in labels],
        },
        features=PIQA_FEATURES,
    )


def _multi_news(split: str) -> Dataset:
    """Multi-News from the raw files the loader script read, with its newline fix."""
    src = hf_hub_download(MULTI_NEWS_REPO, f"data/{split}.src.cleaned", repo_type="dataset")
    tgt = hf_hub_download(MULTI_NEWS_REPO, f"data/{split}.tgt", repo_type="dataset")
    with open(src, encoding="utf-8") as src_f, open(tgt, encoding="utf-8") as tgt_f:
        pairs = [(s.strip().replace("NEWLINE_CHAR", "\n"), t.strip())
                 for s, t in zip(src_f, tgt_f)]
    return Dataset.from_dict({"document": [d for d, _ in pairs],
                              "summary": [s for _, s in pairs]})


def _write_parquet(ds: Dataset, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(path))
    return len(ds)


def _write_source(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("downloaded", date.today().isoformat())
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def download_eval(root: Path) -> None:
    ds = _load("openai/gsm8k", "main")
    rows = {"test.parquet": _write_parquet(ds["test"], root / "eval/gsm8k/test.parquet")}
    _write_source(root / "eval/gsm8k/SOURCE.json", source_dataset="openai/gsm8k",
                  source_config="main", rows=rows)

    ds = _load("HuggingFaceH4/MATH-500")
    rows = {"test.parquet": _write_parquet(ds["test"], root / "eval/math500/test.parquet")}
    _write_source(root / "eval/math500/SOURCE.json", source_dataset="HuggingFaceH4/MATH-500",
                  source_config=None, rows=rows)

    ds = _load("Idavidrein/gpqa", "gpqa_main")
    rows = {"main.parquet": _write_parquet(ds["train"], root / "eval/gpqa/main.parquet")}
    _write_source(root / "eval/gpqa/SOURCE.json", source_dataset="Idavidrein/gpqa",
                  source_config="gpqa_main", rows=rows)

    download_humaneval(root)

    rows = {}
    for subject in MATH_SUBJECTS:
        ds = _load("EleutherAI/hendrycks_math", subject)
        rows[f"{subject}-test.parquet"] = _write_parquet(
            ds["test"], root / f"eval/hendrycks_math/{subject}-test.parquet")
    _write_source(root / "eval/hendrycks_math/SOURCE.json",
                  source_dataset="EleutherAI/hendrycks_math", rows=rows)

    ds = _load("cais/mmlu", "all")
    rows = {}
    for split in ("dev", "validation", "test"):
        rows[f"all/{split}-00000-of-00001.parquet"] = _write_parquet(
            ds[split], root / f"eval/mmlu/all/{split}-00000-of-00001.parquet")
    # lm-eval scores MMLU as 57 per-subject tasks: the description names the subject and the
    # 5 shots come from that subject's dev rows. Split "all" by subject so those tasks can read
    # local files instead of the Hub.
    subjects = set()
    for split in ("dev", "test"):
        frame = ds[split].to_pandas()
        for subject, part in frame.groupby("subject", sort=False):
            path = root / f"eval/mmlu/{subject}/{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(path, index=False)
            subjects.add(subject)
    rows["<subject>/{dev,test}.parquet"] = f"{len(subjects)} subjects, split out of all/"
    _write_source(root / "eval/mmlu/SOURCE.json", source_dataset="cais/mmlu",
                  source_config="all", rows=rows)

    ds = _load("allenai/ai2_arc", "ARC-Challenge")
    rows = {
        "validation.parquet": _write_parquet(ds["validation"], root / "eval/arc_challenge/validation.parquet"),
        "test.parquet": _write_parquet(ds["test"], root / "eval/arc_challenge/test.parquet"),
    }
    _write_source(root / "eval/arc_challenge/SOURCE.json", source_dataset="allenai/ai2_arc",
                  source_config="ARC-Challenge", rows=rows)

    ds = _piqa("dev")
    rows = {"validation.parquet": _write_parquet(ds, root / "eval/piqa/validation.parquet")}
    _write_source(root / "eval/piqa/SOURCE.json", source_dataset="ybisk/piqa",
                  source_config=None, source_url=PIQA_URL, rows=rows)


def download_humaneval(root: Path) -> None:
    ds = _load("openai/openai_humaneval")
    rows = {"test.parquet": _write_parquet(ds["test"], root / "eval/humaneval/test.parquet")}
    _write_source(root / "eval/humaneval/SOURCE.json",
                  source_dataset="openai/openai_humaneval", source_config=None, rows=rows)


def download_train(root: Path) -> None:
    ds = _load("openai/gsm8k", "main")
    rows = {"train.parquet": _write_parquet(ds["train"], root / "train/gsm8k/train.parquet")}
    _write_source(root / "train/gsm8k/SOURCE.json", source_dataset="openai/gsm8k",
                  source_config="main", rows=rows)

    rows = {}
    for subject in MATH_SUBJECTS:
        ds = _load("EleutherAI/hendrycks_math", subject)
        rows[f"{subject}/train-00000-of-00001.parquet"] = _write_parquet(
            ds["train"], root / f"train/hendrycks_math/{subject}/train-00000-of-00001.parquet")
    _write_source(root / "train/hendrycks_math/SOURCE.json",
                  source_dataset="EleutherAI/hendrycks_math", rows=rows)

    ds = _load("google-research-datasets/mbpp", "full")
    rows = {"full/train-00000-of-00001.parquet": _write_parquet(
        ds["train"], root / "train/mbpp/full/train-00000-of-00001.parquet")}
    if "validation" in ds:
        rows["full/validation-00000-of-00001.parquet"] = _write_parquet(
            ds["validation"], root / "eval/mbpp/full/validation-00000-of-00001.parquet")
    if "prompt" in ds:
        rows["full/prompt-00000-of-00001.parquet"] = _write_parquet(
            ds["prompt"], root / "eval/mbpp/full/prompt-00000-of-00001.parquet")
    if "test" in ds:
        rows["full/test-00000-of-00001.parquet"] = _write_parquet(
            ds["test"], root / "eval/mbpp/full/test-00000-of-00001.parquet")
    ds = _load("google-research-datasets/mbpp", "sanitized")
    if "train" in ds:
        rows["sanitized/train-00000-of-00001.parquet"] = _write_parquet(
            ds["train"], root / "train/mbpp/sanitized/train-00000-of-00001.parquet")
    if "test" in ds:
        rows["sanitized/test-00000-of-00001.parquet"] = _write_parquet(
            ds["test"], root / "eval/mbpp/sanitized/test-00000-of-00001.parquet")
    _write_source(root / "train/mbpp/SOURCE.json",
                  source_dataset="google-research-datasets/mbpp", rows=rows)
    _write_source(root / "eval/mbpp/SOURCE.json",
                  source_dataset="google-research-datasets/mbpp", rows=rows)

    ds = _load("ccdv/govreport-summarization")
    rows = {"train.parquet": _write_parquet(ds["train"], root / "train/gov_report/train.parquet")}
    _write_source(root / "train/gov_report/SOURCE.json",
                  source_dataset="ccdv/govreport-summarization", rows=rows)

    ds = _multi_news("train")
    rows = {"train.parquet": _write_parquet(ds, root / "train/multi_news/train.parquet")}
    _write_source(root / "train/multi_news/SOURCE.json", source_dataset=MULTI_NEWS_REPO,
                  source_files=["data/train.src.cleaned", "data/train.tgt"], rows=rows)

    # Training set is limited to: gsm8k, hendrycks_math (math5s), mbpp, gov_report, multi_news,
    # musique. Others (mmlu/arc_challenge/piqa: single-letter or binary targets; qasper,
    # narrativeqa, triviaqa, 2wikimqa, repobench-p) are excluded from training. Some remain in
    # download_eval / download_longbench for evaluation only.

    raw = hf_hub_download("dgslibisey/MuSiQue", "musique_ans_v1.0_train.jsonl", repo_type="dataset")
    out = root / "train/musique/musique_ans_v1.0_train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw, out)
    rows = {"train": sum(1 for _ in out.open())}
    _write_source(root / "train/musique/SOURCE.json", source_dataset="dgslibisey/MuSiQue",
                  source_file="musique_ans_v1.0_train.jsonl", rows=rows)


def download_longbench(root: Path) -> None:
    """The repo's loader script cannot run under datasets 5.0, but the same repo also
    holds data.zip -- the official release, one jsonl per task, already in the layout
    the eval tasks read."""
    archive = _hf_download("zai-org/LongBench", "data.zip")
    dst = root / "longbench/data"
    dst.mkdir(parents=True, exist_ok=True)
    rows = {}
    with zipfile.ZipFile(archive) as zf:
        for task in LONGBENCH_TASKS:
            payload = zf.read(f"data/{task}.jsonl")
            (dst / f"{task}.jsonl").write_bytes(payload)
            rows[f"{task}.jsonl"] = sum(
                1 for line in payload.decode("utf-8").splitlines() if line.strip())
    _write_source(root / "longbench/SOURCE.json", source_dataset="zai-org/LongBench",
                  source_file="data.zip", split="test", rows=rows)


def main() -> int:
    global HF_CACHE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument(
        "--hf-cache",
        default=str(Path(__file__).resolve().parents[1] / ".hf_cache"),
        help="Hugging Face cache used while downloading; not needed afterwards",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("eval", "humaneval", "train", "longbench"),
        default=["eval", "train", "longbench"],
    )
    args = parser.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    HF_CACHE = Path(args.hf_cache).expanduser().resolve()
    if "eval" in args.parts:
        download_eval(root)
    elif "humaneval" in args.parts:
        download_humaneval(root)
    if "train" in args.parts:
        download_train(root)
    if "longbench" in args.parts:
        download_longbench(root)
    print(f"datasets written under {root}")
    print("Manual upstream-only sources still needed for exact LongBench-source training: SAMSum, TREC, QMSum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
