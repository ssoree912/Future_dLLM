#!/usr/bin/env python
"""Download datasets into the local layout used by training and evaluation.

The runtime scripts default to offline mode. Run this once in an online
environment, then share the resulting data directory together with the repo.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
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

HF_CACHE: Path | None = None


def _load(repo: str, config: str | None = None, **kwargs) -> DatasetDict | Dataset:
    if HF_CACHE is not None:
        kwargs.setdefault("cache_dir", str(HF_CACHE))
    return load_dataset(repo, config, **kwargs) if config else load_dataset(repo, **kwargs)


def _write_parquet(ds: Dataset, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(path))
    return len(ds)


def _write_jsonl(ds: Dataset, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in ds:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
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

    ds = _load("ybisk/piqa")
    rows = {"validation.parquet": _write_parquet(ds["validation"], root / "eval/piqa/validation.parquet")}
    _write_source(root / "eval/piqa/SOURCE.json", source_dataset="ybisk/piqa",
                  source_config=None, rows=rows)


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

    ds = _load("allenai/qasper")
    rows = {
        "train.parquet": _write_parquet(ds["train"], root / "train/qasper/train.parquet"),
        "validation.parquet": _write_parquet(ds["validation"], root / "train/qasper/validation.parquet"),
    }
    _write_source(root / "train/qasper/SOURCE.json", source_dataset="allenai/qasper", rows=rows)

    ds = _load("deepmind/narrativeqa")
    rows = {"train.parquet": _write_parquet(ds["train"], root / "train/narrativeqa/train.parquet")}
    _write_source(root / "train/narrativeqa/SOURCE.json", source_dataset="deepmind/narrativeqa", rows=rows)

    ds = _load("alexfabbri/multi_news")
    rows = {"train.parquet": _write_parquet(ds["train"], root / "train/multi_news/train.parquet")}
    _write_source(root / "train/multi_news/SOURCE.json", source_dataset="alexfabbri/multi_news", rows=rows)

    ds = _load("mandarjoshi/trivia_qa", "rc.web")
    rows = {"train.parquet": _write_parquet(ds["train"], root / "train/triviaqa/train.parquet")}
    _write_source(root / "train/triviaqa/SOURCE.json", source_dataset="mandarjoshi/trivia_qa",
                  source_config="rc.web", rows=rows)

    ds = _load("cais/mmlu", "all")
    rows = {"auxiliary_train.parquet": _write_parquet(
        ds["auxiliary_train"], root / "train/mmlu/auxiliary_train.parquet")}
    _write_source(root / "train/mmlu/SOURCE.json", source_dataset="cais/mmlu",
                  source_config="all", rows=rows)

    ds = _load("allenai/ai2_arc", "ARC-Challenge")
    rows = {"train.parquet": _write_parquet(ds["train"], root / "train/arc_challenge/train.parquet")}
    _write_source(root / "train/arc_challenge/SOURCE.json", source_dataset="allenai/ai2_arc",
                  source_config="ARC-Challenge", rows=rows)

    ds = _load("ybisk/piqa")
    rows = {"train.parquet": _write_parquet(ds["train"], root / "train/piqa/train.parquet")}
    _write_source(root / "train/piqa/SOURCE.json", source_dataset="ybisk/piqa", rows=rows)

    raw = hf_hub_download("dgslibisey/MuSiQue", "musique_ans_v1.0_train.jsonl", repo_type="dataset")
    out = root / "train/musique/musique_ans_v1.0_train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw, out)
    rows = {"train": sum(1 for _ in out.open())}
    _write_source(root / "train/musique/SOURCE.json", source_dataset="dgslibisey/MuSiQue",
                  source_file="musique_ans_v1.0_train.jsonl", rows=rows)

    raw = hf_hub_download("xanhho/2WikiMultihopQA", "train.parquet", repo_type="dataset")
    out = root / "train/2wikimqa/train.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw, out)
    _write_source(root / "train/2wikimqa/SOURCE.json",
                  source_dataset="xanhho/2WikiMultihopQA",
                  rows={"train.parquet": len(pd.read_parquet(out))})

    frames = []
    files = [
        "data/cross_file_first-00000-of-00002-baebda7f3a6e980a.parquet",
        "data/cross_file_first-00001-of-00002-5780ed62c5162a3e.parquet",
    ]
    for filename in files:
        frames.append(pd.read_parquet(hf_hub_download(
            "tianyang/repobench_python_v1.1", filename, repo_type="dataset")))
    out = root / "train/repobench-p/cross_file_first.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(out)
    _write_source(root / "train/repobench-p/SOURCE.json",
                  source_dataset="tianyang/repobench_python_v1.1",
                  source_files=files, rows={"cross_file_first.parquet": len(table)})


def download_longbench(root: Path) -> None:
    rows = {}
    for task in LONGBENCH_TASKS:
        ds = _load("zai-org/LongBench", task, split="test")
        rows[f"{task}.jsonl"] = _write_jsonl(ds, root / f"longbench/data/{task}.jsonl")
    _write_source(root / "longbench/SOURCE.json", source_dataset="zai-org/LongBench",
                  split="test", rows=rows)


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
