"""Turn evaluation datasets into prompt shards the teacher extractor can read.

Only non-test splits are used so the teacher never sees an evaluation item.

카테고리별 대표 하나씩. 토픽보다 중요한 축은 캐시에서 프롬프트가 차지하는 비율
P/(P+32b)이라, 장문(0.9대)과 자기추론(0.2~0.3)의 양 끝을 모두 덮도록 골랐다.

  samsum_lb   장문 요약        LongBench 형식 train, 프롬프트 2048
  trec_lb     장문 few-shot 분류 LongBench 형식 train, 프롬프트 2048
  wiki2_lb    장문 멀티홉 QA    LongBench 형식 train, 프롬프트 ~1000
  math        수학 추론        hendrycks_math train (MATH500은 test에서 나온다)
  mbpp_full   코드            mbpp full train
  gsm8k/mmlu/mbpp  기존 버킷

LongBench 계열은 평가와 같이 chat template 없이 오른쪽 절단한다 (llada_wrapper의
drop_middle=False 경로와 동일). 나머지는 평가와 같이 chat template을 씌운다.
"""

from __future__ import annotations

import argparse, glob, json, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("FUTURE_DLLM_DATA", REPO_ROOT / "data"))

import torch
from transformers import AutoTokenizer

MATH_INSTRUCTION = ("Please reason step by step, and put your final answer within "
                    "\\boxed{}.")


def _drop_test_overlap(table, column, test_glob):
    """MMLU and MBPP each repeat a few items across splits upstream - 43 of MMLU's
    1816 validation+dev questions appear verbatim in test, and three MBPP task
    descriptions do under a different task_id. Drop them before sampling."""
    import pandas as pd, re
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(str(test_glob)))]
    if not frames:
        return table
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
    seen = {norm(v) for v in pd.concat(frames)[column]}
    keep = [norm(v) not in seen for v in table[column]]
    dropped = len(keep) - sum(keep)
    if dropped:
        print(f"  dropped {dropped} rows that also appear in test", flush=True)
    return table[keep]


def _balanced(table, columns, limit, seed=0):
    """Take `limit` rows spread evenly over the sub-categories.

    Several of these datasets arrive sorted by category - MMLU's validation split
    is in subject order - so head(limit) picks the first few subjects and nothing
    else. Taking one row from each group in turn covers all of them, and a group
    that runs out simply drops out of the rotation.
    """
    import pandas as pd
    if not columns or not all(c in table.columns for c in columns):
        return table.sample(frac=1.0, random_state=seed).head(limit)
    groups = [g.sample(frac=1.0, random_state=seed).to_dict("records")
              for _, g in table.groupby(list(columns), sort=True)]
    out, i = [], 0
    while len(out) < limit and any(groups):
        g = groups[i % len(groups)]
        if g:
            out.append(g.pop())
        i += 1
        if i % max(1, len(groups)) == 0:
            groups = [g for g in groups if g]
            i = 0
            if not groups:
                break
    return pd.DataFrame(out[:limit])


def samsum(limit):
    """LongBench-style single dialogue, byte-for-byte the format the first 300
    teacher shards were built with (no chat template, no few-shot block)."""
    rows = []
    with open(DATA / "train/samsum/a100_source_train.jsonl") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            rows.append(json.loads(line))
    return [(f"{r['_id']}-1",
             "Summarize the dialogue into a few short sentences. "
             "The following are some examples.\n\n"
             f"Dialogue: {r['context']}\n\n"
             "Summarize the dialogue into a short summary.")
            for r in rows]


def gsm8k(limit):
    """카테고리 축이 없어서 균등성은 셔플 샘플링이 전부다."""
    import pandas as pd
    table = _balanced(pd.read_parquet(DATA / "train/gsm8k/train.parquet"), [], limit)
    return [(f"gsm8k-{i}", f"{row.question}\n\n{MATH_INSTRUCTION}")
            for i, row in enumerate(table.itertuples())]


def mmlu(limit):
    import pandas as pd
    frames = [pd.read_parquet(f) for split in ("validation", "dev")
              for f in glob.glob(str(DATA / f"eval/mmlu/all/{split}-*.parquet"))]
    table = _drop_test_overlap(pd.concat(frames), "question",
                               DATA / "eval/mmlu/all/test-*.parquet")
    table = _balanced(table, ["subject"], limit)               # 57 subjects
    out = []
    for i, row in enumerate(table.itertuples()):
        options = "\n".join(f"{chr(65 + j)}. {c}" for j, c in enumerate(row.choices))
        out.append((f"mmlu-{i}",
                    f"The following is a multiple choice question about "
                    f"{row.subject.replace('_', ' ')}.\n\n{row.question}\n{options}\n\n"
                    f"Answer with the letter of the correct option.\nAnswer:"))
    return out


def mbpp(limit):
    import pandas as pd
    frames = [pd.read_parquet(f) for split in ("validation", "prompt")
              for f in glob.glob(str(DATA / f"eval/mbpp/full/{split}-*.parquet"))]
    table = _balanced(pd.concat(frames), [], limit)
    return [(f"mbpp-{i}",
             f"You are an expert Python programmer. {row.text}\n"
             f"Your code should pass these tests:\n" + "\n".join(row.test_list) + "\n")
            for i, row in enumerate(table.itertuples())]


def _longbench(path, limit):
    """LongBench train 파일은 평가 parquet과 필드가 바이트 단위로 같다.
    평가에서 쓰는 프롬프트는 context + "\n\n" + question 이다."""
    rows, out = [], []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            rows.append(json.loads(line))
    for i, r in enumerate(rows):
        out.append((f"{Path(path).parent.name}-{i}", f"{r['context']}\n\n{r['question']}"))
    return out


def samsum_lb(limit):
    return _longbench(DATA / "train/samsum/train.jsonl", limit)


def trec_lb(limit):
    return _longbench(DATA / "train/trec/train.jsonl", limit)


def wiki2_lb(limit):
    return _longbench(DATA / "train/2wikimqa/train.jsonl", limit)


def musique(limit):
    """LongBench musique 평가 프롬프트를 원본 train 문단으로 재현. gen 32 = 1블록."""
    import random
    rows = [json.loads(l) for l in open(DATA / "train/musique/musique_ans_v1.0_train.jsonl")]
    random.Random(0).shuffle(rows)
    out = []
    for i, r in enumerate(rows[:limit]):
        ctx = "\n".join(f"Passage {j+1}:\n{p['title']}\n{p['paragraph_text']}"
                        for j, p in enumerate(r["paragraphs"]))
        out.append((f"musique-{i}",
                    "Answer the question based on the given passages. Only give me the "
                    "answer and do not output any other words.\n\nThe following are given "
                    f"passages.\n{ctx}\n\nAnswer the question based on the given passages. "
                    "Only give me the answer and do not output any other words.\n\n"
                    f"Question: {r['question']}\nAnswer:"))
    return out


def qasper(limit):
    """LongBench qasper 평가 프롬프트를 원본 train 논문으로 재현: 논문당 첫 질문 하나.
    gen 128 = 4블록."""
    import pandas as pd
    table = pd.read_parquet(DATA / "train/qasper/train.parquet")
    table = table.sample(frac=1.0, random_state=0)
    out = []
    for row in table.itertuples():
        if len(out) >= limit:
            break
        qs = list(row.qas["question"])
        if not qs:
            continue
        sections = [f"{n}\n" + "\n".join(ps) for n, ps
                    in zip(row.full_text["section_name"], row.full_text["paragraphs"])]
        article = f"{row.title}\n{row.abstract}\n" + "\n".join(sections)
        out.append((f"qasper-{len(out)}",
                    "You are given a scientific article and a question. Answer the question "
                    "as concisely as you can, using a single phrase or sentence if possible. "
                    "If the question cannot be answered based on the information in the "
                    "article, write \"unanswerable\". If the question is a yes/no question, "
                    "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
                    f"explanation.\n\nArticle: {article}\n\n Answer the question based on "
                    "the above article as concisely as you can, using a single phrase or "
                    "sentence if possible. If the question cannot be answered based on the "
                    "information in the article, write \"unanswerable\". If the question is "
                    "a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not "
                    f"provide any explanation.\n\nQuestion: {qs[0]}\n\nAnswer:"))
    return out


def gov_report(limit):
    """LongBench gov_report 평가 프롬프트를 원본 train 보고서로 재현. gen 512 = 16블록."""
    import pandas as pd
    table = pd.read_parquet(DATA / "train/gov_report/train.parquet")
    table = table.sample(frac=1.0, random_state=0).head(limit)
    return [(f"gov_report-{i}",
             "You are given a report by a government agency. Write a one-page summary "
             f"of the report.\n\nReport:\n{row.report}\n\nNow, write a one-page summary "
             "of the report.\n\nSummary:")
            for i, row in enumerate(table.itertuples())]


def multi_news(limit):
    """LongBench multi_news 평가 프롬프트를 원본 train 문서로 재현. gen 512 = 16블록."""
    import pandas as pd
    table = pd.read_parquet(DATA / "train/multi_news/train.parquet")
    table = table.sample(frac=1.0, random_state=0).head(limit)
    return [(f"multinews-{i}",
             "You are given several news passages. Write a one-page summary of all news. "
             f"\n\nNews:\n{row.document}\n\nNow, write a one-page summary of all the news."
             "\n\nSummary:")
            for i, row in enumerate(table.itertuples())]


def math(limit):
    """MATH500은 test에서 뽑은 것이라 train split은 겹치지 않는다."""
    import pandas as pd
    frames = [pd.read_parquet(f) for f
              in sorted(glob.glob(str(DATA / "train/hendrycks_math/*/train-*.parquet")))]
    table = _balanced(pd.concat(frames), ["type", "level"], limit)  # 7 x 5 cells
    return [(f"math-{i}", f"{row.problem}\n\n{MATH_INSTRUCTION}")
            for i, row in enumerate(table.itertuples())]


MATH5S_TRAIN = ["Prealgebra", "Algebra", "Geometry", "Number Theory", "Precalculus"]
MATH5S_HELDOUT = ["Intermediate Algebra", "Counting & Probability"]


def _math_subjects(subjects, limit, prefix):
    """[asy] 그림 코드가 든 문제는 어느 과목에서든 먼저 거른 뒤 과목을 나눈다.
    held-out 2과목(근거리 intermediate_algebra, 원거리 counting_and_probability)은
    recall 평가 전용이라 학습 5과목과 같은 필터·균형으로 뽑되 따로 저장한다."""
    import pandas as pd
    frames = [pd.read_parquet(f) for f
              in sorted(glob.glob(str(DATA / "train/hendrycks_math/*/train-*.parquet")))]
    table = pd.concat(frames)
    table = table[~table["problem"].str.contains(r"\[asy\]", regex=True)]
    table = table[table["type"].isin(subjects)]
    table = _balanced(table, ["type", "level"], limit)
    return [(f"{prefix}-{i}", f"{row.problem}\n\n{MATH_INSTRUCTION}")
            for i, row in enumerate(table.itertuples())]


def math5s(limit):
    return _math_subjects(MATH5S_TRAIN, limit, "math5s")


def math_ho_near(limit):
    """근거리 held-out: algebra 계열에서 난이도만 올라간 과목."""
    return _math_subjects(["Intermediate Algebra"], limit, "mathhonear")


def math_ho_far(limit):
    """원거리 held-out: 조합론, 풀이 스타일이 학습 5과목과 이질적."""
    return _math_subjects(["Counting & Probability"], limit, "mathhofar")


def repobench_p(limit):
    """LongBench repobench-p 평가 프롬프트를 원본 train으로 재현: cross-file
    스니펫(path+snippet) 뒤에 in-file 코드, 다음 줄 완성. gen 64 = 2블록."""
    import pandas as pd
    table = pd.read_parquet(DATA / "train/repobench-p/cross_file_first.parquet")
    table = table.sample(frac=1.0, random_state=0).head(limit)
    out = []
    for i, row in enumerate(table.itertuples()):
        ctx = "".join(f"{s['path']}\n{s['snippet']}\n" for s in row.context)
        code = row.cropped_code if row.cropped_code.endswith("\n") else row.cropped_code + "\n"
        out.append((f"repobenchp-{i}",
                    f"Please complete the code given below. \n{ctx}{code}"
                    "Next line of code:\n"))
    return out


def mbpp_full(limit):
    import pandas as pd
    table = _drop_test_overlap(
        pd.read_parquet(DATA / "train/mbpp/full/train-00000-of-00001.parquet"),
        "text", DATA / "eval/mbpp/full/test-*.parquet")
    table = _balanced(table, [], limit)
    return [(f"mbppfull-{i}",
             f"You are an expert Python programmer. {row.text}\n"
             f"Your code should pass these tests:\n" + "\n".join(row.test_list) + "\n")
            for i, row in enumerate(table.itertuples())]


BUILDERS = {"samsum": samsum, "gsm8k": gsm8k, "mmlu": mmlu, "mbpp": mbpp,
            "samsum_lb": samsum_lb, "trec_lb": trec_lb, "wiki2_lb": wiki2_lb,
            "math": math, "mbpp_full": mbpp_full,
            "musique": musique, "qasper": qasper, "gov_report": gov_report,
            "multi_news": multi_news,
            "math5s": math5s, "math_ho_near": math_ho_near,
            "math_ho_far": math_ho_far, "repobench_p": repobench_p}

# LongBench 계열은 평가 경로가 chat template을 쓰지 않는다.
RAW_TEXT = {"samsum", "samsum_lb", "trec_lb", "wiki2_lb",
            "musique", "qasper", "gov_report", "multi_news", "repobench_p"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(BUILDERS), required=True)
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--model", default=str(REPO_ROOT / "model" / "LLaDA-8B-Instruct"))
    p.add_argument("--chat-template", type=int, default=-1,
                   help="-1이면 데이터셋 기본값(LongBench 계열은 끔)")
    p.add_argument("--out-root", default=str(REPO_ROOT / "artifacts" / "prompt_shards"))
    args = p.parse_args()

    chat = (args.dataset not in RAW_TEXT) if args.chat_template < 0 else bool(args.chat_template)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # 평가(lm_eval_model)는 프롬프트 뒤쪽 2048을 유지하므로 teacher도 같은 쪽을 본다.
    tok.truncation_side = "left"
    out = Path(args.out_root) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    added = 0
    for sid, text in BUILDERS[args.dataset](args.limit):
        if (out / f"{sid}.pt").exists():      # raising --limit only adds new samples
            continue
        added += 1
        if chat:
            text = tok.apply_chat_template([{"role": "user", "content": text}],
                                           add_generation_prompt=True, tokenize=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=not chat,
                  truncation=True, max_length=2048).input_ids[0]
        torch.save({"sample_id": sid, "dataset": args.dataset,
                    "prompt_input_ids": ids.to(torch.long)}, out / f"{sid}.pt")
    n = len(list(out.glob("*.pt")))
    print(f"{args.dataset}: {n} shards total, {added} new "
          f"(chat_template={chat}) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
