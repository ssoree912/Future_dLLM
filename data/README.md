# Datasets

각 데이터셋의 출처와 다운로드 위치입니다. 데이터는 저장소 안 `data/` 아래에 저장됩니다:
`data/eval/<name>/` · `data/train/<name>/` · `data/longbench/data/`.


## 다운로드

```bash
python scripts/download_data.py                                   # 전체 (eval + train + longbench)
python scripts/download_data.py --parts eval                      # 평가 데이터만
python scripts/download_data.py --parts humaneval                 # HumanEval 만
python scripts/download_data.py --parts train                     # 학습 데이터만
python scripts/download_data.py --parts longbench                 # LongBench 만
```


## 평가 — `data/eval/<name>/`

| 데이터셋 | split / 행 수 | 출처 |
|---|---|---|
| gsm8k | test 1,319 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` |
| math500 | test 500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| gpqa | main 448 | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) `gpqa_main` |
| humaneval | test 164 | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) |
| hendrycks_math | test 5,000 (7과목) | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) |
| mmlu | test 14,042 · validation 1,531 · dev 285 (+과목별 분할) | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) `all` |
| arc_challenge | validation 299 · test 1,172 | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) `ARC-Challenge` |
| piqa | validation 1,838 | [physicaliqa-train-dev.zip](https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip) (dev split, `ybisk/piqa` 의 원본) |
| mbpp | full: validation 90 · prompt 10 · test 500 / sanitized: test | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) — 

## LongBench — `data/longbench/data/`

| | 행 수 | 출처 |
|---|---|---|
| 16개 태스크 jsonl | 각 200행 (lcc / repobench-p 는 500행) | [zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench) `data.zip` (test) |

## 학습 — `data/train/<name>/`

| 데이터셋 | split / 행 수 | 출처 |
|---|---|---|
| gsm8k | train 7,473 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` |
| hendrycks_math | train 7,500 (7과목) | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) |
| mbpp | full: train 374 / sanitized: train | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| gov_report | train 17,517 | [ccdv/govreport-summarization](https://huggingface.co/datasets/ccdv/govreport-summarization) |
| multi_news | train 44,972 | [alexfabbri/multi_news](https://huggingface.co/datasets/alexfabbri/multi_news) (`data/train.src.cleaned` · `data/train.tgt`) |
| musique | train 19,938 | [dgslibisey/MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue) (`musique_ans_v1.0_train.jsonl`) |




