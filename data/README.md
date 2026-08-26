# Datasets

각 데이터셋 출처 
데이터는 저장소 안 `data/` 아래,
`data/eval/<name>/` · `data/train/<name>/` · `data/longbench/data/` 

## 다운로드

```bash
python scripts/download_data.py                                   # 전체
python scripts/download_data.py --parts eval                      # 평가 데이터만
python scripts/download_data.py --parts humaneval                 # HumanEval만
python scripts/download_data.py --parts train                     # 학습 데이터만
python scripts/download_data.py --parts longbench                 # LongBench 만
```


## 평가

| 데이터셋 | 행 수 | 출처 | 출처 확인 | `run_eval.sh` |
|---|---|---|---|---|
| MMLU | test 14,042 · validation 1,531 · dev 285 | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | row-count | `mmlu` |
| ARC-Challenge | test 1,172 · validation 299 | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) `ARC-Challenge` | recorded | `arc_c` |
| PIQA | validation 1,838 | [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa) | recorded | `piqa` |
| GPQA | main 448 | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | card | `gpqa` |
| GSM8K | test 1,319 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` | recorded | `gsm8k` |
| MATH | test 5,000 | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | row-count | `math` |
| MATH-500 | 500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | card | `math500` |
| HumanEval | 164 | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) | recorded | `humaneval` |
| LongBench | 16개 태스크, 각 200행 (lcc / repobench-p 는 500행) | [zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench) | recorded | 태스크 이름 |


### LongBench 원본 데이터셋(test)

| 태스크 | 행 수 | 출처 | 출처 확인 |
|---|---|---|---|
| samsum | train 14,732 | 공식 SAMSum 릴리스, [`corpus.7z`](https://arxiv.org/src/1911.12237v2/anc/corpus.7z) | card |
| 2wikimqa | train 167,454 | [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop), 미러 `xanhho/2WikiMultihopQA` | card |
| trec | `train_5500.label` | TREC 원본 | card |
| triviaqa | 64,916 | [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa) `rc.web` | card |
| narrativeqa | train 32,747 | [deepmind/narrativeqa](https://huggingface.co/datasets/deepmind/narrativeqa) | recorded |
| qasper | train 888 · validation 281 | [allenai/qasper](https://huggingface.co/datasets/allenai/qasper) | recorded |
| gov_report | train 17,517 | [ccdv/govreport-summarization](https://huggingface.co/datasets/ccdv/govreport-summarization) | recorded |
| multi_news | train 44,972 | [alexfabbri/multi_news](https://huggingface.co/datasets/alexfabbri/multi_news) | recorded |
| musique | train 19,938 | [dgslibisey/MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue) | recorded |
| repobench-p | cross_file_first 8,033 | [tianyang/repobench_python_v1.1](https://huggingface.co/datasets/tianyang/repobench_python_v1.1) | recorded |
| qmsum | — | [Yale-LILY/QMSum](https://github.com/Yale-LILY/QMSum) | 미다운로드 |
