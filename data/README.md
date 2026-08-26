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

SAMSum · TREC · QMSum 세 개는 스크립트가 받지 않습니다. 원본 릴리스에서 직접 받아야 합니다.

PIQA · Qasper · Multi-News · LongBench 는 Hub 에 로더 스크립트로만 올라와 있어
`datasets` 5.0 이 실행을 거부합니다. 네 개는 아래 표의 공식 릴리스에서 직접 받아
로더 스크립트가 만들던 컬럼 구조로 다시 씁니다.


## 평가

| 데이터셋 | 행 수 | 출처 |
|---|---|---|
| MMLU | test 14,042 · validation 1,531 · dev 285 | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| ARC-Challenge | test 1,172 · validation 299 | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) `ARC-Challenge` |
| PIQA | validation 1,838 | [physicaliqa-train-dev.zip](https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip) (`ybisk/piqa` 의 원본) |
| GPQA | main 448 | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) |
| GSM8K | test 1,319 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` |
| MATH | test 5,000 | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) |
| MATH-500 | 500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| HumanEval | 164 | [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) |
| LongBench | 16개 태스크, 각 200행 (lcc / repobench-p 는 500행) | [zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench) `data.zip` |

LongBench 평가는 `data.zip` 하나로 끝납니다. 그 안의 `data/<task>.jsonl` 이
`eval/tasks/longbench/*.yaml` 이 읽는 형식 그대로라, 별도 변환 단계가 없습니다.


### LongBench 원본 데이터셋(test)

teacher 프롬프트 샤드가 쓰는 train split 입니다. 위의 평가용 `data.zip`(test 200행)과는
별개 데이터입니다. 평가는 어느 태스크든 `data.zip` 만으로 돌아갑니다.

`미다운로드` 는 스크립트가 받지 않는다는 뜻입니다. SAMSum 과 TREC 은 `samsum` ·
`samsum_lb` · `trec_lb` 샤드가 요구하므로 없으면 그 샤드를 만들 수 없습니다. QMSum 은
`build_prompt_shards.py` 의 `BUILDERS` 에 샤드가 없어 train split 을 쓰는 곳이 없습니다.

| 태스크 | 행 수 | 출처 |
|---|---|---|
| samsum | train 14,732 · 미다운로드 | 공식 SAMSum 릴리스, [`corpus.7z`](https://arxiv.org/src/1911.12237v2/anc/corpus.7z) |
| 2wikimqa | train 167,454 | [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop), 미러 `xanhho/2WikiMultihopQA` |
| trec | `train_5500.label` · 미다운로드 | TREC 원본 |
| triviaqa | 64,916 | [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa) `rc.web` |
| narrativeqa | train 32,747 | [deepmind/narrativeqa](https://huggingface.co/datasets/deepmind/narrativeqa) |
| qasper | train 888 · validation 281 | [qasper-train-dev-v0.3.tgz](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz) (`allenai/qasper` 의 원본) |
| gov_report | train 17,517 | [ccdv/govreport-summarization](https://huggingface.co/datasets/ccdv/govreport-summarization) |
| multi_news | train 44,972 | [alexfabbri/multi_news](https://huggingface.co/datasets/alexfabbri/multi_news) `data/train.src.cleaned` · `data/train.tgt` |
| musique | train 19,938 | [dgslibisey/MuSiQue](https://huggingface.co/datasets/dgslibisey/MuSiQue) |
| repobench-p | cross_file_first 8,033 | [tianyang/repobench_python_v1.1](https://huggingface.co/datasets/tianyang/repobench_python_v1.1) |
| qmsum | 미다운로드 · 쓰는 샤드 없음 | [Yale-LILY/QMSum](https://github.com/Yale-LILY/QMSum) |
