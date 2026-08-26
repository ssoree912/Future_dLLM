# Datasets

각 데이터셋이 어디에서 왔고 어떻게 검증했는지 정리한 문서입니다. 데이터는 저장소 안 `data/` 아래,
`data/eval/<name>/` · `data/train/<name>/` · `data/longbench/data/` 에 놓입니다.

## 다운로드

```bash
python scripts/download_data.py                                   # 전체
python scripts/download_data.py --parts eval                      # 평가 데이터만
python scripts/download_data.py --parts humaneval                 # HumanEval만
python scripts/download_data.py --parts train                     # 학습 데이터만
python scripts/download_data.py --parts longbench                 # LongBench 만
```

Hugging Face Datasets/Hub API로 받은 내용을 `data/` 아래 parquet/jsonl 파일로 저장합니다. 학습과 평가
스크립트는 이 로컬 파일만 읽으므로, 다운로드가 끝난 뒤에는 네트워크가 필요 없습니다.

평가할 때 `datasets` 는 이 parquet 을 arrow 로 한 번 더 변환해 캐시에 둡니다. `scripts/run_eval.sh` 가
그 캐시를 저장소의 `.hf_cache/` 로 고정하므로 홈 디렉터리나 공유 캐시를 건드리지 않습니다. 파생물이라
지워도 되고, 다음 실행에서 다시 만들어집니다.

MMLU 는 `all` config 를 받아 `eval/mmlu/all/` 에 저장하고, 같은 내용을 subject 별로도 나눠
`eval/mmlu/<subject>/{dev,test}.parquet` 으로 씁니다. lm-eval 이 MMLU 를 57개 subject 태스크로 채점하기
때문입니다. 지시문에 subject 이름이 들어가고 5-shot 예시도 그 subject 의 dev 행에서 뽑습니다. 추가로
받는 것은 없고, 이미 받은 `all` 을 나누기만 합니다.

아래는 이 데이터가 어디에서 왔는지에 대한 기록입니다. 실행에는 필요 없고, 논문에 출처를 적거나
데이터를 다시 받을 때 보시면 됩니다. `출처 확인` 열은 **recorded** 다운로드 기록이 남아 있음 ·
**row-count** 저장소를 추정했고 행 수가 공식 릴리스와 일치함 · **card** 함께 배포된 데이터셋 카드로
확인함을 뜻합니다.

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

PIQA 의 test split 에는 정답 라벨이 없어 validation 을 평가 세트로 사용합니다. lm-eval 도 동일합니다.

LongBench 데이터는 공식 릴리스 그대로입니다. `input` 과 가공되지 않은 `context` 로 구성되며, 프롬프트는
`config/dataset2prompt.json` 에 들어 있습니다. `eval/tasks/longbench/` 의 평가 태스크 파일은 그 config
로부터 생성한 것입니다. 프롬프트를 `context` 안에 미리 넣어 재포장한 사본을 쓰면 지시문이 두 번
렌더링됩니다.

MMLU, ARC-C, PIQA, GPQA 는 lm-eval 의 표준 객관식 태스크 정의를 그대로 복사해 데이터 출처만 로컬
parquet 으로 바꾼 `eval/tasks/local_mc/` 로 채점합니다. 프롬프트와 split, 지표가 모두 원본과 같으므로
점수는 lm-eval 기본 태스크로 돌린 결과와 비교할 수 있습니다.

`data/eval/` 에는 MBPP, MMLU-Pro, BBH, LongProc 과 Hendrycks MATH 저장소 사본도 함께 있습니다. 다만
아직 MBPP, MMLU-Pro, BBH, LongProc 으로 평가하는 코드는 없습니다.

## 학습


| 데이터셋 | 행 수 | 출처 | 출처 확인 |
|---|---|---|---|
| MMLU | auxiliary_train 99,842 | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | row-count |
| ARC-Challenge | train 1,119 | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | recorded |
| PIQA | train 16,113 | [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa) | recorded |
| GSM8K | train 7,473 | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main` | row-count |
| MATH | train 7,500 | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | row-count |
| MBPP | full train 374 · sanitized 120 | [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) | row-count |


### LongBench 원본 데이터셋

LongBench 자체는 테스트 전용입니다. 아래는
[`LongBench/task.md`](https://github.com/THUDM/LongBench/blob/main/LongBench/task.md) 기준으로 각
태스크가 만들어진 원본 데이터셋이며, 가공하지 않은 상태로 보관합니다. LongBench 프롬프트 형식으로
변환하는 것은 별도 단계입니다.

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
