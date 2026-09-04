# future_dllm

디퓨전 LLM을 위한 KV 캐시 축출(eviction)입니다. 완성될 답변이 무엇을 필요로 할지를 기준으로 순위를 매깁니다.

모델: `GSAI-ML/LLaDA-8B-Instruct`

## 설치

```bash
conda env create -f environment.yml
conda activate future-dllm
```

설치 관련 참고 자료입니다.

- [모델 다운로드 스크립트](scripts/download_model.sh)
- [데이터셋 다운로드 스크립트](scripts/download_data.py)
- [기본 teacher 추출 스크립트](scripts/extract_default_teacher.sh)
- [기본 student 학습 스크립트](scripts/train_default_student.sh)
- [데이터셋 출처와 디렉터리 구조](data/README.md)
- [평가 태스크 구조](eval/tasks/README.md)



```
sparse_future/
├── model/
│   └── LLaDA-8B-Instruct/        # scripts/download_model.sh 가 받는 모델 가중치
├── data/
│   ├── eval/<name>/              # 평가용 parquet
│   ├── train/<name>/             # 학습용 parquet/jsonl
│   └── longbench/data/*.jsonl    # LongBench 공식 형식
├── artifacts/                    # 프롬프트 샤드, teacher 라벨, 체크포인트
└── results/                      # 평가 결과 json
```


```bash
scripts/download_model.sh
python scripts/download_data.py
```


```bash
python scripts/download_data.py --parts eval
python scripts/download_data.py --parts train
python scripts/download_data.py --parts longbench
```

## 평가 데이터셋

 `max_seq_len` :  4096 

| 데이터셋 | 생성 길이 |
|---|---:|
| `gov_report` / `multi_news` / `qmsum` | 512 |
| `samsum` / `qasper` / `narrativeqa` | 128 |
| `trec` / `lcc` / `repobench-p` / `multifieldqa_en` | 64 |
| `triviaqa` / `2wikimqa` / `hotpotqa` / `musique` / `passage_retrieval_en` / `passage_count` | 32 |
| `gsm8k` (5-shot) | 256 * |
| `math` (4-shot) | 256 * |
| `math500` (4-shot) | 256 * |
| `humaneval` | 512 |
| `mmlu` (5-shot) / `arc_c` (25-shot) / `piqa` / `gpqa` (5-shot) | 생성 없음 ** |

`**` 4지선다 loglikelihood 로 채점하므로 아무것도 생성하지 않습니다.

## 학습 데이터셋


| 데이터셋 | 샘플 수 | 생성 길이 | 프롬프트 상한 | teacher 블록 | 
|---|---:|---:|---:|---:|---|
| `math5s` | 500 | 256 | 3,840 | 8 | 
| `mbpp_full` | 371 | 256 | 3,840 | 8 | 
| `gov_report` | 150 | 512 | 3,584 | 16 | 
| `multi_news` | 100 | 512 | 3,584 | 16 | 
| `musique` | 500 | 32 | 4,064 | 1 | 
teacher 블록 = 생성 길이 / block_length(32). 프롬프트 상한 = 4096 − 생성 길이.



## Teacher 라벨

현재 기본 학습 구성을 처음부터 추출하려면 다음 스크립트를 사용합니다.
기본 총 시퀀스 길이는 `프롬프트 + 생성 = 최대 4096`이며, 데이터셋별 생성
길이를 먼저 확보한 나머지를 프롬프트에 사용합니다.

```bash
scripts/extract_default_teacher.sh
```

데이터셋별 샘플 수·생성 길이·프롬프트 상한·블록 수는 위 [학습 데이터셋](#학습-데이터셋)
표를 참고하십시오.

개별 데이터셋만 추출

```bash
python teacher/build_prompt_shards.py    --model model/LLaDA-8B-Instruct --dataset samsum --limit 300 --max-seq-len 4096
python teacher/extract_teacher_llada.py --model model/LLaDA-8B-Instruct --dataset samsum --n-samples 300 --max-seq-len 4096

# Dream 은 같은 인자에 전용 스크립트만 바꿔 씁니다.
python teacher/extract_teacher_dream.py --model model/Dream-v0-Instruct-7B --dataset samsum --n-samples 300 --max-seq-len 4096
```

teacher 추출과 student 학습은 total 4096을 기준으로 합니다. 4096을 넘는
LongBench 장문 실험은 학습 데이터를 다시 만들지 않고 추론에서만
`MAX_SEQ_LEN`을 늘립니다.

## 학습

default 학습스크립트 

```bash
scripts/train_default_student.sh
```

개별 구성 학습

```bash
python student/train_student.py --model model/LLaDA-8B-Instruct --teacher-root artifacts/teacher/samsum
```
sample ckpt : [https://huggingface.co/solhee/future-dllm-scorer/blob/main/default_5ds_500-371-150-100-500_e15_lr2e-4_20260827_002758_best10.zip](https://huggingface.co/solhee/future-dllm-scorer/blob/main/checkpoint-best.zip)


## 추론

```bash
scripts/run_eval.sh <dataset> <keep_ratio> [checkpoint]

scripts/run_eval.sh samsum 0.1 artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best
scripts/run_eval.sh gsm8k  1.0            # 축출 없음, 체크포인트 없음

# 기본은 평가 데이터 전체, LIMIT을 지정한 경우에만 샘플 수 제한
LIMIT=200 scripts/run_eval.sh math 0.1 artifacts/ckpts/<run>/checkpoint-best

# 기본 총 길이는 4096. 10K LongBench는 추론에서만 선택
MAX_SEQ_LEN=10240 scripts/run_eval.sh gov_report 0.1 artifacts/ckpts/<run>/checkpoint-best

# 길이 일반화 비교가 필요하면 프롬프트만의 상한도 별도로 선택
MAX_SEQ_LEN=4096 MAX_PROMPT_LEN=2048 scripts/run_eval.sh gov_report 0.1 artifacts/ckpts/<run>/checkpoint-best
```
