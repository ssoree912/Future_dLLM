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

## Future / current eviction 유사도 진단

Dream에서 같은 샘플을 (1) full cache, (2) student의 future score, (3) 현재
블록 query와 외부 cache key의 attention score로 각각 생성한 뒤 full-cache
출력과 비교합니다. 순서를 반영하는 token `SequenceMatcher`와 token-set
Jaccard를 함께 기록합니다. Student가 학습된 `decoding.json`의 step 수를 세
실행 모두에 자동 적용하며, 이 워크스페이스에서는 물리 GPU 2 하나만
`cuda:0`으로 노출합니다.

```bash
LIMIT=10 PY=/opt/conda/envs/future-dllm/bin/python \
  scripts/run_eviction_similarity.sh gsm8k 0.1 \
  artifacts/ckpts/<run>/checkpoint-best
```

결과는 `results/eviction_similarity/.../summary.json`과 `samples.jsonl`에
저장됩니다. 유사도는 task filter를 거친 정답 문자열이 아니라 raw 생성문으로
계산합니다. full-cache 출력을 본 뒤 샘플별로 방법을 고르는 것은 online
oracle이 되므로, 실제 배포 라우팅은 별도 held-out split에서 얻은 **task별**
결정만 사용해야 합니다.

held-out 진단에서 current가 선택된 task는 teacher label과 같은 scaled
attention/softmax 및 row/group reduction을 선택 시점 현재 블록에 적용하는
`eviction_method=current`로 실행합니다. Sparse-dLLM의 kernel-3 pooling을 쓰는
`eviction_method=sparse`와는 별도이며, Student 체크포인트는 필요하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=2 EVICTION_METHOD=current \
  scripts/run_eval.sh gsm8k 0.1
```

Teacher future-attention Top-K 자체에 대한 선택 유사도는 다음처럼 확인합니다.
출력에는 future student와 current attention 각각의 Recall@K와 Jaccard@K가
함께 포함됩니다.

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/baseline_recall.py \
  --student artifacts/ckpts/<run>/checkpoint-best --limit 1
```
