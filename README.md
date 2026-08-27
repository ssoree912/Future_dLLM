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

## 데이터셋

| 데이터셋 | 출처(테스트가 아닌 split) | 평가 태스크 | teacher `--gen-length` |
|---|---|---|---|
| `samsum` | LongBench samsum 대화 | `longbench_samsum` | 128 |
| `gsm8k` | gsm8k train | `local_gsm8k`, 5-shot | 256 |
| `mmlu` | mmlu validation + dev | `local_mc_mmlu`, 5-shot | 64 |
| `math` | hendrycks_math train | `local_math` | 256 |
| `mbpp` | mbpp validation + prompt | 프롬프트 샤드만 | 128 |
| `mbpp_full` | mbpp 전체 train | `mbpp` | 256 |
| `samsum_lb` / `trec_lb` / `wiki2_lb` | LongBench 형식, 약 2048 토큰 프롬프트 | `longbench_*` | 128 / 64 / 32 |
| `musique` / `qasper` / `gov_report` / `repobench_p` | LongBench 원본 train | `longbench_*` | 32 / 128 / 512 / 64 |
| `arc_c` | ARC-Challenge train | `local_mc_arc_challenge`, 25-shot | teacher 미지원 |
| `piqa` | PIQA train | `local_mc_piqa` | teacher 미지원 |
| `gpqa` | GPQA main | `local_mc_gpqa_main_n_shot`, 5-shot | teacher 미지원 |
| `math500` | MATH-500 test | `local_math500` | teacher 미지원 |
| `humaneval` | HumanEval test | `local_humaneval` | teacher 미지원 |



## Teacher 라벨

현재 기본 학습 구성을 처음부터 추출하려면 다음 스크립트를 사용합니다.

```bash
scripts/extract_default_teacher.sh
```

| 데이터셋 | 프롬프트 | 생성 길이 | teacher 블록 |
|---|---:|---:|---:|
| `math5s` | 500 | 256 | 8 |
| `mbpp_full` | 371 | 256 | 8 |
| `musique` | 1,600 | 32 | 1 |
| `gov_report` | 150 | 512 | 16 |
| `repobench_p` | 800 | 64 | 2 |


개별 데이터셋만 추출

```bash
python teacher/build_prompt_shards.py --dataset samsum --limit 300
python teacher/extract_teacher.py     --dataset samsum --n-samples 300
```

## 학습

default 학습스크립트 

```bash
scripts/train_default_student.sh
```

개별 구성 학습

```bash
python student/train_student.py --teacher-root artifacts/teacher/samsum
```
sample ckpt : https://huggingface.co/solhee/future-dllm-scorer/blob/main/default_5ds_500-371-150-100-500_e15_lr2e-4_20260827_002758_best10.zip


## 추론

```bash
scripts/run_eval.sh <dataset> <keep_ratio> [checkpoint]

scripts/run_eval.sh samsum 0.1 artifacts/ckpts/1ds_300_e6_lr2e-4_6a5fc6/checkpoint-best
scripts/run_eval.sh gsm8k  1.0            # 축출 없음, 체크포인트 없음
```
