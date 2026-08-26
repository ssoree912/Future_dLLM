# tasks

```
metrics.py            LongBench의 스코어링, 원본 그대로
local/*.yaml          로컬 parquet 생성형 태스크 정의 (GSM8K, MATH, MATH-500, HumanEval)
local_mc/*.yaml       로컬 parquet 객관식(loglikelihood) 태스크 정의
local_*.py            lm-eval 처리 유틸리티 래퍼
generate_tasks.py     LongBench 자체 config로부터 longbench/ 를 다시 생성
generate_local_mc.py  local_mc/ 를 다시 생성
longbench/            태스크당 yaml 하나, 그 외에는 없음
```

모든 태스크가 저장소의 `data/` 아래 파일만 읽습니다. 실행 중에 Hugging Face Hub 를 호출하는 태스크는
없습니다.

## longbench/

yaml 파일들은 LongBench의 `config/dataset2prompt.json` 과 `config/dataset2maxlen.json` 을 기준으로,
아래 공식 데이터셋에 맞춰 생성한 것입니다.

    https://huggingface.co/datasets/zai-org/LongBench

데이터는 `data/longbench/data/<task>.jsonl` 을 사용합니다. 공식 스키마이며, `input` 과 가공되지 않은
`context` 로 구성됩니다. 프롬프트는 LongBench 가 두는 위치와 동일하게 `doc_to_text` 에 들어 있습니다.
프롬프트를 `context` 안에 미리 넣어 재포장한 사본을 쓰면 지시문이 두 번 렌더링되니 주의해 주세요.

## local_mc/

MMLU, ARC-Challenge, PIQA, GPQA 는 lm-eval 의 표준 객관식 태스크 정의를 복사한 뒤 `dataset_path` 만
`parquet` 으로 바꾼 것입니다. 프롬프트, split, few-shot 샘플러, 지표가 모두 원본과 같고 데이터를 읽는
곳만 다르므로, 점수는 lm-eval 기본 태스크로 돌린 결과와 그대로 비교할 수 있습니다.

- `local_mc_mmlu` — subject 57개 태스크와 stem / other / social_sciences / humanities 4개 그룹,
  크기 가중 평균까지 원본과 동일한 구조입니다. `data/eval/mmlu/<subject>/` 를 읽습니다.
- `local_mc_arc_challenge`, `local_mc_piqa`, `local_mc_gpqa_main_n_shot` — 각각 하나의 yaml 입니다.
- `local_mc_gpqa_utils.py` — GPQA 보기 섞기. 원본과 같은 시드(42)를 같은 순서로 쓰기 때문에 정답
  위치까지 동일합니다.

정의를 고칠 때는 yaml 을 직접 건드리지 마시고 `generate_local_mc.py` 를 고친 뒤 다시 실행해 주세요.

## local/

GSM8K, MATH, MATH-500, HumanEval 입니다. 모두 `generate_until` 로 풀고, 정답 추출과 채점은
`local_*_utils.py` 가 맡습니다.

## 실행 시점 동작

`scripts/run_eval.sh` 는 이 파일들을 하나의 스크래치 디렉터리로 복사하면서 데이터 경로(`DATA_DIR`,
`LONGBENCH_DATA_DIR`)를 치환합니다. lm-eval 이 `!function ...` 참조를 yaml 파일 옆에서 해석하기
때문입니다.
