# Figure 1

`figure_1.py`는 실제 full-cache denoising trajectory 한 샘플에서 다음 네
값을 같은 candidate 축으로 추출합니다.

- Sparse-dLLM current-state score: step 1 진입 상태의 current-block query와
  candidate key로 계산하며, 원 구현처럼 query-token 평균, head 평균,
  kernel-3 max pooling을 적용합니다.
- Future answer usage: block 완성 후 한 번 더 forward해서 얻은
  `[layer, answer token, candidate]` attention 행렬입니다.
- Oracle: 위 행렬을 answer-token 축으로 max한 `label_final_rowmax`입니다.
- Ours: `candidate`, `block mean`, `candidate × block` 특징을 사용하는 학습된
  student 점수입니다.

기본 정성 패널은 미리 고정한 layer index 24와 두 번째 generation block을
사용합니다. 오른쪽 Mass@K/Recall@K는 해당 샘플의 모든 layer와 모든 block을
평균합니다. `--keep-ratio 0.1`이면 후보 수의 10%를 Top-K budget으로 사용하며,
오른쪽 점선 `Full cache (1.0)`을 기준으로 그 작은 캐시에 미래 attention mass와
oracle Top-K가 얼마나 남는지 보여줍니다. 정수 K를 내림하므로 실제 유지 비율은
후보 수에 따라 10%보다 조금 작을 수 있습니다.

```bash
conda activate future-dllm
python visualization/figure_1.py \
  --dataset math5s \
  --sample-index 0 \
  --student artifacts/ckpts/<run>/checkpoint-best \
  --layer 24 \
  --block-index 1 \
  --keep-ratio 0.1
```

다른 프로세스가 GPU를 사용 중이면 기본적으로 2 GiB를 student, cache,
activation용으로 남기고 일부 model layer를 CPU에 배치합니다. 전용 GPU에서는
`--gpu-reserve-gib 0`으로 자동 배치의 전체 GPU 메모리를 사용할 수 있습니다.

기본 출력 위치는 `artifacts/figure_1/`입니다.

- `figure_1.png`, `figure_1.pdf`: 논문용 정성/정량 패널
- `analysis.pt`: current score, token별 future attention, row-max oracle,
  prediction, 세 Top-K 집합을 포함한 원자료
- `metrics.csv`: block/layer별 Mass@K와 Recall@K
- `summary.json`: 전체 평균과 선택한 정성 패널의 수치

모델을 다시 실행하지 않고 K나 정성 layer/block만 바꾸려면 저장된 원자료를
사용합니다.

```bash
python visualization/figure_1.py \
  --analysis-input artifacts/figure_1/analysis.pt \
  --cache-budget 72 \
  --layer 24 \
  --block-index 1
```

Teacher shard에 token별 행까지 임시로 보관해야 하는 별도 분석에서는
`teacher/extract_teacher.py --save-attention-rows`를 사용할 수 있습니다. 큰
분석 행렬이 학습 shard와 섞이지 않도록 별도 `--output-root`를 지정하는 편이
좋습니다.

```bash
python teacher/extract_teacher.py \
  --dataset math5s --n-samples 1 \
  --save-attention-rows \
  --output-root artifacts/figure_1_teacher
```

기본 teacher 추출은 기존처럼 `label_final_rowmax`만 저장합니다.
