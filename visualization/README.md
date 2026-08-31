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

## MultiNews paper figure

MultiNews에서는 `|||||` 문서 구분자를 자동으로 찾아 prompt token 축에
`Article 1 | Article 2 | ...` 경계를 표시할 수 있습니다. 본문 Figure에는
긴 summary를 생성하고 네 기사 경계를 보존하는 `multinews-77`, 고정 layer
index 24, summary 중간의 block index 7을 사용합니다.

```bash
python visualization/figure_1.py \
  --dataset multi_news \
  --sample-index 77 \
  --student artifacts/ckpts/<run>/checkpoint-best \
  --layer 24 \
  --block-index 7 \
  --keep-ratio 0.15 \
  --summary-granularity token \
  --qualitative-scope all \
  --output-dir artifacts/figure_1_multinews_77/all_topk_r015
```

세 heatmap은 모두 같은 full-cache `future_attention_rows`에서 시작합니다.
각각 Sparse-dLLM, `label_final_rowmax` oracle, ours가 선택한 global Top-K 열을
표시하고, heatmap 밝기는 선택된 열에서 실제 future attention 크기만을
나타냅니다. 위의 굵은 binary 띠에는 attention 크기와 무관하게 각 방법이 남긴
모든 Top-K 위치가 표시됩니다. `all` scope에서는 prompt, previous summary,
future masked summary가 경쟁하는 전체 candidate pool을 그대로 표시하며,
오른쪽 Mass/Recall도 같은 pool에서 계산합니다.

`sentence`를 사용하면 completed answer token 행을 sentence-wise max로 줄일 수
있지만, Sparse-dLLM 논문의 query-token × key-token attention map과 같은 형태의
본문 Figure에는 `token`을 사용합니다.

저장된 teacher shard가 있으면 selection state를 replay하고 마지막 block만
denoise합니다. replay한 token별 row의 max가 저장 `label_final_rowmax`와
일치하는지 모든 block에서 검사하며, 결과는 `summary.json`에도 기록합니다.

모델을 다시 실행하지 않고 K나 정성 layer/block만 바꾸려면 저장된 원자료를
사용합니다.

```bash
python visualization/figure_1.py \
  --analysis-input artifacts/figure_1/analysis.pt \
  --cache-budget 72 \
  --layer 24 \
  --block-index 1
```

## Multi-layer and layer-average export

저장된 `analysis.pt`에서 early/middle/late layer 3개 × 방법 3개의 9-panel
heatmap과 전체 layer 평균 heatmap을 동시에 만들 수 있습니다.

```bash
python visualization/figure_1_layers.py \
  --analysis-input artifacts/figure_1_gsm8k_16/analysis_r015/analysis.pt \
  --keep-ratio 0.1 \
  --block-index 4 \
  --layers 4,15,24 \
  --output-dir artifacts/figure_1_gsm8k_16/layer_grid_r010
```

평균 파일은 layer score를 먼저 평균하지 않습니다. 각 layer의 고유 Top-K를
그 layer의 future attention에 적용한 뒤 retained attention을 layer 방향으로
평균합니다. 평균 heatmap 위의 띠는 candidate별 layer 선택 빈도입니다.

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
