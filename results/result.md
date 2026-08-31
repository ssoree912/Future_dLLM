# LongBench / 벤치마크 결과

| 항목 | 값 |
|---|---|
| 모델 | `GSAI-ML/LLaDA-8B-Instruct` |
| total context | **4096** |
| keep ratio | **0.1** |
| block length | 32 |
| sparse-dllm | `Sparse-dLLM`, `kernel_size=3`, `total_context_len=4096` |
| ours | `Future_dLLM`, student `checkpoint-epoch6` |
| ours 체크포인트 | `default_5ds_500-371-150-100-500_e10_lr2e-4_20260829_105222` |
| ours 선택 기준 | val recall macro 0.7720 (epoch 6, 10 epoch 중 최고) |
| ours 학습 설정 | `epochs=10`, `lr=2e-4`, `seed=0`, `proj_dim=256`, `mlp_dim=512`, `pairs=4096`, `val_ratio=0.1`, `max_seq_len=4096` |
| ours 학습 도메인 | math5s 500 / mbpp_full 371 / gov_report 150 / multi_news 100 / musique 500 (teacher 라벨 4096 기준 추출) |

## LongBench

| 분류 | 데이터셋 | sparse-dllm | ours | Δ |
|---|---|---:|---:|---:|
| Few-shot QA | TriviaQA | 67.99 | 65.52 | -2.47 |
|  | TREC | 35.96 | 47.29 | +11.33 |
|  | SAMSum | 34.57 | 35.01 | +0.44 |
| Code | LCC | 63.00 | 68.13 | +5.13 |
|  | RepoBench-P | 60.47 | 63.47 | +3.01 |
| Single-doc QA | MultiFieldQA-en | 29.25 | 30.46 | +1.21 |
|  | Qasper | 29.20 | 34.41 | +5.21 |
|  | NarrativeQA | 14.92 | 15.47 | +0.55 |
| Multi-doc QA | 2WikiMQA | 14.61 | 14.75 | +0.14 |
|  | HotpotQA | 13.29 | 14.19 | +0.90 |
|  | MuSiQue | 5.95 | 6.29 | +0.33 |
| Summarization | GovReport | 19.19 | 23.56 | +4.37 |
|  | MultiNews | 17.67 | 22.87 | +5.19 |
|  | QMSum | 16.41 | 18.20 | +1.80 |
| Synthetic | Passage Retrieval | 41.00 | 41.00 | +0.00 |
|  | Passage Count | 1.50 | 1.50 | +0.00 |
| **평균** | **16개 태스크** | | | **+2.32** |

## 그 외 벤치마크

| 데이터셋 | 지표 | sparse-dllm | ours | Δ |
|---|---|---:|---:|---:|
| GSM8K | flexible-extract | 47.99 | 77.26 | +29.26 |
| GSM8K | strict-match | 12.66 | 58.30 | +45.64 |
| MATH-500 | math_verify | 14.60 | 34.60 | +20.00 |
| MATH-500 | exact_match | 0.00 | 11.80 | +11.80 |
| MATH | math_verify | 15.90 | – | – |
| MATH | exact_match | 0.00 | – | – |
| HumanEval | pass@1 | 15.85 | – | – |
| GPQA (5-shot) | acc | – | – | – |
| PIQA | acc | – | – | – |
| ARC-Challenge (25-shot) | acc_norm | – | – | – |

`–` 는 아직 실행하지 않았거나 해당 방법에 결과가 없는 항목입니다.
