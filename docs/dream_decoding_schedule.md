# Dream 디코딩 스케줄: timestep 은 누구 것인가

`future_dllm/dream_generate.py` 를 Sparse-dLLM 형태로 옮기면서 무엇이 Dream 원본이고
무엇이 Sparse-dLLM 의 추가인지 정리한다. 결론부터:

> **timestep 스케줄은 Dream 자체 로직이다.** Sparse-dLLM 이 만든 게 아니다.
> Sparse-dLLM 이 더한 것은 그 스케줄을 **블록 단위로 잘라 쓰는 것**과 **KV 캐시 축출**이다.
> 우리 기존 코드가 쓰던 고정 예산(`get_num_transfer_tokens`)은 **LLaDA 것**이고,
> Dream 에는 원래 없던 규칙이다.

## 근거

`model/Dream-v0-Instruct-7B/generation_utils.py` 는 HKUNLP 가 배포한 체크포인트에
동봉된 원본이다 (`future_dllm/generation_utils.py` 와 byte-identical). 여기에
이미 다음이 들어 있다:

```python
# model/Dream-v0-Instruct-7B/generation_utils.py:405
timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
...
# :437
number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
```

같은 파일에서 `block_length` / `block_start` 는 **0회** 등장한다. 즉 원본 Dream 은
전체 시퀀스를 한 번에 확산 복원하며, 블록 개념이 없다.

## timestep 이란

확산 모델의 **노이즈 레벨 t**. "지금 시퀀스가 얼마나 마스킹돼 있어야 하는가" 를
나타내는 연속 축이고 1(전부 마스크) → eps(거의 다 확정)로 내려간다.

```python
timesteps = torch.linspace(1, eps, steps + 1)   # eps = 1e-3, 눈금 steps+1 개
```

스텝 `i` 는 구간 `t = timesteps[i]` → `s = timesteps[i+1]` 을 담당한다.
`1 - s/t` 가 **"남은 마스크 중 이번 스텝에 벗길 비율"** 이다.

```python
num_mask_token = mask_index.sum() / mask_index.shape[0]     # 지금 남은 마스크 수
number_transfer_tokens = int(num_mask_token * (1 - s / t))  # 마지막 스텝은 int(num_mask_token)
```

핵심은 **매 스텝 남은 마스크 수로부터 다시 계산**한다는 점이다. 시작할 때 한 번
정해두는 게 아니다. `t` 가 분모라서 뒤로 갈수록 같은 눈금 간격이 더 큰 비율이 된다:

| step | t | s | 1 − s/t | 남은 마스크 | reveal |
|---|---|---|---|---|---|
| 1 | 0.969 | 0.938 | 0.032 | 31 | 0 |
| 15 | 0.532 | 0.501 | 0.059 | 18 | 1 |
| 27 | 0.157 | 0.126 | 0.199 | 6 | 1 |
| 30 | 0.063 | 0.032 | 0.492 | 3 | 1 |
| 31 | 0.032 | 0.001 | 0.969 | 2 | **2** |

남은 마스크는 줄고 비율은 커져서 가운데 구간은 상수처럼 보이지만, 양 끝에서 어긋난다.

## Sparse-dLLM 이 Dream 원본에서 바꾼 것

Sparse-dLLM 의 Dream 구현과 원본을 비교하면 다음이 전부다. 그 구현은 이 브랜치에
없다 - 레퍼런스 체크아웃
(`/home/M2026107/dllm/Sparse-dLLM/opencompass/models/sparse_dllm/dream/`) 이나,
`origin/feat#2/dream_instruct_ours` 에 벤더링된
`baselines/sparse_dllm/dream/generation_utils.py` 를 보면 된다.

| | Dream 원본 | Sparse-dLLM |
|---|---|---|
| timestep | `linspace(1, eps, steps+1)` | `linspace(1, eps, steps_per_block+1)` — **블록마다 t 가 1 로 리셋** |
| 루프 | `for i in range(steps)` 전체 시퀀스 | `for block: for i in range(steps_per_block)` |
| 마스크 범위 | `x == mask` (전체) | `model_input == mask` (블록 한정) |
| confidence 버퍼 | `full_confidence` (전체 길이) | `block_confidence` (블록 길이) |
| forward 입력 | 항상 전체 `x` | `cache_state` 0/1/2 로 분기 |
| KV 캐시 | 없음 | `CustomCache` — 블록당 1회 선택 후 축출 |

**바뀌지 않은 것**: `sample_tokens`, `alg` 4종(`origin`/`maskgit_plus`/`topk_margin`/`entropy`),
`1 - s/t` 공식, `int(...)` 절삭, `topk` / `alg_temp` + `multinomial` 분기,
`x_` scatter, `shift_logits`. 변수명까지 거의 같다 (`full_confidence` → `block_confidence`,
`x_` → `x_block`).

`cache_state` 3단계는 Sparse-dLLM 것이다:

- **state 0** — 전체 시퀀스 forward. 블록 첫 토큰만 확정하고 `continue`.
  Dream 은 logit 을 한 칸 shift 하므로 (`r` 행이 `r+1` 토큰을 예측) 블록의 첫 토큰은
  블록 **이전** 행에서 읽어야 하는데, 블록만 forward 하면 그 행이 없다. 그래서 그 행이
  존재하는 step 0 에서 미리 확정해 둔다.
- **state 1** — 전체 시퀀스 forward. 여기서 캐시 후보 풀이 만들어지고 축출이 일어난다.
- **state 2** — 블록만 forward. 축출된 캐시에 대고 decode.

## 우리 기존 코드가 쓰던 것과의 차이

우리 `dream_generate.py` 는 LLaDA 경로에서 가져온 고정 예산을 쓰고 있었다:

```python
# future_dllm/llada_generate.py:26
def get_num_transfer_tokens(mask_index, steps):
    """How many tokens each step reveals, under LLaDA's linear noise schedule."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base, remainder = mask_num // steps, mask_num % steps
    counts = ... + base
    for i in range(mask_num.size(0)):
        counts[i, :remainder[i]] += 1     # 나머지는 앞쪽 스텝에 분배
    return counts
```

블록 시작 시 **한 번** 계산하고 끝난다. 나머지를 앞쪽에 분배하므로 **총합은 블록을
정확히 채운다** — 토큰이 미확정으로 남지는 않는다. 어긋나는 것은 **분포**다
(block=32 기준):

```
steps_per_block=32
  Dream/Sparse : [1,0,1,1,...,1,2]   합계 32
  고정 예산    : [1,1,1,1,...,1,1]   합계 32
  누적 불일치  : step 1~30  (한 칸 밀림)

steps_per_block=16
  Dream/Sparse : [1,2,2,...,2,3]     합계 32
  고정 예산    : [2,2,2,...,2,2]     합계 32
  누적 불일치  : step 0~14

steps_per_block=8
  Dream/Sparse : [1,4,4,4,4,4,5,6]   합계 32
  고정 예산    : [4,4,4,4,4,4,4,4]   합계 32
  누적 불일치  : step 0~6
```

`steps == gen_length` 일 때 두 스케줄은 step 1(0 vs 1)과 마지막(2 vs 1)만 다르다.
그런데 그 한 칸 밀림이 이후 모든 스텝의 문맥을 바꾸므로 결과 토큰은 거의 전부
달라질 수 있다. 이것이 GSM8K 4개 프롬프트에서 63/64, 58/64, 13/64, 57/64 만
일치했던 것의 정체다. 스텝 수를 줄일수록 초반부터 어긋나서 격차가 커진다.
측정 기록은 `origin/feat#2/dream_instruct_ours` 의
`future_dllm/sparse_dllm_student.py` docstring 에 있다.

`alg="entropy"` 만 맞추고 스케줄을 그대로 두면 이 불일치는 사라지지 않는다.
**reveal 개수 규칙 자체를 옮겨야 한다.**

## 현재 상태

`future_dllm/dream_generate.py` 가 위 스케줄을 그대로 이식했다
(`feat#3/dream_sparse_decoding` 의 2e27e53).

대조는 `scripts/check_dream_decoding.py` 가 한다. 레퍼런스 체크아웃을 import 해
같은 프롬프트를 양쪽으로 디코드하고 최종 토큰, 중간 history, RNG 소비량을 비교한다.
2026-09-10 실행 결과는 두 스케줄(gen64/steps64, gen64/steps32) 모두
`tokens_equal` / `history_equal` / `rng_equal` 이 참이고 `passed: true` 였다.
출력은 `--output` 으로 지정한 곳에 남는데 그 경로가 `logs/` 아래면 커밋되지
않으므로(.gitignore:13), 재확인이 필요하면 스크립트를 다시 돌리는 편이 빠르다:

```
python scripts/check_dream_decoding.py \
  --model model/Dream-v0-Instruct-7B --output <경로>/check.json
```

## 용어 주의

`eps` 는 두 군데서 다른 뜻으로 쓰인다.

- Dream 디코딩의 `eps=1e-3` — timestep 하한. 위에서 설명한 그것.
- 옵티마이저/정규화의 eps — 무관하다.

`steps` 도 원본에서는 전체 시퀀스 기준, Sparse-dLLM 과 우리 코드에서는
`steps_per_block = steps // num_blocks` 로 나뉜다. 로그를 읽을 때 어느 쪽인지 확인할 것.
