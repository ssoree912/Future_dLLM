# Future-attention KV 캐시 Eviction: 수식과 방법론

이 repo가 쓰는 축출 방법의 정의를 한곳에 모은 문서다. teacher 라벨 → student 학습 →
배포 시 선택까지가 본문이고, 마지막에 Sparse-dLLM / H2O 의 점수식과 나란히 놓아
무엇이 실제로 다른지 정리한다. 블록 간 누적(`across_blocks`)은 §6 에 있다.

구현 위치는 §8 의 표를 보면 된다.

---

## 1. 표기

확산 LLM 은 길이 $G$ 의 답변을 길이 $B$ 의 블록 $N = G/B$ 개로 나눠 순서대로 채운다.
프롬프트 길이를 $P$, 전체 시퀀스 길이를 $T = P + G$ 라 하자. 블록 $b$ 가 차지하는 위치는

$$\mathcal{B}_b = \{\,P + bB,\ \dots,\ P + (b+1)B - 1\,\}$$

캐시 후보는 **현재 블록을 제외한 나머지 전부**다.

$$\mathcal{C}_b = \{0, \dots, T-1\} \setminus \mathcal{B}_b$$

여기에는 프롬프트, 이미 확정된 앞 블록, 그리고 **아직 mask 인 뒤 블록**이 모두 들어간다.
블록은 자기 자신의 key 에는 직접 attend 하므로 후보에서 빠진다
(`cache.py:filter_cache` 가 `cur_filtered_len` 앞뒤로 잘라내는 부분).

보존 예산은 keep ratio $\rho$ 에 대해

$$k_b = \lfloor |\mathcal{C}_b| \cdot \rho \rfloor$$

레이어 $l \in \{0, \dots, L-1\}$ 마다 독립적으로 선택하고, head 는 **공통 인덱스**를 쓴다
(head 별로 다른 집합을 유지하지 않는다 — §7 참고).

선택은 **블록당 정확히 1회**, 블록의 step-1 full-sequence forward 직후에 일어난다.

---

## 2. Teacher 라벨: final $x$ row-max

블록을 **전체 캐시로 끝까지 채운 뒤**, 완성된 블록에 대해 forward 를 한 번 더 돌려
그 답변 토큰들이 무엇을 봤는지 읽는다. head 평균 attention 을

$$a^{(l)}_{rj} = \frac{1}{H}\sum_{h=1}^{H} \operatorname{softmax}_j\!\left(\frac{q^{(l)}_{h,r} \cdot k^{(l)}_{h,j}}{\sqrt{d}}\right), \qquad r \in \mathcal{B}_b,\ j \in \mathcal{C}_b$$

라 할 때 라벨은 블록 행에 대한 **최댓값**이다.

$$I^{(l)}_{b,j} = \max_{r \in \mathcal{B}_b} a^{(l)}_{rj}$$

합이 아니라 최댓값인 것이 핵심이다. 합을 쓰면 한 토큰만 강하게 의존한 항목이
평균에 씻겨 사라지는데, 그런 항목이야말로 남겨야 하는 것이다.

이 라벨이 "완성될 답변이 무엇을 필요로 하는가"를 정의한다. 배포 시점에는 아직
답변이 없으므로 이 값을 직접 계산할 수 없고, 그래서 student 가 필요하다.

라벨과 함께 블록 step-1 시점의 **모델 입력**($x$)을 저장한다. hidden state 를 통째로
저장하지 않고 학습 때 같은 forward 를 재현(replay)하기 위해서다.

---

## 3. Student: 블록 조건부 스코어러

레이어 $l$ 의 hidden state 를 $h^{(l)} \in \mathbb{R}^{T \times d}$ 라 하자
(배포 시 실제로 쓰이는 prompt+generation forward 의 것, 학습 시에는 replay 로 재현).

후보 $j$ 와 현재 블록을 각각 사영한다.

$$u_j = W^{(l)}_{\text{tok}} h^{(l)}_j \in \mathbb{R}^{p}, \qquad
v_b = W^{(l)}_{\text{blk}} \left( \frac{1}{B} \sum_{r \in \mathcal{B}_b} h^{(l)}_r \right) \in \mathbb{R}^{p}$$

둘과 그 원소별 곱을 이어 붙여 2층 MLP 에 넣는다.

$$s^{(l)}_{b,j} = \operatorname{MLP}^{(l)}\!\left(\left[\,u_j \,;\, v_b \,;\, u_j \odot v_b\,\right]\right), \qquad \operatorname{MLP}: \mathbb{R}^{3p} \to \mathbb{R}^{m} \xrightarrow{\text{GELU}} \mathbb{R}$$

레이어마다 파라미터가 따로 있다 ($L$ = 32). 기본값은 $p = 256$, $m = 512$.

$v_b$ 항이 이 방법의 정체성이다. 점수는 후보 토큰만의 함수가 아니라
**"지금 채우는 블록에게" 얼마나 쓸모 있는가**의 함수다. 같은 후보라도 블록이 바뀌면
점수가 달라진다.

---

## 4. 손실: 랭킹 목적함수

라벨의 절대 크기가 아니라 **순서**가 목적이므로 두 항을 쓴다. 라벨을 분포로 정규화하고

$$t_j = \frac{I^{(l)}_{b,j}}{\sum_{j'} I^{(l)}_{b,j'}}, \qquad p_j = \operatorname{softmax}_j\big(s^{(l)}_{b,\cdot}\big)$$

listwise 항은 그 분포에 대한 KL, pairwise 항은 무작위 쌍의 순서 맞추기다.

$$\mathcal{L} = \underbrace{\lambda \sum_{j} t_j \log \frac{t_j}{p_j}}_{\text{listwise}}
\;+\; \underbrace{\mathbb{E}_{(i,j)}\Big[\operatorname{softplus}\big(-\operatorname{sign}(t_i - t_j)\,(s_i - s_j)\big)\Big]}_{\text{pairwise}}$$

KL 의 reduction 은 `sum` 이어야 한다. `batchmean` 은 후보 수로 나누기 때문에
프롬프트가 긴 도메인의 listwise 항만 수백~수천 배 작아져서, 의도치 않게
도메인 가중치가 프롬프트 길이에 반비례하게 된다.

**여기서 나오는 결론 하나**: $s$ 는 랭킹 점수이지 교정된(calibrated) 크기가 아니다.
학습이 실제로 맞춘 양은 $s$ 가 아니라 $\operatorname{softmax}(s)$ 다. §6 에서 다시 쓴다.

체크포인트 선택은 손실이 아니라 **recall** 로 한다. 비율 격자
$r \in \{0.05, 0.1, 0.2, 0.3, 0.5\}$ 에 대해

$$\operatorname{Recall}@r = \frac{\big|\operatorname{TopK}_{k}(s) \cap \operatorname{TopK}_{k}(I)\big|}{k}, \qquad k = \max(1, \lfloor |\mathcal{C}_b| r \rfloor)$$

의 평균을 쓴다. 특정 예산에 과적합된 체크포인트를 고르지 않기 위해서다.

---

## 5. 배포 시 선택 (현재 기본 동작)

블록 $b$, 레이어 $l$ 에서

$$\mathcal{K}^{(l)}_b = \operatorname*{arg\,TopK}_{j \in \mathcal{C}_b,\ k_b}\ s^{(l)}_{b,j}$$

를 남기고 나머지 KV 를 버린다. 프롬프트와 미래 블록이 **하나의 top-k 안에서 경쟁**하므로
예산은 정확히 $|\mathcal{C}_b| \cdot \rho$ 이고, 구간별 할당량을 따로 주지 않는다.

블록이 끝나면 캐시를 통째로 버리고 다음 블록의 step 0/1 에서 다시 만든다
(`llada_generate.py`: *a fresh cache per block*). 즉 **한 블록에서 버린 항목이
다음 블록에서 다시 후보가 된다** — 뒤에서 중요해지는 성질이다.

---

## 6. 블록 간 누적 (`eviction_accum=across_blocks`)

H2O 의 시간축을 이 구조에 옮긴 축이다. AR 의 "생성이 진행되며 새 query row 가 생긴다"에
대응하는 것은 블록 내부의 재채점이 아니라 **블록 그 자체**다.

블록별 점수를 후보 집합 위에서 정규화하고

$$\pi^{(l)}_{b,j} = \operatorname{softmax}_{j \in \mathcal{C}_b}\big(s^{(l)}_{b,\cdot}\big)_j$$

절대 위치 축의 누적 버퍼 $A^{(l)} \in \mathbb{R}^{T}$ 에 감쇠 $\gamma$ 로 쌓는다.

$$A^{(l)}_{b,j} = \gamma\, A^{(l)}_{b-1,j} + \pi^{(l)}_{b,j}\,\mathbb{1}[\,j \in \mathcal{C}_b\,], \qquad A^{(l)}_{-1} = 0$$

선택은 누적값 위에서 한다.

$$\mathcal{K}^{(l)}_b = \operatorname*{arg\,TopK}_{j \in \mathcal{C}_b,\ k_b}\ A^{(l)}_{b,j}$$

**왜 $s$ 가 아니라 $\pi$ 를 더하는가.** §4 에서 본 대로 $s$ 의 스케일에는 의미가 없다.
raw logit 을 그대로 더하면 우연히 분산이 큰 블록이 전체 합을 지배한다. softmax 를 거치면
모든 블록이 정확히 질량 1 씩 기여하고, 이는 H2O 가 head 마다 softmax 로 질량을 맞추는 것과
같은 역할이다. 게다가 $\pi$ 는 학습이 실제로 맞춘 바로 그 양이다.

**$\gamma$ 의 의미.**

| $\gamma$ | 동작 |
|---|---|
| $0$ | $A_b = \pi_b$ 이고 softmax 는 단조라 **현재 기본 동작과 선택이 동일**하다 |
| $1$ | 순수 누적 — H2O 의 `across_blocks` 에 대응 |
| $(0,1)$ | 최근 블록 위주 + 과거의 지수 감쇠 기억 (EMA) |

**이 구조에서만 가능한 단순화.** Sparse-dLLM/H2O 는 캐시가 블록을 넘어 살아남기 때문에
누적 점수를 생존자 인덱스로 다시 매핑해야 한다(`prev[:, :, keep_pos]`). 여기서는 블록마다
캐시를 새로 만들고 후보 전체를 다시 채점하므로, 절대 위치로 그냥 더하면 된다.
부작용으로 **한 번 버려진 항목이 영구히 배제되지 않는다** — 누적 점수가 낮아 계속 밀릴 뿐,
후보 자격은 매 블록 회복된다.

**비용.** 추가 forward 없음. 연산은 후보 길이만큼의 덧셈 한 번, 메모리는 레이어당 $T$ 개
float ($L \cdot T$, 32×4096 이면 0.5 MB 수준). 캐시 크기와 속도는 바뀌지 않는다.

**두 가지 근사.**

1. 블록이 넘어가면 mask 였던 위치가 디코딩되며 $h^{(l)}_j$ 자체가 바뀐다. 서로 다른
   표현에서 잰 점수를 더하는 근사다 (프롬프트 구간은 고정).
2. 더 본질적으로, $v_b$ 로 준 **블록 조건성이 누적하면서 희석된다**. 조건부 점수를
   블록에 대해 평균 내면 조건이 없는 점수에 가까워진다. H2O 에서 누적이 이득이었던 것은
   원래 조건부가 아닌 스코어러였기 때문이고, 여기서는 같은 이유로 이득이 안 될 수 있다.
   $\gamma$ 를 노브로 둔 이유가 이것이다.

---

## 7. 다른 방법들과의 비교

같은 자리(블록당 1회, 같은 후보 집합, 같은 예산)에서 무엇을 점수로 쓰느냐만 다르다.

**Sparse-dLLM** — 현재 블록 query 평균과의 raw 내적, head 평균, 폭 $w$ 의 stride-1 max pool:

$$\hat{s}_j = \operatorname{maxpool}_w\left(\frac{1}{H}\sum_h \left(\frac{1}{B}\sum_{r \in \mathcal{B}_b} q_{h,r}\right) \cdot k_{h,j}\right)$$

softmax 도 $\sqrt{d}$ 스케일도 없다. 내적이 선형이라 query 평균은 "raw logit 의 query 평균"과
수학적으로 동일하고, 실질적 차이는 head 집계에서 나온다 — 정규화가 없으면 logit 스케일이 큰
head 몇 개가 점수를 지배한다.

**H2O (sibling repo)** — 행별 softmax 후 합, head 별 인덱스 유지:

$$\alpha_{h,j} = \sum_{r \in \mathcal{B}_b} \operatorname{softmax}_j\!\left(\frac{q_{h,r} \cdot k_{h,j}}{\sqrt{d}}\right)$$

**Ours** — $s^{(l)}_{b,j}$ (§3). 앞의 둘이 **현재 블록이 지금 무엇을 보고 있는지**를 재는 반면,
이쪽은 **완성될 블록이 무엇을 필요로 할지**를 teacher 라벨로부터 예측한다. mask 상태의
attention 은 아직 그 정보를 갖고 있지 않다는 것이 전제다.

세 방법 모두 §6 의 누적을 붙일 수 있고, 축은 서로 직교한다.

---

## 8. 구현 위치

| 위치 | 내용 |
|---|---|
| `teacher/extract_teacher.py` | 라벨 정의 (§2), 블록 루프 |
| `future_dllm/cache.py:record_attention` | head 평균 attention 행 기록 |
| `student/train_student.py` | 손실 (§4), replay forward, recall 기반 체크포인트 선택 |
| `future_dllm/student_cache.py` | student 구조 (§3) |
| `future_dllm/cache.py:filter_cache` | 배포 시 선택 (§5), 누적 (§6) |
| `future_dllm/llada_generate.py` | 블록 루프, 누적 상태 소유 |
| `future_dllm/cache.py:sparse_dllm_current_score` | Sparse-dLLM 점수식 (§7) |

**플래그** (기본값은 기존 동작과 동일하고, 켰을 때만 `model_args` 와 resume 키에 기록된다):

```bash
EVICTION_ACCUM=across_blocks EVICTION_ACCUM_DECAY=1.0 \
  scripts/run_eval.sh humaneval 0.2 <checkpoint>
```

---

## 9. 재학습이 필요한 경우 / 아닌 경우

**재학습·재추출 없이 되는 것**

- 블록 간 누적 (§6). student 가 이미 내놓는 출력을 추론 시점에 집계할 뿐이다.
- sink / recent 강제 보존. 점수를 무시하고 자리를 예약하는 마스크이므로 학습과 무관하다.
  예산에 더해지는 것이 아니라 예산 **안에서** 예약하는 것이라 캐시 크기도 그대로다.

**재추출까지 필요한 것**

- **head 별 인덱스 유지** (H2O 의 per-head 축). 라벨이 이미
  `cache.py:record_attention` 에서 head 평균으로 뭉개져 있고 student 출력도 위치당 스칼라
  하나다. head 별 집합을 쓰려면 라벨을 head 축을 살려 다시 뽑고 student 출력 차원을
  $H$ 로 늘려야 한다.
- 라벨 정의 자체를 바꾸는 경우 (예: row-max 대신 누적 타깃, 또는 블록 조건 없는 타깃).
