# Phase 3-A: Heterogeneous Expert MoE 구체화 (v3)

## 1. 핵심 문제 정의

현재 baseline(ST-Aware Router + Cap 1.5)은 **누구에게 보낼지**(라우팅)는 시공간 인지형으로 잘 설계되어 있지만, **보낸 곳에서 무엇을 하는지**(처리)는 4개가 동일한 `Linear→GELU→Linear`입니다. 비유하면, 환자 분류(트리아지)는 완벽하지만 모든 수술실에 같은 장비만 놓여 있는 상황입니다.

`experts_v3.py`에는 이미 DWT 대역별로 설계된 4종의 이종 전문가가 존재합니다. 이들은 채널 고정 분할 MoE(`moe_4expert.py`)에서 사용하기 위해 4D `(B, T, N, D)` 인터페이스로 만들어졌지만, 그 안에 담긴 **기상학적 분업 설계**는 지금의 토큰 라우팅 MoE에서도 핵심적으로 유효합니다.

---

## 2. `experts_v3.py` 전문가들의 재평가

### 각 전문가의 설계 의도와 현재 가치

#### Expert1_LLL — Global Spatio-Temporal Attention
- **원래 목적**: LL 대역(ch0, 전체 에너지의 95% 이상)의 거시적 대기 순환 패턴 추적
- **내부 구조**: Temporal Cross-Attention(과거→미래 시계열 매칭) + Spatial Self-Attention(전역 공간 파악) + adaLN-FFN
- **True MoE에서의 가치**: 넓은 공간 범위의 기압골·기압능 같은 **행성 스케일 패턴**을 담당하는 토큰들에게 이 전문가가 배정되면, 단순 FFN으로는 불가능한 **장거리 공간 의존성**을 학습할 수 있습니다.
- **한계**: 내부에 self-attention이 포함되어 있어 이미 STDiTBlock 상위에서 수행한 attention과 **이중 연산** 우려가 있음.

#### Expert2_LLH — Directional Axial Attention
- **원래 목적**: LLH 대역(ch1~3, 수직·수평·대각 엣지)의 방향성 보존
- **내부 구조**: 채널을 3등분하여 각각 수직/수평/대각 축별 독립 attention 수행
- **True MoE에서의 가치**: 해안선, 산맥 릿지라인, 전선대 등 **방향성이 강한 경계 토큰**에 배정 시, 방향별 분리 처리가 경계 선명도를 직접 개선함.
- **한계**: `dim // 3` 분할과 group 연산이 sparse routing의 가변 토큰 수에서 shape 불일치를 일으킬 수 있음.

#### Expert3_LH — Windowed Cross-Attention
- **원래 목적**: LH 대역(ch4~15, 중간 질감)의 지역적 구조 포착
- **내부 구조**: window_size=8의 Windowed Cross-Attention으로 국소 영역 내 context-target 매칭
- **True MoE에서의 가치**: 적운 구름군, 국소 강수 셀 등 **중규모 기상 현상**의 공간 구조를 윈도우 단위로 정밀 처리.
- **한계**: Windowed attention은 토큰의 2D 좌표를 기반으로 윈도우를 구성해야 하므로, sparse routing으로 비연속 토큰이 들어오면 윈도우 구성 자체가 불가능.

#### Expert4_H — 3D Depthwise Conv + Pointwise
- **원래 목적**: H 대역(ch16~63, 극단 고주파)의 격자·체커보드 구조 보존
- **내부 구조**: Context + Target을 시간축으로 concat한 뒤 3D DWConv(3×3×3) + Pointwise Conv
- **True MoE에서의 가치**: 기상학적으로 의미 없는 **미세 텍스처·잔류 노이즈** 토큰을 경량 Conv로 처리하여 연산 효율성 확보.
- **한계**: 3D Conv는 시간축을 포함하므로 `(B, T, N, D)` 4D가 필수. True MoE의 `(B, N, D)`에서는 시간축이 이미 사라진 상태.

### experts_v3과의 관계 정리

| 항목 | 처분 |
|------|------|
| `experts_v3.py` 소스코드 | **보존**. `moe_4expert.py`(채널 분할 MoE)에서 여전히 사용 가능 |
| Expert1~4의 설계 철학 | **계승**. 수용장 크기·방향성·경량화 원칙을 신규 모듈에 반영 |
| 4D 인터페이스 | **사용하지 않음**. True MoE는 3D `(B, N, D)` 전용 |
| attention_v3.py (방향별 attention) | **보존**. v3 경로에서 사용 중이며, 향후 연구에서 참조 가능 |
| adaLN modulation 로직 | **불필요**. STDiTBlock 상위에서 이미 처리됨 |

> **핵심 원칙**: `experts_v3`를 폐기하는 것이 아니라, 그 설계 의도를 MLP-position compatible한 형태로 **재표현**하는 것입니다. 원본은 레거시 코드 경로(`moe_4expert.py`)에서 계속 사용됩니다.

---

## 3. 인터페이스 불일치 문제의 본질

`experts_v3` ↔ `DenseMoEFFN` 사이의 간극은 단순히 tensor shape 차이가 아니라, **MoE 파이프라인 내 전문가의 역할 위치**가 근본적으로 다릅니다.

| 비교 항목 | experts_v3 (채널 분할 MoE) | DenseMoEFFN (토큰 라우팅 MoE) |
|----------|--------------------------|------------------------------|
| **입력 shape** | `(B, T, N, D)` — 시공간 4D | `(B, N, D)` — 공간 2D (시간축 이미 처리됨) |
| **context 접근** | 전문가 내부에서 x_ctx 직접 수신 | 불가. STDiTBlock 상위에서 cross-attn 완료 |
| **conditioning** | 전문가 내부 adaLN으로 c 수신 | 불가. STDiTBlock 상위에서 adaLN 적용 |
| **라우팅 방식** | 채널별 고정 할당 (결정론적) | 토큰별 동적 라우팅 (확률적) |
| **Sparse 호환** | 해당 없음 (전문가가 전체 채널 수신) | `(1, capacity, D)` 가변 토큰 수 |

이 때문에 `experts_v3`의 코드를 **그대로 import하여 사용하는 것은 불가능**합니다. 다만, 그 안에 담긴 **설계 철학**(전역 attention, 방향성 분리, 윈도우 처리, 경량 conv)은 인터페이스를 맞춘 새 모듈로 계승해야 합니다.

---

## 4. 강제 매핑 부재 — 토큰 라우팅 vs 채널 분할 패러다임 차이

### 왜 대역→전문가 강제 매핑을 하지 않는가

현재 파이프라인에서 패치화(patchify library)가 수행되면:

```
DWT latent (B, 64, 32, 32)
    ↓ patch_size=2
Token at position (i,j) = (B, 1, D=512)
    → 이 토큰 안에 ch0(LL)~ch63(HH) 정보가 모두 혼합되어 있음
```

**하나의 토큰이 이미 모든 주파수 대역을 갖고 있으므로**, "이 대역은 이 전문가"라는 매핑 자체가 성립하지 않습니다. 대신 **"이 공간 위치는 어떤 처리 전략이 필요한가"**를 라우터가 판단합니다.

### 두 패러다임의 근본적 차이

| | `moe_4expert.py` (채널 분할, 레거시) | `DenseMoEFFN` (토큰 라우팅, 현재) |
|--|---|---|
| **분할 단위** | **채널** (주파수 대역) | **토큰** (공간 위치) |
| **매핑** | ch0→Expert1, ch1~3→Expert2, ... (하드코딩) | Router가 **학습으로** 결정 |
| **토큰의 정보** | 단일 대역 정보만 담김 | 해당 위치의 **64채널 전부** 담김 |

### 공간 위치 기반 라우팅의 기상학적 정당성

- **해안선 토큰** — LL에서는 육지/바다 경계, HH에서는 날카로운 엣지. 이 토큰은 AxialConvFFN(방향성 처리)이 **모든 대역에 걸쳐** 처리하는 것이 유리

- **open ocean 토큰** — LL에서도 평탄, HH에서도 거의 0. 모든 대역이 동일하게 단순 → PureFFN이 **대역 무관하게** fast-track 처리

- 만약 강제로 "ch0→GlobalConv"를 하면, open ocean의 ch0도 불필요하게 7×7 Conv를 거치게 됨. **공간 위치의 특성을 무시하는 낭비.**

> **`moe_4expert.py`**: "주파수별로 나눠서 각자 처리해라" (분리 후 전문 처리)
> **`DenseMoEFFN` + 이종 전문가**: "모든 정보를 가진 토큰에 대해, 제일 적합한 도구를 골라 처리해라" (통합 후 선택적 처리)

라우터가 `Z = [X ∥ ΔX ∥ P_emb]`를 보고 **위치의 기상학적 성격**을 스스로 파악하여 적절한 수용장의 전문가를 선택합니다. 우리는 전문가에게 **서로 다른 도구(수용장)**를 줄 뿐, 어떤 토큰에 어떤 도구를 쓸지는 라우터가 결정합니다.

---

## 5. Weight Transfer 전략 — 베이스라인 성과 보존

### 핵심 통찰

baseline(SSIM=0.500)이 이미 잘 학습된 상태이므로, 이종 전문가를 **from scratch** 학습하면 이 성과를 잃을 위험이 있습니다.

모든 이종 전문가의 FFN 부분은 기존 동질 FFN과 **구조가 동일**합니다:

```
기존 FFN Expert:       Linear(512→2048) → GELU → Linear(2048→512)
GlobalConvFFN Expert:  DWConv7x7 → LN → Linear(512→2048) → GELU → Linear(2048→512)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                        기존 FFN과 완전히 동일한 부분
```

따라서 **baseline checkpoint에서 weight를 직접 계승**할 수 있습니다:

| 모듈 | FFN weight | Conv weight | 효과 |
|------|-----------|-------------|------|
| **PureFFN (Expert 3)** | baseline FFN **그대로** 로드 | 없음 | baseline과 100% 동치 |
| **LocalConvFFN (Expert 2)** | baseline FFN **그대로** 로드 | DWConv 3×3 **zero-init** | 시작 시 baseline 동치, 학습 중 분화 |
| **AxialConvFFN (Expert 1)** | baseline FFN **그대로** 로드 | 1×5+5×1 **zero-init** | 시작 시 baseline 동치, 학습 중 분화 |
| **GlobalConvFFN (Expert 0)** | baseline FFN **그대로** 로드 | DWConv 7×7 **zero-init** | 시작 시 baseline 동치, 학습 중 분화 |
| **Shared Expert** | baseline shared **그대로** 유지 | 변경 없음 | bias-free baseline 보존자 |
| **ST-Aware Router** | baseline router **그대로** 로드 | — | 라우팅 지식 보존 |

### Zero-init의 수학적 의미

Conv를 zero-init하면, 학습 시작 시점에서:
- `DWConv(x) = 0` → `x + 0 = x` → 이종 전문가가 **기존 FFN과 수학적으로 동치**
- 학습이 진행되면서 Conv가 서서히 **잔차(residual)로** 공간 bias를 추가
- **baseline의 학습 성과를 1도 잃지 않으면서** 점진적으로 specialization

이 접근은 사실상 **fine-tuning**입니다. 새 모델을 학습하는 것이 아니라, **baseline 모델에 공간 처리 능력을 추가**하는 것.

### Shared Expert의 강화된 역할

Shared Expert는 이종 전문가 체제에서 오히려 **더 중요해집니다**:

| 구성 | Shared Expert 역할 |
|------|-------------------|
| **동질 (현재)** | 4개 전문가와 동일 구조 → 사실상 5번째 FFN, 앙상블 효과만 |
| **이질 (3-A)** | 4개 전문가가 각각 다른 spatial bias → Shared가 **bias-free baseline 보존자** |

이종 전문가 체제에서 Shared Expert는:
- 라우터의 실수를 보정 (잘못된 전문가에 보내도 shared가 기본 처리)
- 모든 토큰에 대한 **공통 표현력** 보장
- 학습 안정성의 안전망 (기존 PureFFN과 같은 역할이지만 **무조건 적용**)

---

## 6. 통합 전략 비교

### Strategy A: experts_v3을 4D Wrapper로 감싸기

MoE 토큰 라우팅 결과를 다시 `(B, T, N, D)` 형태로 재구성한 뒤 v3 전문가를 호출하는 방식.

- **장점**: 기존 코드 100% 재사용, v3의 복잡한 attention 로직 보존
- **단점**:
  - Sparse routing 시 비연속 토큰 → 2D/3D 공간 재구성 불가능 (치명적)
  - context와 conditioning을 MoE 내부로 추가 전달해야 하는 대규모 인터페이스 변경
  - STDiTBlock의 attention과 v3 내부 attention의 이중 연산 → 메모리와 속도 2배
- **평가**: 기술적 부채가 심각. **비추천.**

### Strategy B: v3의 inductive bias를 MLP-compatible 모듈로 재설계 ✅ 추천

v3 전문가들의 핵심 inductive bias(수용장 크기, 방향성 분리, 경량 처리)만 추출하여 `forward(x, h_shape, w_shape)` 인터페이스의 신규 모듈로 구현. baseline weight를 계승하고 conv를 zero-init.

- **장점**:
  - 기존 라우터·routing·load balancing 코드 **무변경**
  - Capacity routing과 자연스럽게 호환 (`h_shape=None` 시 conv skip → FFN fallback)
  - 상위 STDiTBlock의 attention과 역할 분리 명확 (전문가는 FFN 위치에서 **공간 처리 전략만 차별화**)
  - **baseline checkpoint에서 weight transfer** 가능 → 학습 안정성 극대화
  - v3 전문가 내부의 attention을 제거하고 Conv로 대체하므로 **파라미터 효율적**
- **단점**: v3의 attention 로직(특히 directional axial attention)의 정밀함이 conv로 얼마나 근사되는지는 실험적 검증이 필요함
- **평가**: 실현 가능성과 효과의 균형이 최적. **추천.**

### Strategy C: 하이브리드 (v3 보존 + 신규 추가)

`moe_4expert.py`의 채널 분할 MoE를 유지하면서, True MoE에도 별도의 이종 전문가를 추가.

- **장점**: v3을 완전히 보존, 독립적 효과 검증 가능
- **단점**: 모델 깊이/파라미터 2배 증가, 학습 불안정, 복잡도 과다
- **평가**: 연구 가치는 있으나, 현 단계에서는 과도한 복잡성. **보류.**

---

## 7. Strategy B 상세 설계

### 기상학적 전문가-현상 매핑

```
┌─────────────────── 기상 현상의 공간 스케일 ───────────────────┐
│                                                               │
│  행성 스케일        중규모             국소               균질 │
│  (1000km+)          (100km)           (10km)             배경 │
│                                                               │
│  기압골/능          전선대             적운군          맑은 바다 │
│  대기 순환          해안 수렴대        강수 셀         사막/대양 │
│  제트 기류          산맥 차폐          뇌우             평탄 지역 │
│                                                               │
│       ↓                 ↓                ↓                ↓   │
│  GlobalConvFFN    AxialConvFFN     LocalConvFFN      PureFFN  │
│  (RF: 7×7)        (RF: 1×5+5×1)    (RF: 3×3)        (RF: 1×1)│
│  전역 공간 통합    방향축 분리 처리  국소 정밀 처리   fast-track│
│                                                               │
│  ← experts_v3.Expert1_LLL의 정신 계승                        │
│  ←── experts_v3.Expert2_LLH의 정신 계승                      │
│  ←──── experts_v3.Expert3_LH의 정신 계승                     │
│  ←────── experts_v3.Expert4_H의 정신 계승                    │
└───────────────────────────────────────────────────────────────┘
```

### v3 → 신규 모듈 매핑 논리

| experts_v3 | 계승하는 핵심 bias | 신규 모듈 | RF |
|-----------|-------------------|----------|------|
| Expert1_LLL (Global ST Attn) | **장거리 공간 의존성** | GlobalConvFFN (대형 DWConv 7×7) | 7×7 |
| Expert2_LLH (Axial Attn) | **방향성 분리 처리** | AxialConvFFN (1×5 + 5×1 분리 DWConv) | ~5×5 |
| Expert3_LH (Windowed Attn) | **국소 영역 정밀 처리** | LocalConvFFN (소형 DWConv 3×3) | 3×3 |
| Expert4_H (3D DWConv) | **경량 통과 처리** | PureFFN (Conv 없음, Linear만) | 1×1 |

변환 핵심:
- v3에서 **attention이 수행하던 역할**을 **DWConv의 수용장**으로 대체합니다.
- 이유: MoE MLP 위치에서는 self/cross-attention이 이미 완료되었으므로, 전문가가 해야 할 일은 **공간적 귀납 편향(spatial inductive bias)** 주입뿐이며, 이는 Conv으로 충분히 달성됩니다.
- Expert4_H의 3D Conv(시간축 포함)은 시간축이 이미 사라진 MLP 위치에서 의미가 없으므로, 가장 경량인 PureFFN으로 대체합니다. 이는 "고주파 배경 토큰은 공간 처리 없이 fast-track 통과"라는 원래 설계 의도와 일치합니다.

### 라우터와의 시너지

현재 ST-Aware Router는 `Z = [X ∥ ΔX ∥ P_emb]`를 입력으로 받아 라우팅 결정을 내립니다.

- **ΔX (시간 변화량)이 큰 토큰** → 기상 변화가 활발한 영역 → GlobalConvFFN 또는 AxialConvFFN으로 라우팅 예상. 넓은 수용장으로 변화의 맥락을 파악할 수 있으므로.
- **ΔX가 작고 P_emb이 해안선 근처** → 시간 변화는 작지만 공간 경계가 존재 → AxialConvFFN으로 라우팅 예상.
- **ΔX가 작고 X 분산이 높은 토큰** → 구름 클러스터 내부의 복잡한 텍스처 → LocalConvFFN으로 라우팅 예상.
- **ΔX가 작고 X 분산도 낮은 토큰** → 균질한 배경 → PureFFN으로 라우팅 예상.

이러한 라우팅 패턴이 자연스럽게 발현되는지가 3-A의 **핵심 검증 대상**입니다.

### Capacity Routing 호환성

Capacity routing에서 각 전문가는 `(1, capacity, D)` 형태의 비연속 토큰을 받습니다.
- **Graceful Fallback**: `h_shape`가 전달되지 않거나 `N ≠ h_shape × w_shape`이면 Conv를 skip하고 FFN만 수행.
- 이 경우 GlobalConvFFN도 PureFFN과 동일하게 동작하지만, full-batch inference에서는 Conv가 정상 작동.

### 기존 Homogeneous와의 공존

**기존 동질 전문가 구성은 유지합니다.** Config 플래그 `--use_hetero_experts`로 전문가 생성부만 분기:

| Config | 전문가 구성 | 용도 |
|--------|-----------|------|
| 기본값 (미지정) | FFN × 4 (동질) | 기존 baseline, Phase 2 실험 |
| `--use_hetero_experts` | Global + Axial + Local + Pure (이질) | Phase 3-A 실험 |
| `--use_local_phase_preserver` | LocalPhasePreserverFFN × 4 (동질+Conv) | 레거시 |

---

## 8. 파라미터 및 연산량 분석

### 현재 Homogeneous (FFN × 4)
- 전문가 1개: `Linear(512→2048) + Linear(2048→512)` = **2.1M params**
- 4개 합계: **8.4M params**

### Strategy B Heterogeneous
| Expert | Conv 파라미터 | FFN 파라미터 | 합계 |
|--------|-------------|-------------|------|
| GlobalConvFFN (DWConv 7×7) | 512×49 = 25K | 2.1M | **2.13M** |
| AxialConvFFN (1×5 + 5×1) | 512×5×2 = 5K | 2.1M | **2.11M** |
| LocalConvFFN (DWConv 3×3) | 512×9 = 4.6K | 2.1M | **2.10M** |
| PureFFN | 0 | 2.1M | **2.10M** |
| **합계** | **~35K** | **8.4M** | **~8.43M** |

> DWConv 추가분은 전체의 **0.4%**에 불과합니다. 파라미터 오버헤드 무시 가능.

---

## 9. 단계적 Ablation 계획

### Step 1: Homogeneous vs Heterogeneous 직접 비교
```
8A-base:   Phase 2-Final 최적 config + FFN×4 (동질)
8A-hetero: Phase 2-Final 최적 config + --use_hetero_experts (이질)
```
- 측정: SSIM, FID, Latent_MSE_Low, Latent_MSE_High
- 분석: 대역별 MSE 변화량으로 어떤 전문가가 어떤 대역에 기여했는지 추론

### Step 2: 라우팅 패턴 시각화 분석
- 학습 완료 후 test set에서 각 토큰의 expert 할당 맵 시각화
- 기대 패턴: 해안선 토큰 → Expert 1(Axial), 배경 토큰 → Expert 3(Pure) 등
- 패턴이 기상학적으로 해석 가능한지가 **논문 기여점의 핵심**

### Step 3: 개별 전문가 기여도 검증 (Ablation)
하나씩 PureFFN으로 교체하여 각 전문가의 독립 기여 측정:

| 실험 | Expert 구성 | 검증 대상 |
|------|-----------|----------|
| 8A-no-global | **Pure** + Axial + Local + Pure | GlobalConvFFN의 기여 |
| 8A-no-axial | Global + **Pure** + Local + Pure | AxialConvFFN의 기여 |
| 8A-no-local | Global + Axial + **Pure** + Pure | LocalConvFFN의 기여 |
| 8A-all-conv | Global + Axial + Local + **Local** | PureFFN 제거 효과 |
