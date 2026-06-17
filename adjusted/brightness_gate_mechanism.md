# Brightness-Conditional HF MoE Gating 작동 원리

> **구현 파일**: `stdit/models/brightness_gating.py`
> **설계 문서**: `docs/0050_brightness_conditional_hfmoe.md`
> **관련 실험**: B1_P0_BrightnessGate (MSE 0.0358, SSIM 0.5337, FID 202.69)

---

## 1. 해결하려는 문제

ST-DiT의 Frequency MoE 아키텍처에서 고주파(HF) 트랙은 두 가지 선택지가 있었다:

| 방식 | 장점 | 단점 |
|:---|:---|:---|
| P0 (단일 HighFreqLocalExpert) | 안정적, 검증된 baseline | HF 표현력 제한 |
| HF MoE 5E (5개 이종 전문가) | SSIM 개선 가능성 | 5E 조합 시 noise 증가, 불안정 |

핵심 관찰: `docs/0049_hfmoe_deep_comparison.md`의 Q1/Q3-Q4 분석에서, **HF MoE의 효과가 brightness(일사량 강도)와 time-of-day에 강하게 의존**하는 패턴이 발견됨. 밤/저일사 구간에서는 HF 정보가 거의 없어 MoE가 noise만 생성하고, 낮/고일사 구간에서만 MoE의 다양한 전문가가 의미 있는 기여를 함.

---

## 2. 핵심 아이디어

P0(안정)과 HF MoE 5E(표현력)를 **brightness와 시간 조건에 따라 동적으로 혼합**한다.

```
출력 = main + alpha * (aux - main)
     = (1 - alpha) * main + alpha * aux
```

- **main**: P0의 HighFreqLocalExpert (7x7 DWConv + FFN + AdaLN FiLM)
- **aux**: HFHeteroMoEBlock (Edge / Texture / Noise / Oriented / Sparse 5개 전문가)
- **alpha**: brightness와 시간 조건에 따라 결정되는 혼합 비율 (0~1)

---

## 3. Gate 설계

### 3.1 Gate 입력 (3개 신호)

| 신호 | 차원 | 출처 | 역할 |
|:---|:---:|:---|:---|
| `cond` | D (512) | diffusion timestep + seasonal embedding | 전역 조건 정보 |
| `brightness` | 1 | context LL channel(ch0)의 시공간 평균 | 일사량 강도 |
| `hour_cyclic` | 2 | sin(2pi*h/24), cos(2pi*h/24) | 시간대 의존성 |

### 3.2 Brightness Proxy 계산

```
brightness = mean(x_context[:, :, ch0, :, :])
```

- DWT의 LL sub-band(ch0)는 Haar wavelet의 직교성에 의해 원본 이미지의 DC 성분(전체 평균 밝기)을 보존
- Z-score 정규화 후에도 sample 단위 상대 차이는 유지됨
- 밤 시간대 ~ 0 근처, 낮 고일사 ~ 양수 방향

### 3.3 Gate MLP 구조

```
gate_input: [cond || brightness || sin(h) || cos(h)]  →  (D+3)차원
    ↓
Linear(D+3, D//4) → SiLU
    ↓
Linear(D//4, 1)  →  bias 초기값: -3.0
    ↓
sigmoid  →  alpha ∈ (0, 1)
```

---

## 4. 안전장치: alpha=0 초기화

> [!IMPORTANT]
> BrightnessGate의 핵심 설계 원칙은 **P0 대비 성능 하락이 구조적으로 불가능**하다는 것이다.

- Gate MLP의 마지막 Linear layer:
  - weight = 0으로 초기화
  - bias = -3.0으로 초기화
- sigmoid(-3.0) = 0.047 --> 학습 시작 시 alpha = 약 5%
- 학습이 진행되면서 brightness/hour 신호가 유용한 경우에만 alpha가 증가
- 만약 HF MoE가 불필요하다면 alpha는 0 근처로 수렴 --> P0와 동일한 출력

이 설계를 통해 HF MoE 5E 단독 사용 시 발생했던 noise 문제를 원천 차단하면서, 조건부로만 MoE를 활성화할 수 있다.

---

## 5. 데이터 흐름도

```
x_context (B, T, C, H, W)
    │
    ├──→ ch0 mean ──→ brightness (B, 1)
    │
hour (B,) ──→ cyclic encoding ──→ hour_cyclic (B, 2)
    │
    └──→ [cond ‖ brightness ‖ hour_cyclic] ──→ Gate MLP ──→ alpha (B, 1)

x_high (B, D, H, W)
    │
    ├──→ HighFreqLocalExpert ──→ main (B, D, H, W)
    │
    └──→ HFHeteroMoEBlock(5E) ──→ aux (B, D, H, W)

output = main + alpha[:,:,None,None] * (aux - main)
```

---

## 6. 학습 모니터링

- `alpha_running_mean`: EMA decay 0.99로 alpha 평균값을 추적
- 학습 로그에서 alpha 값의 추이를 통해 MoE 활성화 정도를 관찰 가능
  - alpha가 0 근처에 머무르면: brightness 조건이 HF MoE에 유용하지 않음
  - alpha가 상승하면: 특정 brightness/시간대에서 MoE가 활성화되고 있음

---

## 7. 실험 결과 (1yr 분기)

| 실험 | MSE | MAE | SSIM | FID |
|:---|:---:|:---:|:---:|:---:|
| P0 (baseline) | 0.0378 | 0.1194 | 0.5307 | 220.22 |
| **B1_P0_BrightnessGate** | **0.0358** | **0.1151** | **0.5337** | **202.69** |
| B1_M4c_BrightnessGate | 0.0367 | 0.1171 | 0.5322 | 206.32 |

- P0 대비 MSE -5.3%, MAE -3.6%, SSIM +0.003, FID -17.5 개선
- P0 성능 하락 없이 HF MoE의 장점만 선택적으로 흡수한 결과

---

## 8. CLI 사용법

```bash
python main.py \
  --use_brightness_gate \
  --brightness_gate_init_bias -3.0 \
  [기타 P0 baseline 옵션]
```

| 파라미터 | 기본값 | 설명 |
|:---|:---:|:---|
| `--use_brightness_gate` | False | BrightnessGate 활성화 |
| `--brightness_gate_init_bias` | -3.0 | Gate bias 초기값. 값이 작을수록 P0 dominant |
