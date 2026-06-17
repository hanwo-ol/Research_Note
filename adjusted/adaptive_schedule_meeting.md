# Adaptive Noise Schedule 재설계 회의 보고서

> 일시: 2026-04-24
> 참석: 검토자 관점 (학술 검토), 도메인 관점 (태양광 예보/확산모델)
> 안건: 13B 실험 실패 원인 분석 및 Adaptive Noise Schedule 재설계 방향 논의

---

## 1. 실패 원인 근본 분석

DDIM reverse process는 다음 관계에 의존한다:

$$x_0 = \frac{x_\tau - \sqrt{1-\bar{\alpha}_\tau} \cdot \epsilon_\theta}{\sqrt{\bar{\alpha}_\tau}}$$

이 역변환이 정상 작동하려면, 학습 시 모델이 본 noise 수준과 추론 시 noise 수준이 **공간적으로 일관**되어야 한다. pixel-wise alpha_bar를 적용하면:

1. **학습 시**: 같은 이미지 내에서 pixel A는 alpha_bar=0.7, pixel B는 alpha_bar=0.3을 본다. 모델의 self-attention은 두 pixel을 함께 처리하므로, "이 영역은 noisy, 저 영역은 clean"이라는 비균질 입력을 받는다.

2. **추론 시**: 모델이 예측한 v(또는 epsilon)로 한 step을 역변환할 때, pixel A와 B에 서로 다른 alpha_bar를 적용한다. 그러나 Transformer의 출력은 **전체 이미지를 보고 생성한 전역적 예측**이므로, 부분적으로 다른 역변환 계수를 적용하면 공간적 일관성이 깨진다.

3. **SSIM의 민감점**: SSIM은 local window(11x11) 내의 구조적 상관을 측정한다. 같은 window 내에서 alpha_bar가 다르면 denoising 후 구조적 상관이 왜곡된다.

핵심: **모델은 "전역 예측"을 하는데, 역변환은 "국소 계수"를 적용하는 불일치**가 SSIM 파괴의 원인이다.

checkthis의 Region-Adaptive Schedule도 pixel-wise alpha_bar를 사용한다. 그런데 왜 작동하는가?

1. **U-Net vs Transformer**: U-Net은 다운샘플-업샘플 구조에서 **국소 수용장**으로 처리한다. Conv 필터가 local patch를 독립적으로 denoising하므로, 인접 영역의 noise 수준이 달라도 각 영역을 독립적으로 처리할 수 있다. 반면 DiT의 self-attention은 모든 patch를 전역적으로 참조한다.

2. **Binary mask vs Continuous modulation**: checkthis는 Otsu로 2개 영역(cloud/terrain)을 분리한다. 각 영역 내부는 **완전히 균일한 schedule**이 적용된다. 경계에서만 불연속이 발생하지만, 영역 내부의 공간 일관성은 유지된다. 우리의 continuous modulation은 모든 pixel이 다른 schedule을 가지므로 일관 영역이 없다.

3. **beta_end의 차이가 작다**: cloud 0.02 vs terrain 0.05. alpha_bar로 환산하면 두 schedule의 차이는 크지 않다. 우리의 clamp [0.5, 2.0]은 alpha_bar 지수를 2배까지 변화시키므로 훨씬 공격적이다.

---

## 2. 재설계 방향 논의

### 방향 A: Schedule-side를 포기하고 Loss-side로 이동


| 항목 | 내용 |
|:---|:---|
| 핵심 아이디어 | noise schedule은 표준 유지. 대신 **loss에 공간 가중치**를 적용 |
| 수식 | $L = \sum_{i,j} w_{i,j} \cdot \|v_{pred} - v_{target}\|^2_{(i,j)}$, $w_{i,j} = f(\hat{V}'_{i,j})$ |
| 장점 | diffusion process를 전혀 건드리지 않으므로 이론적 안정성 보장. DDIM 역변환의 공간적 일관성 유지 |
| 단점 | per-channel SNR과 공간 가중치가 교차하여 loss landscape이 복잡해질 수 있음 |
| 구현 난이도 | 낮음 (p_losses에서 loss 계산 후 variance_map으로 element-wise weighting) |

의견: 개념적으로 가장 안전하다. 다만 loss weighting은 "모델에게 특정 영역을 더 열심히 학습하라"는 것이지, "특정 영역의 신호를 더 보존하라"는 것은 아니다. 학습이 수렴한 후에는 고분산 영역의 예측이 개선되겠지만, 추론 시 noise schedule이 동일하므로 신호 보존 효과는 없다.

### 방향 B: Variance Map을 Conditioning으로 주입


| 항목 | 내용 |
|:---|:---|
| 핵심 아이디어 | noise schedule은 표준 유지. variance map을 모델의 **추가 conditioning**으로 주입 |
| 수식 | $v_\theta(x_\tau, \tau, \text{context}, \hat{V}')$ -- variance map이 context embedding에 합산 또는 cross-attention |
| 장점 | 모델이 "이 영역은 변화가 크다"는 정보를 활용하여 denoising 전략을 자체 학습. Schedule 변경 없이 정보 활용 |
| 단점 | 아키텍처 변경 필요 (DiT의 condition injection 경로 추가). 학습 데이터에 의존 |
| 구현 난이도 | 중간 (spatial conditioning을 DiT에 주입하는 경로 설계 필요) |

의견: 이론적으로 가장 타당하다. 모델이 variance 정보를 어떻게 활용할지를 학습하게 두는 것이 하드코딩된 schedule 변경보다 유연하다. 다만 prototyping에서 효과를 보려면 학습 epoch이 충분해야 한다.

### 방향 C: 주파수 대역별 Schedule (채널 축, 공간 축 아님)

| 항목 | 내용 |
|:---|:---|
| 핵심 아이디어 | 공간적 modulation을 포기하고, **DWT 주파수 대역(채널 그룹)별로 다른 schedule** 적용 |
| 수식 | LL channels: $\bar{\alpha}_\tau^{LL}$, HH channels: $\bar{\alpha}_\tau^{HH}$ (LL은 느린 noising, HH는 빠른 noising) |
| 장점 | 공간적 일관성 완전 유지 (같은 채널 내에서는 모든 pixel이 동일 schedule). DWT의 주파수 분리 특성과 자연스럽게 결합 |
| 단점 | per-channel SNR weighting과 기능적 중복 가능성. Loss-side vs Schedule-side의 차이를 분리해야 함 |
| 구현 난이도 | 낮음 (q_sample에서 채널 그룹별 다른 alpha_bar 적용) |

의견: per-channel SNR은 **loss 가중치**이고, 대역별 schedule은 **noise 주입량** 자체를 바꾸는 것이다. 이 둘은 다른 메커니즘이다. LL 대역의 신호를 더 보존하면 denoising 시 대규모 공간 구조(일사량의 공간 패턴)가 더 잘 유지될 수 있다.

의견: 이중 보정 리스크를 피하려면, 이 방향을 채택할 경우 per-channel SNR weighting은 비활성화하고 ablation 비교해야 한다.

### 방향 D: 전역 스칼라 Modulation (per-image, not per-pixel)

| 항목 | 내용 |
|:---|:---|
| 핵심 아이디어 | variance map의 **전역 평균**을 스칼라로 사용하여 이미지 전체의 schedule을 조절 |
| 수식 | $\bar{\alpha}_\tau^{(b)} = \bar{\alpha}_\tau^{1/\bar{V}'_b}$, $\bar{V}'_b = \text{mean}(\hat{V}'_b)$ (배치 내 각 이미지별 단일 스칼라) |
| 장점 | 공간적 일관성 완전 유지. 고변동 이미지는 전체적으로 느린 noising, 저변동 이미지는 빠른 noising. 구현 단순 |
| 단점 | 공간 선택성 없음. "이미지 전체가 고변동"인 경우만 효과적이고, "일부만 고변동"인 경우 대응 불가 |
| 구현 난이도 | 매우 낮음 (variance_map.mean()만 사용) |

의견: 가장 보수적이고 안전한 접근이라고 본다. SSIM을 파괴하지 않을 가능성이 높다. 다만 효과 크기가 미미할 수 있다.

---

## 3. 우선순위 합의

| 우선순위 | 방향 | 이유 |
|:---:|:---|:---|
| 1 | **A (Loss-side 공간 가중치)** | 구현 최소, 리스크 최소, diffusion 이론 불변. 효과가 없더라도 학습에 손상 없음 |
| 2 | **D (전역 스칼라 modulation)** | A와 직교하므로 동시 적용 가능. 공간 일관성 유지 |
| 3 | **C (주파수 대역별 schedule)** | 효과가 있다면 DWT 기반 모델의 고유 장점으로 논문 contribution 가치 높음. 단, per-channel SNR과의 관계 정리 필요 |
| 4 | **B (Conditioning 주입)** | 가장 유연하나 아키텍처 변경 + 충분한 학습 필요. 장기 과제 |

---

## 4. 즉시 실행 가능한 실험

### 실험 13C: Loss-side Spatial Variance Weighting

variance map을 loss 가중치로만 사용. schedule은 표준 유지.

```
w_{i,j} = 1 + lambda * (V_hat'_{i,j} - 1)
```

lambda=0이면 기존과 동일, lambda=1이면 고분산 영역에 2배 가중치.
lambda는 0.5로 시작 (보수적).

### 실험 13D: Per-Image Global Schedule Modulation

variance map의 배치별 평균을 스칼라로 사용. clamp [0.9, 1.1] (매우 보수적).

---

## 5. 핵심 교훈

| 교훈 | 설명 |
|:---|:---|
| **Transformer는 전역 예측기** | self-attention 기반 모델에서는 pixel-wise schedule 변경이 구조적으로 부적합하다. U-Net의 local receptive field와 근본적으로 다르다. |
| **Schedule vs Loss의 역할 분리** | schedule 변경은 "모델이 보는 것"을 바꾸고, loss 변경은 "모델이 학습하는 방향"을 바꾼다. 전자가 더 위험하다. |
| **이중 보정 방지** | 같은 축(채널/공간)에서 schedule과 loss를 동시에 조절하면 과교정된다. 한 축은 하나의 메커니즘만 담당해야 한다. |
| **공정 비교의 중요성** | 파이프라인 차이(test_only.py vs main.py)가 결과를 왜곡할 수 있다. 동일 파이프라인 비교 필수. |
