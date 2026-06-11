# Auxiliary Loss 계열 실험 종합 분석 보고서

## 1. 실험 배경

Auxiliary Loss는 확산 모델의 주 손실(v-prediction MSE) 외에 중간 표현(DWT 서브밴드) 또는 최종 출력(pixel space)에 대한 보조 감독 신호를 추가하여 구조적 복원력(SSIM)을 개선하려는 전략이다.

**기준선**: E1_XavierConv_D4 (SSIM 0.530, MSE 0.038, FID 204)

---

## 2. 실험 결과 요약

### 2.1 DWT Aux Loss 레벨별 비교 (E3 계열)

> [!NOTE]
> 15A baseline (SSIM 0.518) 대비 변화율. E3 계열은 Xavier Conv Init 적용 이전 실험.

| 실험 | Aux 레벨 | MSE | SSIM | FID | 판정 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 15A (baseline) | - | 0.038 | 0.518 | 200 | - |
| E3a_AuxL1 | l1 | **0.036** (-4.0%) | **0.527** (+1.7%) | 204 (+2.1%) | ✅ 최선 |
| E3b_AuxL0 | l0 | 0.038 (-0.8%) | 0.523 (+1.0%) | 216 (+8.3%) | ⚠️ FID 악화 |
| E3c_AuxL0L1 | l0+l1 | 0.041 (+7.4%) | 0.518 (0.0%) | 226 (+13.3%) | ❌ gradient 간섭 |

**핵심 발견**: l1(중간 레벨) 단독 감독이 최적. l0(pixel) 추가 시 gradient 간섭으로 오히려 악화.

---

### 2.2 Pixel-Space Aux SSIM 실험 (E1 baseline 기반)

> [!NOTE]
> E1 baseline (SSIM 0.530) 대비 변화율.

| 실험 | 구성 | MSE | SSIM | FID | 판정 |
|:---|:---|:---:|:---:|:---:|:---:|
| E1 (baseline) | Xavier Conv | 0.038 | 0.530 | 204 | - |
| E12B_AuxPixSSIM | AuxPixel SSIM 단독 | 0.039 (+5.1%) | 0.517 (-2.5%) | 217 (+6.3%) | ❌ |
| E3a_AuxPixSSIM | AuxDWT L1 + AuxPixel SSIM | 0.041 (+9.6%) | 0.444 (-16.2%) | 254 (+24.4%) | ❌ 심각한 악화 |

**핵심 발견**: Pixel-space SSIM aux loss는 단독이든 DWT aux와 조합이든 E1 대비 개선 효과 없음. 특히 두 aux 조합 시 심각한 성능 붕괴 발생.

---

### 2.3 Aux + PhysInput 조합 실험

| 실험 | 구성 | MSE | SSIM | FID | 판정 |
|:---|:---|:---:|:---:|:---:|:---:|
| E1 (baseline) | Xavier Conv | 0.038 | 0.530 | 204 | - |
| E3a_E6a_Combo | AuxL1 + Phys V+SG (pixel) | 0.040 (+5.3%) | 0.520 (-1.9%) | 211 (+3.3%) | ❌ |
| E3a_E6a_dwt_Combo | AuxL1 + Phys V+SG (DWT) | 0.040 (+5.3%) | 0.500 (-5.7%) | 221 (+8.2%) | ❌ |

**핵심 발견**: AuxL1과 PhysInput의 조합에서 시너지가 발생하지 않았다. 특히 DWT 공간에서 계산된 V+SG 피처는 단독(E6a_dwt_VSG, SSIM 0.392)에서도 심각한 악화를 보였으며, AuxL1을 추가해도 그 악화 효과가 지배적이다.

---

## 3. Gradient 간섭 분석

### 3.1 관찰된 패턴

실험 결과에서 일관된 패턴이 나타난다:

1. **단일 aux는 유효** (E3a: l1 단독, SSIM +1.7%)
2. **두 개 이상의 aux 조합 시 성능 급락** (E3c: l0+l1, E3a_AuxPixSSIM: l1+pixel)
3. **aux + phys 조합도 비효과적** (E3a_E6a_Combo)

### 3.2 원인 진단

- 확산 모델의 주 손실(v-prediction)은 noise schedule에 의해 정밀하게 가중치가 조율된 신호이다
- 보조 손실이 추가될수록 주 손실의 gradient 방향과 충돌하여 최적화 궤적이 불안정해진다
- 특히 pixel-space SSIM loss는 미분 불연속 구간이 존재하여 gradient 품질이 낮다
- l0(pixel 레벨)과 l1(중간 레벨) 동시 감독 시, 서로 다른 해상도의 감독 신호가 공유 backbone의 gradient를 상반된 방향으로 견인한다

---

## 4. E1 baseline과의 관계

> [!IMPORTANT]
> E1(Xavier Conv Init)이 현재 D4 1yr 역대 최고 SSIM(0.530)을 달성한 상태에서, **어떤 aux loss 조합도 E1을 개선하지 못했다**.

이는 Xavier Conv Init가 Conv 전문가의 활성화를 정상화하여 이미 구조적 복원력의 상당 부분을 확보했고, 추가적인 aux 감독이 오히려 이 균형을 깨뜨림을 시사한다.

| 전략 | E1 대비 SSIM 변화 |
|:---|:---:|
| AuxPixel SSIM 단독 (E12B) | -2.5% |
| AuxDWT L1 + AuxPixel SSIM (E3a_AuxPixSSIM) | -16.2% |
| AuxDWT L1 + Phys V+SG pixel (E3a_E6a_Combo) | -1.9% |
| AuxDWT L1 + Phys V+SG DWT (E3a_E6a_dwt_Combo) | -5.7% |

---

## 5. 최종 판정 및 권고

### 채택 가능

| 모듈 | 조건 | 근거 |
|:---|:---|:---|
| AuxDWT L1 단독 | Xavier Conv Init **미적용** 시 | 15A 대비 MSE -4.0%, SSIM +1.7% |

### 비채택 (근거 충분)

| 모듈 | 근거 |
|:---|:---|
| AuxDWT L0 단독 | FID +8.3% 악화. l1 대비 열위 |
| AuxDWT L0+L1 | gradient 간섭. 전 지표 악화 |
| AuxPixel SSIM 단독 | E1 대비 -2.5% SSIM |
| AuxDWT L1 + AuxPixel SSIM | -16.2% SSIM. 심각한 붕괴 |
| AuxL1 + PhysInput (pixel/DWT) | 시너지 부재. -1.9% ~ -5.7% SSIM |

### 결론

> [!IMPORTANT]
> E1(Xavier Conv Init) 환경에서 aux loss 계열은 추가적인 개선을 제공하지 않는다.
> Phase 3 아키텍처 혁신(EDM, Heun, Cross-Track 등)으로 전환하는 것이 합리적이다.

AuxDWT L1은 Xavier Conv Init와 조합 실험이 아직 수행되지 않았으므로, 만약 추가 검증이 필요하다면 E1 + AuxL1 단일 조합 실험을 고려할 수 있다. 다만, 위 결과의 일관된 패턴(aux 추가 시 E1 악화)을 고려하면 개선 가능성은 낮다.
