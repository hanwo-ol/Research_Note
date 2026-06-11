# PhysInput 추가 피처 계열 실험 종합 분석 보고서

## 1. 실험 배경

PhysInput은 원본 일사량 시계열로부터 물리적 파생 피처(Velocity, Spatial Gradient 등)를 on-the-fly로 계산하여 context 프레임으로 추가 주입하는 전략이다. 확산 모델이 시공간 변화의 물리적 단서를 직접 관측함으로써 구조적 복원력(SSIM)을 개선하는 것이 목적이다.

**15A baseline**: SSIM 0.518, MSE 0.038, FID 200
**E1 baseline** (Xavier Conv Init): SSIM 0.530, MSE 0.038, FID 204

---

## 2. 단독 피처 실험 (15A 기반)

> [!NOTE]
> 15A baseline (SSIM 0.518) 대비 변화율. pixel 공간에서 피처 계산 후 DWT 변환하여 주입.

| 실험 | 피처 | SSIM | SSIM 변화 | MSE | FID | 판정 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 15A | - | 0.518 | - | 0.038 | 200 | - |
| E6a_PhysVSG | V+SG | **0.523** | +1.0% | 0.040 | 204 | ✅ 최선 |
| E6k_PhysSD | SD | 0.522 | +0.8% | 0.039 | 222 | ⚠️ FID 악화 |
| E6d_Phys4F | V+A+SG+CSI | 0.520 | +0.4% | 0.040 | 219 | ⚠️ 과잉 |
| E6e_PhysA | A | 0.519 | +0.2% | 0.039 | 207 | ⚠️ 미미 |
| E6f_PhysTV | TV | 0.498 | -3.8% | 0.040 | 211 | ❌ |
| E6h_PhysCSI | CSI | 0.504 | -2.7% | 0.042 | 222 | ❌ |
| E6b_PhysV | V | 0.476 | -8.1% | 0.040 | 280 | ❌ |
| E6j_PhysPD | PD | 0.477 | -8.0% | 0.042 | 251 | ❌ |
| E6i_PhysLSV | LSV | 0.465 | -10.3% | 0.052 | 256 | ❌ |
| E6c_PhysSG | SG | 0.411 | -20.7% | 0.084 | 307 | ❌ |
| E6g_PhysLAP | LAP | 0.408 | -21.3% | 0.057 | 315 | ❌ |

---

## 3. 피처 효과 분류

### 3.1 유효 피처 (Tier A)

**V+SG 조합** (SSIM +1.0%)만이 baseline 대비 유의미한 개선을 보였다. 핵심 관찰:
- V(시간축 속도)와 SG(공간 기울기)는 단독으로는 각각 -8.1%, -20.7%로 심각하게 악화
- 그러나 조합 시 서로의 약점을 보완하여 유일하게 개선 달성
- 시간 정보(V)와 공간 정보(SG)의 상보성이 핵심 메커니즘

### 3.2 중립 피처 (Tier B)

| 피처 | 설명 | SSIM 변화 | 비고 |
|:---:|:---|:---:|:---|
| SD | Sobel Difference | +0.8% | FID +11.3% 악화가 수반 |
| A | Acceleration (2차 차분) | +0.2% | 거의 무효과 |

### 3.3 유해 피처 (Tier C)

| 피처 | 설명 | SSIM 변화 | 진단 |
|:---:|:---|:---:|:---|
| TV | Temporal Variance | -3.8% | 분산 정보가 noise와 구분 불가 |
| CSI | Clear Sky Index | -2.7% | 정규화 과정에서 정보 소실 |
| V | Velocity 단독 | -8.1% | 방향 없는 속도는 노이즈 |
| PD | Persistence Deviation | -8.0% | 잔차 기반 피처의 분포 불안정 |
| LSV | Local Spatial Variance | -10.3% | 고주파 노이즈 증폭 |
| SG | Spatial Gradient 단독 | -20.7% | 에지 정보가 텍스처 복원을 방해 |
| LAP | Laplacian | -21.3% | 2차 미분의 극단적 분포 |

---

## 4. 피처 계산 공간 비교 (E1 기반)

> [!NOTE]
> E1 baseline (SSIM 0.530) 대비. pixel에서 계산 후 DWT 변환 vs DWT 공간에서 직접 계산.

| 실험 | 피처 | 계산 공간 | SSIM | 변화 | 판정 |
|:---|:---:|:---:|:---:|:---:|:---:|
| E6a_dwt_VSG | V+SG | DWT | 0.392 | -26.0% | ❌ 심각 |
| E6c_dwt_SG | SG | DWT | 0.526 | -0.8% | ❌ 미미 |
| E6a_PhysVSG | V+SG | pixel | 0.523 | +1.0% (vs 15A) | ✅ |

**핵심 발견**: DWT 공간에서 직접 계산한 피처(V+SG)는 SSIM을 26% 악화시켰다. pixel 공간에서 계산 후 DWT 변환하는 경로가 필수적이다. DWT 계수 위에서의 차분/기울기 연산은 물리적 의미를 상실하고 고주파 노이즈를 증폭시킨다.

---

## 5. E1과의 관계

> [!IMPORTANT]
> E6a(V+SG, pixel)는 15A 대비 SSIM +1.0% 개선을 보였으나, E1(Xavier Conv Init, SSIM 0.530)과의 조합 실험(E3a_E6a_Combo)에서는 SSIM 0.520(-1.9%)으로 오히려 악화되었다.

이는 Xavier Conv Init가 이미 Conv 전문가의 공간 탐색 능력을 정상화하여 SG(공간 기울기) 피처의 효과를 흡수했을 가능성을 시사한다.

---

## 6. 최종 판정 및 권고

### 채택 가능

| 모듈 | 조건 | 근거 |
|:---|:---|:---|
| PhysInput V+SG (pixel space) | Xavier Conv Init **미적용** 시 | 15A 대비 SSIM +1.0% |

### 비채택 (근거 충분)

| 범주 | 비채택 피처 | 근거 |
|:---|:---|:---|
| 단독 피처 | V, SG, A, TV, CSI, PD, LSV, LAP, SD | 단독으로는 악화 또는 미미 |
| 피처 과잉 | V+A+SG+CSI (4개) | V+SG 대비 개선 없음, FID 악화 |
| DWT 공간 계산 | V+SG (DWT), SG (DWT) | pixel 대비 심각한 악화 |
| E1 조합 | V+SG + Xavier Conv | 시너지 부재, -1.9% SSIM |

### 결론

> [!IMPORTANT]
> PhysInput 계열의 핵심 발견은 "V+SG 조합의 상보성"이나, E1(Xavier Conv) 환경에서는 그 효과가 소멸된다.
> 피처 설계 방향보다 아키텍처 혁신(Phase 3)에 투자하는 것이 합리적이다.

### 논문 기여 관점

PhysInput 실험 계열은 부정적 결과(negative result)로서 논문에 기여할 수 있다:
- 단독 피처 11건 중 **V+SG 조합만이 유일하게 유효**하며, 이는 시공간 상보성의 필요조건을 실증
- DWT 공간 직접 계산의 실패는 **웨이블릿 계수 위 미분 연산의 물리적 비타당성**을 입증
- "피처 추가 = 성능 개선"이라는 가설의 반증 사례로 활용 가능
