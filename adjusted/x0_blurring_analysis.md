# x0 Prediction의 구조적 블러링 문제 분석

## 1. 핵심 메커니즘: "왜 x0이 블러를 만드는가"

### 수학적 증명

x0 prediction에서 모델 학습 목표:

```
min_θ  𝔼[L(f_θ(x_t, t), x_0)]    ← target = x_start 직접
```

**L이 MSE/Charbonnier/L1 같은 pixel-wise regression loss일 때**, 최적 해는:

```
f*(x_t, t) = 𝔼[x_0 | x_t]       ← conditional mean
```

> [!WARNING]
> Conditional mean = **가능한 모든 x_0의 가중 평균** → 본질적으로 blurred.

이것은 prediction_type과 무관한 regression loss의 근본 한계이지만, **x0 mode에서 가장 직접적으로 드러남**.

### 현재 설정의 문제

| 설정 | 현재 값 | 블러링 영향 |
|---|---|---|
| `prediction_type` | **x0** | 모델 출력 = pixel space 직접 → 블러가 곧 최종 결과 |
| `loss_type` | **charbonnier** | √(e²+ε²) ≈ L1. MSE보다는 낫지만 여전히 regression |
| `loss_weighting` | **none** | 모든 t에 균일 가중 → low-t (detail 중요)에 불충분 가중 |
| `inference_timesteps` | **10** | Few-step → 각 step에서 x0 estimate 직접 사용 → 블러 누적 |

### 왜 ε-prediction에서는 덜한가

```
ε-prediction: target = noise (Gaussian)
```

- noise space에서의 conditional mean은 **실제 noise의 평균**이므로 여전히 정확
- pixel space 블러는 **sampling trajectory에서 점진적으로 보정** (many steps)
- **하지만** few-step (10 steps)에서는 ε도 블러 문제 동일하게 발생

### v-prediction의 장점

```
v = √α_t · ε - √(1-α_t) · x_0
```

- high-t (noisy): v ≈ ε (noise prediction과 유사)
- low-t (clean): v ≈ -x_0 (x0 prediction과 유사)
- **자연스러운 SNR-adaptive weighting** 효과 → high-t에서 블러 패널티 감소

---

## 2. 8-9월 성능 악화와의 연관

> [!IMPORTANT]
> **분산이 큰 데이터 = conditional mean이 더 블러해짐**

```
Var[x_0 | x_t] ∝ 기상 불확실성
```

| 월 | 기상 특성 | 조건부 분산 | 블러 정도 |
|---|---|---|---|
| 1-3월 | 맑음, 안정 | 낮음 | 약 |
| 4-6월 | 전이기 | 중간 | 중 |
| **7-8월** | **장마, 태풍** | **매우 높음** | **강 ← 핵심!** |
| **9월** | **태풍, 급변** | **높음** | **강** |
| 10-12월 | 건조, 안정 | 낮음 | 약 |

**x0 prediction + regression loss + 높은 기상 분산 = 8-9월 블러 극대화**

이것은 MoE/Monthly Token으로 **조건 세분화**하면 개선되지만, **근본 해결은 아님**.

---

## 3. Prediction Type 비교

### x0 vs ε vs v: 블러 메커니즘

```
                x0-prediction    ε-prediction    v-prediction
target space    pixel (direct)   noise           velocity
loss gradient   ∂L/∂pixel        ∂L/∂noise       ∂L/∂v (mixed)
blur source     direct output    indirect (via    balanced
                = mean           sampling loop)
SNR weighting   implicit:        implicit:        implicit:
                all-t uniform    high-t dominant  naturally balanced
few-step (10)   severe blur      moderate blur    moderate blur
PSNR/MAE        ↑ (by design)    ↓ (indirect)     balanced
FID             ↓ (blurry)       ↑ (sharper)      ↑ (sharper)
```

> [!CAUTION]
> x0 prediction은 **MAE/PSNR은 최적화하기 쉽지만, FID/perceptual quality를 희생**합니다.
> 이것이 "MAE는 좋은데 결과가 블러하다"는 현상의 근본 원인입니다.

### 현재 코드 지원 현황

[gaussian.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L575-L588)에서 **이미 v-prediction과 ε-prediction을 모두 지원**:

```python
if self.prediction_type == "v":
    target = sqrt_alpha_t * noise - sqrt_one_minus_alpha_t * x_start
elif self.prediction_type == "x0":
    target = x_start
else:  # epsilon
    target = noise
```

**→ `--prediction_type v`로 전환만 하면 즉시 v-prediction 학습 가능.**

---

## 4. 실현 가능한 대안들 (현 코드에서)

### 대안 A: v-prediction 전환 (가장 권장)

```bash
--prediction_type v --loss_weighting min_snr --min_snr_gamma 5.0
```

- **장점**: 자연스러운 SNR balancing, few-step에서도 안정, 블러 감소
- **단점**: 기존 x0 ckpt 호환 불가 (처음부터 재학습)
- **예상 효과**: FID ↓(개선), MAE 미세 ↑(트레이드오프)
- **구현 난이도**: 하 (이미 지원됨, flag만 변경)

### 대안 B: x0 유지 + Min-SNR Weighting

```bash
--prediction_type x0 --loss_weighting min_snr --min_snr_gamma 5.0
```

- **장점**: 기존 ckpt fine-tune 가능, low-t (detail phase)에 가중
- **효과**: 블러 **경감** (근본 해결은 아님)
- **구현 난이도**: 하 (이미 구현되어 있음)

### 대안 C: x0 + Lp Loss (p=1.5)

```bash
--prediction_type x0 --loss_type lp --lp_power 1.5
```

- **장점**: 큰 error에 더 큰 gradient → 블러의 "평균으로 수렴" 방지
- **단점**: 작은 error에서 L2보다 gradient 약해질 수 있음
- **이미 구현**: [gaussian.py L596-599](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/diffusion/gaussian.py#L596-L599)

### 대안 D: Perceptual/Adversarial Loss 추가 (중장기)

- LPIPS loss 추가 → 블러 직접 패널티
- GAN discriminator → sharpness 강제
- **구현 난이도**: 상 (새 모듈 필요)

---

## 5. 권장 실험 순서

| 우선순위 | 변경 | 기존 ckpt 호환 | 기대 효과 |
|---|---|---|---|
| **1** | `loss_weighting=min_snr` (x0 유지) | ✅ fine-tune 가능 | 블러 경감, MAE 유지 |
| **2** | `loss_type=lp, lp_power=1.5` | ✅ fine-tune 가능 | 극단값 보존 |
| **3** | `prediction_type=v` (전면 전환) | ❌ 처음부터 학습 | 근본 해결 |
| 4 | v + min_snr + charbonnier | ❌ | 최적 조합 |

> [!TIP]
> 대안 1, 2는 **기존 tgate_baseline ckpt에서 fine-tune으로 빠르게 검증 가능**합니다.
> 대안 3은 100 epoch 재학습이 필요하지만 **근본적 해결책**입니다.

---

## 6. 요약

```
x0 prediction + regression loss + 높은 기상 분산(8-9월)
     ↓
모델이 conditional mean으로 수렴
     ↓
구조적 블러링 (특히 고분산 달에 심화)
     ↓
MAE는 "최적"이지만 FID/LPIPS는 악화
```

**Monthly Token / MoE는 conditional mean의 "조건"을 세분화하여 블러를 줄이는 접근이고,
prediction type / loss weighting 변경은 "loss landscape 자체"를 바꾸는 더 근본적 접근.**

두 접근은 **직교적이므로 병행 가능**.
