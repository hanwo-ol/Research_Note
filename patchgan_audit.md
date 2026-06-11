# Conditional PatchGAN 검수 보고

## 구성 요소

| 파일 | 역할 |
|:---|:---|
| [cond_patchgan.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/cond_patchgan.py) | 모델 정의 (ConditionalPatchGAN) |
| [train.py L565-585](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/engine/train.py#L565-L585) | Generator loss (G step) |
| [train.py L675-700](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/engine/train.py#L675-L700) | Discriminator loss (D step) |
| [pipeline.py L1536-1557](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/pipeline.py#L1536-L1557) | 초기화 |

---

## 데이터 흐름 추적

### G step (train.py L565-585)

```
1. x_noisy_c = √ᾱ_s · target + √(1-ᾱ_s) · ε        ← diffusion forward
2. x0_pred_c = (x_noisy_c - √(1-ᾱ_s)·pred_noise) / √ᾱ_s    ← x0 복원
3. pred_image_abs = x0_pred_c · σ_r + last_ctx_c      ← pixel 복원
4. g_loss = BCE(D(pred_image_abs, context), 1)         ← "진짜"로 속이려 함
```

### D step (train.py L675-700)

```
real_image = gt_absolute_pixel            ← 진짜 GT (residual 변환 전 보존)
fake_image = (x0_pred_det_c · σ_r + last_ctx_c).detach()   ← x0 복원 pixel
d_loss = 0.5 * (BCE(D(real, ctx), 1) + BCE(D(fake, ctx), 0))
```

---

## 발견된 결함

### 결함 #1 (치명적): x0_pred가 임의 timestep t에서의 noisy 복원

> [!CAUTION]
> **G step과 D step 모두에서 x0_pred는 임의의 diffusion timestep t에서 단일 스텝으로 복원한 것입니다.**
>
> 추론 시에는 DDIM 10 스텝으로 깨끗한 r̂₀을 얻지만, 학습 중 D에 입력되는 fake image는:
> ```
> x0_pred = (√ᾱ_s · r + √(1-ᾱ_s) · ε  -  √(1-ᾱ_s) · ε̂) / √ᾱ_s
> ```
> s가 큰 경우(high noise), √ᾱ_s ≈ 0 이므로 **x0_pred ≈ noise/0 → 매우 noisy**합니다.

**결과**: D는 "blurry vs sharp"를 구별하는 대신, **"noisy vs clean"을 구별하는 법**을 학습합니다. 이는 persistence 붕괴 방지에 전혀 도움이 되지 않습니다.

**비교 — 올바른 GAN 통합 방식 (예: 선행 연구)**:
- Denoising Diffusion GAN: D는 clean sample 또는 low-noise sample만 봄
- Adversarial Diffusion Distillation: D는 DDIM 1-step 출력을 봄 (t 고정)

### 결함 #2 (중요): D 입력의 학습/추론 도메인 갭

| | 학습 시 D가 보는 fake | 추론 시 실제 출력 |
|:---|:---|:---|
| **방식** | 단일 스텝 x0 복원 (noisy) | DDIM 10스텝 (clean) |
| **품질** | timestep에 따라 극도로 변동 | 항상 안정적 |
| **D의 학습** | noise artifact 탐지 | (적용 안 됨) |

D가 학습하는 "가짜"의 특성과 실제 추론 출력의 특성이 완전히 다르므로, 학습된 adversarial signal이 추론 품질 향상에 기여하지 못합니다.

### 결함 #3 (경미): context_for_cgan의 CFG dropout

```python
context_for_cgan = context[:, :, :1].clone() if cgan_active else None   # L279
# ... 이후 L286-292에서 context에 CFG dropout 적용
```

`context_for_cgan`은 L279에서 `.clone()`으로 보존되므로 CFG dropout 영향을 받지 않습니다. — **이 부분은 정상.**

---

## 종합 판정

| 항목 | 상태 |
|:---|:---|
| 모델 구조 (ConditionalPatchGAN) | ✅ 표준적인 PatchGAN 설계 |
| Spectral Norm | ✅ 학습 안정화에 적절 |
| G/D loss 함수 | ✅ 표준 BCE |
| D 입력 real (gt_absolute_pixel) | ✅ residual 변환 전 보존 |
| D 입력 fake (x0_pred at random t) | ❌ noisy 복원 — 치명적 도메인 불일치 |
| context_for_cgan 보존 | ✅ CFG dropout 이전 clone |
| D step gradient 차단 (.detach()) | ✅ 정상 |

> [!WARNING]
> **결론**: 현재 구현의 PatchGAN은 **persistence 붕괴 방지에 효과적이지 않습니다.** D가 보는 fake image가 임의 timestep의 noisy 단일 스텝 복원이므로, D는 "blurry vs sharp" 대신 "noisy vs clean"을 학습합니다.

---

## 수정 방향 (제안)

### 옵션 A: t-gating (최소 변경)

높은 noise timestep에서는 GAN loss를 비활성화하고, 낮은 t(clean에 가까운)에서만 적용:

```python
# G step에서 t-gating 추가
t_ratio = t.float() / diffusion.num_timesteps   # 0=clean, 1=noisy
cgan_mask = (t_ratio < 0.3)                       # 하위 30%만 활성
if cgan_mask.any():
    # cgan_mask에 해당하는 샘플만 D에 통과
```

### 옵션 B: 주기적 DDIM sampling (고비용, 고효과)

N epoch마다 현재 모델로 DDIM sampling을 수행하여 clean sample을 얻고, 이를 D에 입력:

```python
if epoch % 5 == 0:  # 5 epoch마다
    with torch.no_grad():
        clean_sample = diffusion.ddim_sample(model, ...)
    fake_clean = clean_sample * sigma_r + last_ctx
    g_adv = cond_g_adv_loss(D, fake_clean, context)
```

### 옵션 C: Consistency Training 스타일

diffusion 모델의 x0 prediction 특성을 활용하여, t=0 근처의 clean prediction만 사용:

```python
# prediction_type = "x0"이므로 model_output ≈ r̂ (clean residual)
# t와 무관하게 model_output 자체가 x0 추정
pred_image_abs = pred_noise * sigma_r + last_ctx  # pred_noise = model_output (x0 mode)
```

> [!TIP]
> **옵션 C가 가장 유력합니다.** `prediction_type = "x0"`에서 `model_output`은 사실상 clean x0 추정이므로, 복잡한 x0 역변환 없이 `model_output`을 직접 사용하면 됩니다. 단, x0 prediction 모드의 model_output 품질은 t에 따라 다르므로(높은 t에서는 부정확), 결국 옵션 A의 t-gating과 결합하는 것이 최선입니다.
