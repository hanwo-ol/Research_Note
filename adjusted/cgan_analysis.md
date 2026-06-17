# cGAN 실험 분석 — 성과와 실패 종합

## 1. 구현

**ConditionalPatchGAN** (`stdit/models/cond_patchgan.py`):
- 아키텍처: 6-layer PatchGAN (6.96M params)
- 입력: pred_image (B,1,512,512) + context_frames (B,5,1,512,512) → channel concat
- Spectral Norm + LeakyReLU, 30×30 patch logits 출력
- Warmup: `cgan_start_epoch` 이후 활성화 (default 20)

## 2. 실험 결과 (Dashboard 기준)

### Phase 1: 기본 cGAN weight sweep (ctx 없는 시절, 2021 only)

| Experiment | λ_cgan | MSE | PSNR | FID | 판단 |
|---|---|---|---|---|---|
| **patch16_cgan_w002** | 0.002 | 0.0419 | 20.89 | **7.88** | ✅ FID 대폭↓ |
| **patch16_cgan_w005** | 0.005 | 0.0438 | 20.74 | **4.82** | ✅ FID 최고 |
| **patch16_cgan_w010** | 0.010 | 0.0459 | 20.56 | 5.97 | MSE↑ |
| **patch16_a3_cgan_combo** | 0.005+A3 | 0.0443 | 20.69 | **3.64** |  **FID 최저** |

> [!IMPORTANT]
> **FID 3.64~4.82는 극적 성공!** GAN 없이 45+ 수준이던 FID가 한 자릿수로 떨어짐.
> MSE는 0.0419~0.0459로 약간 상승하지만 trade-off가 매우 유리.

### Phase 2: Pareto data scaling (y21f=2021 full, y3f=3년 full, y3q=3년 quarter)

| Data | λ_cgan | MSE | PSNR | FID | 문제 |
|---|---|---|---|---|---|
| y21f (2021 full) | 0.002 | 0.0372 | 21.71 | **79.64** | ❌ FID 대폭 악화! |
| y21f (2021 full) | 0.005 | 0.0403 | 21.26 | 20.07 | △ FID 중간 |
| y21f (2021 full) | a3_combo | 0.0505 | 20.35 | 65.06 | ❌ |
| y3q (3년 quarter) | 0.002 | 0.0363 | 21.34 | 50.23 | ❌ |
| y3q (3년 quarter) | a3_combo | 0.0377 | 21.13 | 30.06 | △ |
| **y3f (3년 full)** | **0.002** | **0.0330** | **22.12** | **101.16** | ❌❌ FID 최악 |

> [!WARNING]
> **데이터 양을 늘리면 cGAN의 FID 개선 효과가 완전 소멸!**
> - 2021 only: FID 4.82 → 3년 full: FID 101.16
> - MSE는 개선되지만 FID는 오히려 악화

## 3. 핵심 분석

### 성공 (Phase 1)

- λ=0.005에서 **FID 45 → 4.82** (92% 개선)
- A3(DWT HF loss)와 결합 시 **FID 3.64** 달성
- MSE 손실은 5~10%로 매우 수용 가능한 수준
- **cGAN은 FID 개선에 극적으로 효과적**

### 실패 (Phase 2 - Data Scaling)

데이터 확장 시 FID 악화 원인:
1. **D의 용량 부족**: 6.96M param D가 3년치 다양한 분포를 학습하기엔 부족
2. **D-G 비율 불균형**: 데이터↑ → G가 더 다양한 출력 → D가 못 따라감
3. **Conditional 불일치**: 과거 frames와 미래 예측의 관계가 다년도에서 복잡해짐
4. **"조건부 평균" 문제 심화**: 더 많은 데이터 = 더 많은 가능성의 평균 = 더 흐릿

### MAE vs FID 구조적 트레이드오프

```
Phase 1 (적은 데이터): MSE 0.044, FID 4.8  ← D가 분포 커버 가능
Phase 2 (많은 데이터): MSE 0.033, FID 101  ← D 용량 초과, 평균 수렴
```

## 4. 시사점

현재 **ctx_S_L_D (MAE 0.1062, FID 64.29)** 환경에서 cGAN을 재적용하면:

| 조건 | 예상 |
|---|---|
| 2021 only + ctx_SLD + cGAN | FID ↓↓ 가능 (Phase 1 성공 재현 기대) |
| cGAN λ=0.005 | 최적 weight (이미 검증됨) |
| 단, ctx 토큰이 추가 conditioning | D 아키텍처 조정 필요할 수 있음 |

> [!TIP]
> **즉시 실행 가능한 실험**: ctx_S_L_D + cGAN λ=0.005로 학습.
> Phase 1에서 검증된 효과가 ctx 토큰 환경에서도 재현되는지 확인.
