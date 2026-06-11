# solar-diffuse 코드베이스 조사 결과: DWT vs VAE 비교실험 설계

## 1. 발견된 재사용 가능 자산

### ✅ 학습 완료된 위성 전용 VAE
| 항목 | 값 |
|:---|:---|
| 체크포인트 | `/home/oem/solar-diffuse/models_vae_solar/best_vae.pt` (62MB) |
| 아키텍처 | `diffusers.AutoencoderKL` |
| 입출력 | 1ch → **4ch latent** → 1ch (512×512 → 64×64 → 512×512) |
| Block channels | (64, 128, 256, 256) — "다이어트" 버전 |
| 학습 데이터 | GK-2A 위성 일사량 `.npy`, 2021–2023 전체 |
| 학습 설정 | 50 epochs, AdamW(lr=1e-4), Phase1: L1 → Phase2(ep≥20): MS-SSIM+L1 |
| KL weight | 1e-6 (극도로 낮음, 거의 AE에 가까움) |
| 정규화 | MinMax [-1, 1] |
| 스크립트 | [train_vae.py](file:///home/oem/solar-diffuse/train_vae.py) |

### ✅ 사전 추출된 VAE Latent (19,682개)
- 경로: `/home/oem/solar-diffuse/latent_npy/`
- 형식: `.pt` (4×64×64 tensor), scale_factor=0.18215 적용됨
- 커버리지: 2021-01-01 ~ 2023-12-31 (3년 전체)
- 추출 스크립트: [extract_latents.py](file:///home/oem/solar-diffuse/extract_latents.py)

### ✅ VAE-DiT 실험 결과 (10개 variant)
이미 VAE latent 위에서 DiT를 훈련한 결과가 존재:

| 실험 | Patch | Hidden | Depth | Heads | **Test MSE** | **Test SSIM** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **DiT_Small_P2** | 2 | 768 | 12 | 12 | **0.0490** | **0.5351** |
| DiT_Small_P2_lowerHD | 2 | 384 | 1 | 1 | (학습 중단) | — |
| DiT_Small_P4 | 4 | — | — | — | (로그 미확인) | — |
| … (P4 variants ×4, P8 variants ×3) | | | | | | |

> **핵심**: P2 config (MSE=0.049, SSIM=0.535)는 현 DWT-DiT 9G와 유사한 아키텍처 규모에서 얻은 결과.

---

## 2. DWT vs VAE: 현재 코드 간 차이점 분석

공정한 비교를 위해 **DWT(ST-DiT2)와 VAE(solar-diffuse) 간 변수를 식별**:

| 변수 | ST-DiT2 (DWT) | solar-diffuse (VAE) | 공정 비교 시 통제 필요 |
|:---|:---|:---|:---:|
| **인코딩** | 3-level Haar DWT (64ch) | AutoencoderKL (4ch) | ← **비교 대상** |
| 정규화 | Z-score | MinMax [-1, 1] | ⚠️ |
| 예측 타겟 | **v-prediction** | **ε-prediction** | ⚠️ |
| 샘플링 | DDIM 10 steps | DDPM 1000 steps | ⚠️ |
| MoE | 8 Hetero + Shared, k=2 | 없음 (vanilla MLP) | ⚠️ |
| 손실 함수 | Per-Channel Min-SNR-γ | Standard MSE | ⚠️ |
| 후처리 | BayesShrink | 없음 | ⚠️ |
| Hidden dim | 512 | 768 | ⚠️ |
| Depth | 8 | 12 | ⚠️ |
| Cross-attention | context 5프레임 | context 5프레임 | ✅ 동일 |
| Patchification | 2×2 | 2×2 | ✅ 동일 |

> [!WARNING]
> **교란 변수가 9개나 존재**합니다. 현재 solar-diffuse 결과(MSE=0.049)를 그대로 DWT(9G)와 비교하면 "어떤 변수 때문에 차이가 났는지" 특정 불가.

---

## 3. 실험 설계 옵션

### Option A: 인코딩만 교체 (최소 침습, 추천)

ST-DiT2 코드베이스를 **그대로 유지**하되, DWT 인코딩만 VAE 인코딩으로 교체:

```
[변경점]
1. Dataset: DWT 3-level → VAE encode (best_vae.pt 사용)
2. in_channels: 64 → 4
3. IDWT 복원 → VAE decode 복원
4. Z-score 통계: DWT 64ch용 → VAE 4ch용 재계산
5. 나머지 전부 동일 (v-prediction, DDIM, MoE, Min-SNR, BayesShrink는 제거 or 유지)
```

**장점**: 유일한 변수 = 인코딩 방식 → RQ1에 대한 **깨끗한 답변** 가능
**소요 시간**: VAE latent 재추출(이미 있음) + 학습 100 epochs (~3-4시간)
**BayesShrink 문제**: VAE latent은 wavelet sub-band가 아니므로 BayesShrink 적용 불가 → **제거** (이것도 변수가 되지만, BayesShrink는 DWT 전용이므로 "DWT의 이점"으로 보고 가능)

### Option B: solar-diffuse 결과 직접 인용 (차선)

이미 존재하는 DiT_Small_P2 결과(MSE=0.049, SSIM=0.535)를 "VAE baseline"으로 Table에 넣기.

**장점**: 추가 실험 불필요
**단점**: 교란 변수 9개 → 리뷰어: "이건 공정한 비교가 아닙니다"

### Option C: VAE reconstruction error만 분석 (보조 증거)

VAE의 **인코딩-디코딩 복원 오차**(diffusion 제외)만 정량화:
```python
# VAE: encode → decode → MSE/SSIM vs original
# DWT: DWT → IDWT → MSE/SSIM vs original (이론적으로 0/1.0)
```

**장점**: 실험 1분이면 끝남, "lossless vs lossy" 주장에 대한 **직접 증거**
**단점**: 전체 forecasting 성능 비교는 아님

---

## 4. 재사용 가능 코드 요약

| 파일 | 용도 | 재사용 가능 여부 |
|:---|:---|:---|
| [best_vae.pt](file:///home/oem/solar-diffuse/models_vae_solar/best_vae.pt) | 학습된 VAE 가중치 | ✅ 그대로 사용 |
| [train_vae.py](file:///home/oem/solar-diffuse/train_vae.py) | VAE 재학습 | ✅ (재학습 불필요) |
| [extract_latents.py](file:///home/oem/solar-diffuse/extract_latents.py) | VAE latent 추출 | ✅ Z-score 통계 재계산만 필요 |
| [dataset_vae.py](file:///home/oem/solar-diffuse/dataset_vae.py) | VAE용 단일 프레임 Dataset | ⚠️ 시계열 Dataset으로 교체 필요 |
| [dataset_dit.py](file:///home/oem/solar-diffuse/dataset_dit.py) | VAE latent 시계열 Dataset | ✅ ST-DiT2 dataset 참고용 |
| [model_st_dit.py](file:///home/oem/solar-diffuse/model_st_dit.py) | Vanilla DiT (MoE 없음) | ⚠️ 참고만 (ST-DiT2가 더 발전) |
| [diffusion.py](file:///home/oem/solar-diffuse/diffusion.py) | DDPM 구현 | ❌ ST-DiT2의 v-prediction/DDIM 사용 |
| [engine_dit.py](file:///home/oem/solar-diffuse/engine_dit.py) | VAE decode 로직 | ✅ test 시 VAE decode 참고 |
| `/home/oem/solar-diffuse/latent_npy/` | 사전 추출 latent (19,682개) | ⚠️ Z-score 재정규화 필요 |

---

## 5. 추천 전략

> [!IMPORTANT]
> **Option A + Option C 병행**을 추천합니다.

### 즉시 실행 가능 (Option C): VAE reconstruction error 분석
- **소요**: 스크립트 작성 30분 + 실행 5분
- **산출물**: Table 1줄 — "VAE reconstruction MSE=X, SSIM=Y vs DWT reconstruction MSE=0, SSIM=1.0"
- **논문 위치**: §4.4 Ablation에 "Ablation 0: Latent Space Representation" 추가

### 3yr 완료 후 (Option A): 공정 비교 실험
- **소요**: latent Z-score 재계산 (30분) + 학습 100ep (3-4시간) + 평가 (30분)
- **변경**: ST-DiT2 코드에 `--latent_type vae` 플래그 추가
- **산출물**: §4.4에 DWT vs VAE Table (동일 DiT backbone, 동일 diffusion, 인코딩만 다름)

---

## 6. 결정이 필요한 사항

1. **Option A 실행 시 MoE를 포함할 것인가?**
   - 포함 → "전체 시스템에서 DWT의 기여" 측정
   - 제외 (vanilla MLP) → "순수 인코딩 차이"만 측정
   - 추천: **포함** (MoE는 DWT/VAE 모두에 적용 가능, 인코딩과 독립적)

2. **BayesShrink 처리**
   - VAE 실험에서는 자동으로 비활성화 (wavelet sub-band가 아니므로)
   - 이것을 "DWT가 가능케 하는 추가 최적화"로 논문에서 명확히 기술

3. **Phase 1 (4개월) vs Phase 2 (3년) 중 어디서 실험할 것인가?**
   - Phase 1: 빠름 (3-4시간), ablation으로 충분
   - Phase 2: 시간 많이 소요, main table용
