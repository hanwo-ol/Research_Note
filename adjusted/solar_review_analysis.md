# 이전 연구(UNet+CBAM+FiLM) vs ST-DiT Raw Pixel: 구조 검수

> 대상 코드: `~/solar_review/` (model_gg.py, dataset_gg_nmode_v2.py, engine_gg.py, loss_ms.py)
> 비교 대상: ST-DiT `--use_raw_pixel --prediction_type x0` Stage8 계열 실험

---

## 1. 아키텍처 비교

| 항목 | 이전 연구 (UNet+GLCM) | ST-DiT Raw Pixel |
|:---|:---|:---|
| **모델 유형** | U-Net (Deterministic) | Diffusion Transformer (Generative) |
| **예측 방식** | 단일 forward pass로 직접 예측 | DDIM 10-step iterative sampling |
| **Skip Connection** | 4단계 encoder-decoder skip | 없음 (flat Transformer) |
| **공간 귀납 편향** | Conv2d 전체 (3x3 kernel) | Patch embedding 후 self-attention |
| **입력 채널** | 7ch (5-frame + PW + Sobel_PW) | 1ch x 5-frame (context sequence) |
| **해상도** | 512x512 전체를 Conv로 처리 | 512x512를 16x16 patch로 분할 (32x32 tokens) |
| **파라미터 수** | ~31M (U-Net 64-128-256-512-1024) | ~17M (depth=4, hidden=512) |
| **Attention** | CBAM (Channel + Spatial) | Full Self-Attention |

---

## 2. 입력 설계 차이 (핵심)

### 이전 연구: 7채널 입력
[dataset_gg_nmode_v2.py L155-167](file:///home/oem/solar_review/dataset_gg_nmode_v2.py#L155-L167):
```
입력 = [frame_t-4, frame_t-3, frame_t-2, frame_t-1, frame_t, PW, Sobel_PW]
```

- **PW (Persistence Weighting)**: `(img_t - img_{t-(lp+1)})^2` - 시간 변화량의 제곱
- **Sobel_PW**: Sobel edge 차분의 제곱 - 공간적 경계 변화량

이 2채널은 **변화 영역을 명시적으로 마킹**하여, 모델이 "어디가 변할 것인가"를 직접 학습할 수 있게 합니다.

### ST-DiT: context만 입력
```
context = [frame_t-4, ..., frame_t]  (5 frames, 각 1ch)
```
시간적 변화량은 모델이 self-attention을 통해 암묵적으로 학습해야 합니다.

> **검수 판단**: PW와 Sobel_PW는 현재 ST-DiT의 `--use_phys_spatial_concat --phys_concat_features V` (Velocity)와 유사한 역할이지만, 이전 연구에서는 입력 채널로 직접 concat하므로 U-Net의 모든 레이어에서 접근 가능합니다. ST-DiT의 PhysInput은 cross-attention bias 등으로 주입되어 직접성이 다릅니다.

---

## 3. 정규화 및 Loss 차이

### 정규화
| 항목 | 이전 연구 | ST-DiT raw pixel |
|:---|:---|:---|
| **방식** | Min-Max → [-1, 1] | Min-Max → [-1, 1] |
| **Min** | 0.0 (물리적 하한) | 0.0 (고정) |
| **Max** | 26.347 (train set 100th percentile) | **26.347150802612305** (동일) |
| **수식** | `2*(clamp(x,0,max)-0)/(max-0) - 1` | `2*x/DATA_MAX - 1` |
| **코드** | [main_nmode_v2.py L100-104](file:///home/oem/solar_review/main_nmode_v2.py#L100-L104) | [dataset_vae.py L22-27](file:///home/oem/ST-DiT2/ST-DiT/stdit/data/dataset_vae.py#L22-L27) |

> **검수 결과**: 두 코드 모두 동일한 `DATA_MAX = 26.347150802612305`를 사용하며, 동일한 MinMax [-1, 1] 정규화를 적용합니다. 이전 연구는 clamp를 추가로 적용하지만, 실제 데이터 범위(0~26.35)에서 동작이 동일합니다. **따라서 MSE, MAE, PSNR, SSIM 모두 직접 비교 가능합니다.**

### Loss 함수
| 항목 | 이전 연구 | ST-DiT |
|:---|:---|:---|
| **Loss** | MS-SSIM + L1 (dynamic alpha) | MSE (x0 prediction) |
| **alpha** | 0.025 (cosine schedule) | - |
| **compensation** | 100x | - |
| **수식** | `100 * (0.025*MS_SSIM_loss + 0.975*Gaussian_L1/DR)` | `MSE(pred_x0, x_0)` |

> **검수 판단**: MS-SSIM+L1은 구조적 유사도를 직접 최적화합니다. SSIM을 loss에 포함하면 SSIM 지표에서 유리합니다. ST-DiT는 순수 MSE만 사용하므로 SSIM 지표에서 구조적으로 불리합니다.

---

## 4. 학습 설정 차이

| 항목 | 이전 연구 | ST-DiT |
|:---|:---|:---|
| **Optimizer** | SAM (AdamW base, rho=0.05) | AdamW |
| **LR** | 4e-6 + CosineAnnealing(T=20) | 1e-4 |
| **Batch Size** | 16 | 4 |
| **Epochs** | 40 | 100 |
| **AMP** | FP16 | BF16 |
| **Grad Clip** | 20.0 | 1.0 |
| **Conditioning** | GLCM texture + GBF seasonal | PhysInput(V, SG) + TimestepGate |
| **데이터** | 2023 week_split | 2021 week_split |

---

## 5. Metrics 계산 비교

두 코드의 `calculate_metrics` 함수를 대조하면:

| 단계 | 이전 연구 | ST-DiT |
|:---|:---|:---|
| **MSE** | `F.mse_loss` ([-1,1] 공간) | `F.mse_loss` ([-1,1] 공간) |
| **MAE** | `F.l1_loss` ([-1,1] 공간) | `F.l1_loss` ([-1,1] 공간) |
| **PSNR** | `(x+1)/2*255` 후 계산 | `(x+1)/2*255` 후 계산 |
| **SSIM** | `skimage.ssim(uint8, data_range=255)` | `skimage.ssim(uint8, data_range=255)` |

> **검수 결과**: 두 코드 모두 동일한 [-1,1] 정규화 공간에서 MSE/MAE를 계산하고, `(x+1)/2*255`로 변환 후 PSNR/SSIM을 계산합니다. 정규화 상수도 동일(DATA_MAX=26.347)하므로, **MSE, MAE, PSNR, SSIM 네 지표 모두 직접 비교 가능**합니다.

---

## 6. 이전 연구의 성능 우위 요인 종합

### 6-1. Deterministic vs Generative 패러다임
U-Net은 단일 forward pass로 직접 예측합니다. 오차 누적이 없습니다. ST-DiT는 DDIM 10-step chain에서 각 step의 오차가 누적됩니다. 이것이 가장 근본적인 차이입니다.

### 6-2. 공간적 귀납 편향
U-Net의 Conv2d는 local spatial structure를 본질적으로 보존합니다. 4단계 skip connection으로 fine detail이 직접 전달됩니다. ST-DiT는 patch embedding으로 공간 정보를 토큰화하고, self-attention으로만 공간 관계를 학습합니다. patch_size=16이면 512x512를 32x32=1024 토큰으로 분할하므로, 인접 패치 간 경계 연속성을 보장하는 메커니즘이 없습니다.

### 6-3. 명시적 변화 피처 (PW + Sobel_PW)
입력에 시간적/공간적 변화량을 직접 포함하여, 모델이 "어디가 변할 것인가"를 학습 초기부터 알 수 있습니다. ST-DiT의 PhysInput(Velocity, Spatial Gradient)는 유사한 역할을 하며, 현재 spatial concat과 attention bias로 주입됩니다.

**직접 채널 concat 방안**: 현재 `compute_phys_concat_maps`([phys_concat.py L13-45](file:///home/oem/ST-DiT2/ST-DiT/stdit/models/phys_concat.py#L13-L45))는 V(Velocity) feature를 계산한 뒤 마지막 context frame에 channel concat합니다. 이를 이전 연구와 동등하게 만들려면:

1. **PW(Persistence Weighting) 직접 구현**: `(last_frame - ref_frame)^2`를 계산하여 입력에 채널 추가. 현재 V(Velocity)는 `last_frame - ref_frame`이므로, PW는 `V^2`에 해당
2. **Sobel PW 추가**: `sobel_edges(last_frame) - sobel_edges(ref_frame)` 차분의 제곱. 이전 연구의 [dataset_gg_nmode_v2.py L159-163](file:///home/oem/solar_review/dataset_gg_nmode_v2.py#L159-L163)과 동일한 로직
3. **주입 위치**: patchify 전에 IN_CHANNELS를 1 → 3 (또는 +2)으로 확장하여, PW와 Sobel_PW를 입력 이미지와 같은 채널 축에 직접 concat. 이렇게 하면 모든 Transformer layer에서 이 피처에 접근 가능

### 6-4. MS-SSIM+L1 Loss
SSIM을 직접 최적화하므로 SSIM 지표에서 구조적 이점이 있습니다. ST-DiT의 MSE loss는 픽셀 단위 오차만 최소화합니다.

### 6-5. SAM Optimizer
Sharpness-Aware Minimization은 loss landscape의 flat minima를 찾아 generalization을 개선합니다. ST-DiT는 일반 AdamW만 사용합니다.

---

## 7. 동일 조건 비교를 위한 확인 필요 사항

> [!IMPORTANT]
> 논문에 보고된 test set 수치 (MSE, MAE, PSNR, SSIM)를 알려주시면, ST-DiT Stage8 결과와의 정량적 비교를 추가하겠습니다. 네 지표 모두 직접 비교 가능합니다.

추가로 확인이 필요한 사항:
1. 논문에서 보고된 split/연도가 현재 ST-DiT 실험과 동일한지 (2023 vs 2021)
2. LP 값 동일 여부 (이전 연구 config: LP=0, ST-DiT: LP=0 — 동일)
