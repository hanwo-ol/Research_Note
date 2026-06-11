# ST-DiT × DWT 64채널 구조 적합성 분석

> 분석 대상: 3-Level Haar DWT → 64채널 latent (64×64)의 에너지 분포가 모델에 적절히 반영되는지

## 데이터 현실

```
Channel 0  (LL>LL>LL):  std=7.63   에너지 88.1%  ████████████████████████████████████████
Channel 1-3  (LL>LL>HF): std≈0.67   에너지  5.1%  ███
Channel 4-15 (LL>HF>*):  std≈0.31   에너지  3.4%  ██
Channel 16-63 (HF>*>*):  std≈0.15   에너지  3.4%  ██

→ ch0의 std는 ch48-63 평균의 약 50~62배
```

---

## 체크포인트별 분석

### ✅ CP1. 전처리 정규화 (preprocess_dwt.py)

```python
tensor = 2.0 * (tensor - global_min) / (global_max - global_min) - 1.0
# 원본 픽셀 → [-1, 1] 범위 → DWT 적용
```

**판정**: 정상. Min-Max를 원본 픽셀에 적용한 후 DWT를 수행하므로 wavelet 계수의 자연스러운 에너지 분포가 보존됨.

---

### ✅ CP2. 채널별 Z-Score 스케일링 (train.py, `get_channel_scales()`)

```python
# 학습 시 적용
context = (context - mean_tensor) / (std_tensor + 1e-8)
target  = (target  - mean_tensor) / (std_tensor + 1e-8)
```

- `mean_tensor`: 64채널 각각의 평균 (ch0=-0.38, 나머지 ≈0)
- `std_tensor`: 64채널 각각의 표준편차 (ch0=7.63, ch63=0.12)

**판정**: ✅ **잘 설계됨**. Z-Score 후 모든 채널이 mean≈0, std≈1로 정규화 → 모델 입력은 에너지 균등.

> [!NOTE]
> 이 Z-Score 덕분에 모델이 보는 입력은 "ch0이 50배 크다"가 아니라 "모든 채널이 비슷한 분포"임. **하지만** 이것이 loss와 diffusion noise에도 동일하게 적용되는지가 CP5-CP7의 핵심.

---

### 🟡 CP3. Stem 임베딩 (st_dit.py L418-439)

```python
c_low = self.freq_channels[0]   # = 16
x_target_low  = x_target[:, :c_low]      # ch 0-15  (LL 경로 전부)
x_target_high = x_target[:, c_low:]       # ch 16-63 (HF 경로 전부)
```

| 트랙 | 채널 | Stem | 출력 |
|------|------|------|------|
| Low  | ch 0-15 (16채널) | `LinearStem(16, 512)` | [B, N, 512] 토큰 |
| High | ch 16-63 (48채널) | `Conv2d(48, 512, k=3)` | [B, 512, 64, 64] 공간 |

**관찰**:

> [!WARNING]
> **ch0 (LL³, 에너지 88%)과 ch1-15 (에너지 8.5%)가 같은 Low 트랙에 묶임.**
>
> Z-Score 정규화 덕분에 값 범위는 비슷하지만, **물리적 의미가 극적으로 다른 채널**(전역 구조 vs 중간 에지)이 16채널짜리 하나의 Linear projection으로 혼합됨.
>
> 이로 인해 ch0의 거시적 패턴 정보가 ch1-15의 에지 잔차와 섞여 희석될 수 있음.

**영향**: 중간. Z-Score가 값 범위를 맞추므로 치명적이진 않지만, ch0를 별도로 처리하면 더 좋을 수 있음.

---

### 🟡 CP4. FreqMoE 트랙 내 True MoE FFN (moe_blocks.py)

Low 트랙 내부에서 `DenseMoEFFN`이 FFN을 4개 전문가로 분기:

```python
self.experts = nn.ModuleList([
    FFN(hidden_size, mlp_ratio) for _ in range(4)   # ← 4개 동일 FFN
])
```

**판정**: 🟡

> [!IMPORTANT]
> **4개 전문가가 동일한 `Linear → GELU → Linear` 구조.**
>
> 라우터(ST-Aware Router)가 토큰별로 할당을 분기하지만, 전문가 자체의 연산 전략이 동일하므로:
> - "전역 패턴 토큰"과 "에지 토큰"이 본질적으로 다른 처리를 받을 수 없음
> - 초기화 차이에 의한 암묵적 분화만 가능
> - 이것이 **Phase 3-A (Hetero Expert)** 의 근거

---

### 🔴 CP5. Diffusion Noise Schedule — 64채널 동일 노이즈

```python
noise = torch.randn_like(x_start)              # 모든 채널에 동일 N(0,1) 노이즈
x_noisy = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
```

**핵심 문제**: Z-Score 정규화 후 모든 채널이 std≈1이므로:
- `noise ~ N(0,1)`과 `x_start ~ N(0,1)`의 SNR이 모든 채널에서 동일
- **물리적으로는** ch0(전역 패턴)이 ch48-63(초미세 노이즈)보다 **훨씬 예측하기 쉬워야** 하지만, 정규화 후에는 이 난이도 차이가 소실됨

**판정**: ✅ **실제로는 의도된 설계**.

> Z-Score 정규화의 목적이 정확히 이것: 모든 채널을 동등한 신호 조건으로 표준화하여, 모델이 각 채널을 균등하게 복원하도록 유도. 만약 ch0에만 노이즈가 적으면 모델이 고주파 채널을 무시하게 됨. **현재 설계가 올바름.**

---

### 🔴 CP6. Loss 계산 — 채널 기여도 불균형 (핵심 이슈)

```python
raw_loss = F.mse_loss(target, model_output, reduction='none')   # [B, 64, 64, 64]
loss = raw_loss.mean(dim=[1, 2, 3])                              # 64채널 균등 평균
```

Z-Score 정규화 후 **loss 공간에서의 채널 기여도**:
- ch0: 전체 loss의 **1/64 = 1.6%** (에너지는 88%인데!)
- ch16-63: 전체 loss의 **48/64 = 75%** (에너지는 3.4%인데!)

**판정**: 🔴 **구조적 주의 필요**

> [!CAUTION]
> **Z-Score 정규화 후 균등 평균 MSE → 고주파 채널이 loss를 지배함.**
>
> 이것이 의미하는 바: 모델은 고주파 채널(std=0.08~0.15, 거의 0에 가까운 잔차)의 재구성에 75%의 학습 자원을 투입.
> 하지만 이 채널들은 IWT 복원 시 픽셀 공간에서의 기여가 3.4%에 불과.
>
> **결과**: 픽셀 공간에서의 SSIM/MSE와 latent 공간에서의 loss 사이에 **목적 불일치(objective mismatch)** 발생.

**현재 대응책**:
- `USE_VARIANCE_LOSS_WEIGHTING=True`: `loss_weights = std / std.mean()` (σ 기반 가중치)
  - ch0 가중치 ≈ 7.63/0.96 = **7.95x**, ch63 가중치 ≈ 0.12/0.96 = **0.13x**
  - → 고주파 페널티 감소, 저주파 페널티 증가 (**부분적 해결**)
- `asymmetric_freq` 가중치: ch0-15에 1.5x, ch16-63에 0.5x (**존재하지만 현재 min_snr과 공존 안 함**)

> **Q: 현재 Phase 2-B 실험들(H/I/K)에서 `USE_VARIANCE_LOSS_WEIGHTING`이 켜져 있는가?**
> → 확인 필요. 꺼져 있다면 CP6 문제가 **완전히 미해결** 상태.

---

### ✅ CP7. Pixel-Space Loss (Phase 2-B)

```python
I_pred   = iwt_module(pred_x0_unscaled)   # [B, 1, 512, 512]
I_target = iwt_module(target_unscaled)     # [B, 1, 512, 512]
loss_pixel = compute_pixel_loss(I_pred, I_target, loss_type="l1")
loss = loss + 0.1 * loss_pixel
```

**판정**: ✅ **CP6의 문제를 우회함.**

pixel loss는 IWT 후 픽셀 공간에서 계산되므로:
- ch0의 88% 에너지 기여가 자연스럽게 pixel loss에 반영됨
- 고주파 채널의 과민 학습을 pixel loss가 상쇄

> 이것이 **Phase 2-B pixel loss가 critical한 이유**: latent MSE만으로는 채널 기여도가 맞지 않고, pixel loss가 물리적 중요도를 보정하는 역할을 함.

---

## 종합 판정

| 체크포인트 | 상태 | 심각도 | 현재 대응 |
|-----------|------|--------|----------|
| CP1. 전처리 | ✅ | — | 정상 |
| CP2. Z-Score | ✅ | — | 정상 |
| CP3. Stem 분할 | 🟡 | 낮음 | ch0-15 혼합. LinearStem이 암묵적 분리 가능 |
| CP4. 전문가 동질성 | 🟡 | 중간 | Phase 3-A (Hetero Expert)로 해결 예정 |
| CP5. Noise Schedule | ✅ | — | Z-Score 후 균등이 의도된 설계 |
| **CP6. Loss 불균형** | **🔴** | **높음** | `USE_VARIANCE_LOSS_WEIGHTING` + pixel loss로 부분 대응 |
| CP7. Pixel Loss | ✅ | — | CP6 우회 역할 |

---

## 핵심 발견 3가지

### 1. 🔴 Loss 목적 불일치 (CP6)
Z-Score 후 **64채널 균등 평균 MSE**는 고주파 48채널에 75% 학습 자원을 할당하지만, 픽셀 복원 기여도는 3.4%.
- **즉시 확인**: 현재 실험에서 `USE_VARIANCE_LOSS_WEIGHTING` 상태
- **최적 해법**: Phase 2-B pixel loss (현재 구현 중) + variance weighting 병행

### 2. 🟡 전문가 동질성 (CP4)
4개 전문가가 동일 FFN → 라우터의 구조적 한계. Phase 3-A에서 해결.

### 3. 🟡 Stem에서 ch0과 ch1-15 혼합 (CP3)
에너지 88%인 ch0이 나머지 15채널과 하나의 LinearStem으로 혼합. 차후 고려 가능하나 Z-Score가 완화하므로 우선순위 낮음.
