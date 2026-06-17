# ST-DiT 구조 레벨 성능 병목 분석

> 분석 기준: raw pixel 모드 (1ch, 512x512, patch_size=16, depth=4, hidden_size=512)
> 현재 최고: MSE 0.0328 / MAE 0.1142 / SSIM 0.5324 (Stage7_Idea1_TimestepGate)
> 목표: MSE 0.025 / MAE 0.09 / SSIM 0.60

---

## 1. Context-Target 정보 흐름의 단방향 bottleneck

### 현재 구조
```
context 5 frames (B, 5, 1, 512, 512)
  → patchify → 5 * 1024 = 5120 tokens
  → context_seq (B, 5120, 512)

target (noisy) (B, 1, 512, 512)
  → patchify → 1024 tokens + 1 lead_token = 1025
  → x (B, 1025, 512)

Self-Attention: x attends to x only (1025 tokens)
Cross-Attention: x queries, context_seq is K/V (1025 x 5120)
```

### 문제

- **Self-attention이 target 토큰 간에만 적용**. target의 1024 spatial token은 서로의 공간 관계만 학습한다.
- Context 정보는 **cross-attention을 통해서만** 유입되는데, cross-attention은 target의 query로 context를 조회하는 **단방향** 구조.
- Context 5 frames 자체는 서로 간 attention이 없다. 즉, **시간적 상관관계(temporal correlation)가 context tokens 사이에서 직접 모델링되지 않음.**
- checkthis(AC U-Net)는 encoder-decoder 구조로 context를 multi-scale skip connection을 통해 target에 직접 주입. 정보 전달 경로가 근본적으로 다름.

### 영향도: 

Context frames 간의 temporal pattern(가속, 감속, 이동 방향)은 단순 token concatenation으로는 cross-attention이 학습하기 어려움. SSIM 0.53 -> 0.60 gap의 상당 부분이 여기서 발생할 수 있음.

### 가능한 대응

- Context 내 self-attention 추가 (Context Encoder, Phase 17-3에서 시도했으나 -2.1%. 구현 수준의 문제일 수 있음)
- Context + Target 합동 self-attention (concat all tokens, but O(N^2) memory)
- Temporal aggregation module: context 5 frames를 RNN/1D-Conv로 사전 압축 후 cross-attn

---

## 2. Patch 경계 아티팩트 (Linear Unpatchify)

### 현재 구조
```python
# final_layer: LayerNorm → Linear(512, 16*16*1 = 256)
# unpatchify: (B, 1024, 256) → reshape → (B, 1, 512, 512)
```

### 문제

- 각 패치(16x16)가 **독립적으로 생성**됨. 인접 패치 간의 **연속성이 보장되지 않음.**
- Linear projection은 패치 경계에서 불연속(discontinuity)을 만들며, 이는 **SSIM에 직접적 악영향.** SSIM은 11x11 window로 계산하므로 16x16 패치 경계를 반드시 포함.
- SmoothUnpatchifyLayer(ConvTranspose2d 기반 겹침 복원)가 구현되어 있으나 **현재 사용되지 않음** (smooth_patch=False).

### 영향도: 

Patch boundary artifact는 SSIM을 직접 저하시키지만, depth 4 blocks의 self-attention이 어느 정도 패치 간 coherence를 학습하고 있음. PBR(Patch Boundary Refiner) 실험은 별도로 수행되었으나 효과가 제한적이었던 것으로 기록됨.

### 가능한 대응

- `--smooth_patch` 활성화 (OverlappingCNNStem + SmoothUnpatchifyLayer)
- Final layer에 3x3 Conv refinement 추가 (post-processing 수준)

---

## 3. 단일 해상도 attention (Single-Scale Processing)

### 현재 구조

모든 4개 블록이 동일한 해상도(32x32 patches, hidden_size=512)에서 작동:
```
Block 0: 1024 tokens × 512 dim → self-attn → cross-attn → FFN
Block 1: 1024 tokens × 512 dim → self-attn → cross-attn → FFN
Block 2: 1024 tokens × 512 dim → self-attn → cross-attn → FFN
Block 3: 1024 tokens × 512 dim → self-attn → cross-attn → FFN
```

### 문제

- 기상 현상은 **다중 스케일** 특성을 가짐: 전체 구름 이동(global), 구름 경계(medium), 미세 텍스처(local)
- 단일 해상도에서는 **global context 포착과 local detail 보존을 동시에 달성하기 어려움.**
- U-Net 계열(checkthis)은 encoder에서 점진적 downsampling으로 global 정보를 포착하고, decoder에서 skip connection으로 local detail을 복원.
- DiT-XL(원본)은 256x256에서 patch_size=2로 16,384 tokens를 처리하지만, 512x512에서 patch_size=16은 token 수가 같아도 패치 내부 정보가 손실됨.

### 영향도: 

이것은 DiT 아키텍처의 본질적 한계. U-Net의 multi-scale 처리가 SSIM과 같은 구조적 유사성 지표에서 우위를 점하는 핵심 이유.

### 가능한 대응

- Hourglass Transformer (Phase 17-4에서 시도, SSIM -22.4% 실패. 단, 구현이 단순 pooling 기반이었을 수 있음)
- Multi-resolution patch: 초기 블록은 큰 패치(coarse), 후기 블록은 작은 패치(fine)
- Window attention + shifted window (Swin Transformer 방식)

---

## 4. DDIM 10-step 추론의 누적 오차

### 현재 구조
```
x_T (noise) → 10 DDIM steps → x_0 (prediction)
각 step: model(x_t, context, t) → pred_noise → x_{t-1}
```

### 문제

- 10-step DDIM은 1000-step 대비 **100x 압축된 궤적**을 따름. 각 step에서의 noise 추정 오차가 다음 step에 누적.
- v-prediction은 epsilon-prediction 대비 DDIM 궤적에서 **더 안정적인 x0 추정**을 제공하는 것으로 알려져 있음. 현재 x0-prediction은 이 이점을 활용하지 못함.
- Ensemble K=5가 MSE 0.0309를 달성한 것은 **DDIM 궤적의 stochastic 분산을 평균으로 상쇄**한 결과. 이는 단일 궤적의 한계를 직접 보여줌.

### 영향도: 

현재 실험에서 step 수를 늘려도 성능 향상이 미미했던 전례가 있을 수 있음 (40-step 실험 존재). 근본적 병목은 아니지만, 현재 진행 중인 v-prediction 전환이 이 문제를 부분적으로 완화할 수 있음.

---

## 5. Context 시간 차원의 정보 압축

### 현재 구조
```
context 5 frames: t-4, t-3, t-2, t-1, t0 (각 30분 간격, 총 2시간)
→ 개별 patchify → 5120 tokens → 단순 concatenation
```

### 문제

- 5 frames × 1024 patches = **5120 context tokens**. Cross-attention에서 target의 각 query가 5120 keys를 조회해야 함.
- 시간적으로 중요한 정보(최근 frame이 더 중요)에 대한 **명시적 가중치 부여가 없음.** Learnable temporal embedding이 있으나 이는 position encoding일 뿐, attention 가중치를 직접 제어하지 않음.
- 5개 frame의 **변화 패턴(가속도, 방향)**은 인접 frame 간 차분에 있지만, 이 정보가 개별 token embedding에 암묵적으로만 포함됨.

### 영향도: 

Temporal gradient (Phase 3-D)와 Velocity+SG (E6a) 실험에서 시간 차분 피처의 효과가 확인되었으나 미미함. 근본적으로는 attention이 시간 구조를 발견하기 어려운 표현 형태의 문제.

---

## 종합 우선순위

| 순위 | 병목 | 영향도 | 상태 | 실험 난이도 |
|:---:|:---|:---:|:---|:---:|
| 1 | Context-Target 정보 흐름 | ★★★★ | 미해결 (Phase 17-3 실패) | 높음 |
| 2 | 단일 해상도 attention | ★★★★ | 미해결 (Hourglass 실패) | 높음 |
| 3 | Context 시간 압축 | ★★★ | 부분 (V+SG, TimestepGate) | 중간 |
| 4 | Patch 경계 아티팩트 | ★★★ | 미해결 | 낮음 |
| 5 | DDIM 추론 오차 | ★★ | v-prediction 실험 진행 중 | 낮음 |

> **핵심 인사이트**: 1번과 2번은 **DiT vs U-Net 아키텍처의 근본적 차이**에서 비롯됨. U-Net은 multi-scale skip connection으로 context-target 간 직접적 정보 전달 + multi-resolution 처리를 동시에 달성하지만, DiT는 flat token sequence에서 이를 학습해야 함. 이 gap이 SSIM 0.53 vs 0.60의 주요 원인일 가능성이 높음.
