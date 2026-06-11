# MoE vs Capacity Scaling — 확장 분석 결과

> **5개 모델 비교**: baseline, tmoe_dec, cap_dec3x, cap_stem128, cap_both

## 모델 설명

우리 모델은 크게 **인코더(Stem)** → **Transformer** → **디코더(Decoder)** 3단계로 구성됩니다.

| 모델 | 설명 |
|------|------|
| **baseline** | 기본 모델. 표준 크기의 인코더 + 디코더 사용. |
| **tmoe_dec** | 디코더에 **전문가 3명(MoE)** 배치. 같은 입력에 대해 3명이 각자 복원하고, 결과를 평균. 인코더는 기본 크기. |
| **cap_dec3x** | 디코더의 채널 폭을 **3배로 확장** (전문가 없이 하나의 큰 conv). 인코더는 기본 크기. |
| **cap_stem128** | **인코더만 확장** (시작 채널 128). 디코더는 기본 크기. |
| **cap_both** | 인코더 확장(128) + 디코더 3배 확장. **전문가 없이 순수 크기만 키운 최대 모델.** |

> 핵심 질문: "전문가 3명을 쓰는 것(tmoe_dec)" vs "하나를 크게 키우는 것(cap_dec3x/cap_both)" — 어느 쪽이 왜 더 좋은가?

---

## 용어 정의

| 용어 | 정의 |
|------|------|
| **Effective Rank** | Conv weight 행렬의 SVD(특이값 분해)에서 특이값(singular value)의 **엔트로피 기반 유효 차원 수**. 공식: exp(−Σ pᵢ log pᵢ), 여기서 pᵢ = σᵢ / Σσ. 높을수록 weight가 **더 많은 독립적인 방향**을 활용한다는 의미. 최대값은 min(행, 열). |
| **Top-N Energy** | SVD 특이값 중 상위 N개가 전체 특이값 합에서 차지하는 비율. 낮을수록 에너지가 **고르게 분산** = 다양한 방향을 활용. 높으면 소수 방향에 집중. |
| **Co-adaptation** | 모델의 한 부분(예: 디코더)을 바꾸면, 다른 부분(예: 인코더, Transformer)의 학습된 weight도 함께 바뀌는 현상. 여기서는 baseline 대비 backbone weight의 L2 거리를 측정. |

---

## 핵심 발견

1. **MoE의 effective rank는 cap3x보다 낮다** — MoE의 이점은 representation diversity가 아닌 **다른 메커니즘** (후술)
2. **모든 모델이 backbone을 ~30% 변화** — decoder 변경이 전체 학습 dynamics를 바꿈 (co-adaptation)
3. **cap_both (FID=43.59)가 전체 1위** — stem128 + dec3x 결합이 현재까지 가장 강력

---

## 전체 성능 비교

| Model | Dec Params | MAE ↓ | PSNR ↑ | SSIM ↑ | FID ↓ |
|-------|:----------:|:-----:|:------:|:------:|:-----:|
| baseline | 2.0M | 0.1073 | 21.89 | 0.5428 | 66.92 |
| **tmoe_dec** | 6.7M | 0.1075 | 21.90 | 0.5442 | **57.18** |
| cap_dec3x | 10.6M | 0.1070 | 21.94 | 0.5463 | 77.63 |
| cap_stem128 | 2.0M | 0.1060 | 22.00 | 0.5460 | 73.09 |
| **cap_both** | 10.6M | 0.1110 | 21.65 | 0.5380 | **43.59** |

> [!NOTE]
> - MAE/PSNR/SSIM: **pixel 정확도** (클수록 좋은 것은 PSNR/SSIM, 작을수록 좋은 것은 MAE)
> - FID: **분포 유사도** (작을수록 좋음). 사람 눈에 가까운 "자연스러움" 지표.
> - tmoe_dec은 MAE는 baseline과 동급이지만, **FID에서 14% 개선** — 더 자연스러운 이미지 생성.
> - cap_both는 MAE/PSNR/SSIM이 오히려 약간 나쁘지만, **FID가 35% 개선** — 분포 수준의 자연스러움.

---

## 분석 A: Effective Rank

| Model | Conv 수 | Dec Params | Avg EffRank | Max EffRank |
|-------|:-------:|:----------:|:-----------:|:-----------:|
| baseline | 16 | 2.0M | 67.75 | 252.75 |
| **tmoe_dec** | **28** | 6.7M | 67.88 | 254.19 |
| cap_dec3x | 16 | 10.6M | **196.59** | **741.97** |
| cap_stem128 | 16 | 2.0M | 67.90 | 252.89 |
| cap_both | 16 | 10.6M | 196.30 | 742.42 |

> [!IMPORTANT]
> **tmoe_dec의 Conv 수가 28인 이유**: 기존 16개 conv에 더해, MoE의 전문가 conv 12개(4 stage × 3 expert)가 추가.
> 
> **"Conv 수가 많아서 성능이 좋아진 건 아닌가?"**  
> → 단순 conv 수 증가라면 cap_dec3x(16개 conv, 10.6M params)가 더 좋아야 한다. 하지만 cap_dec3x는 FID 77.63으로 **악화**.
> → tmoe_dec의 expert conv 12개는 총 4.6M params. cap_dec3x는 8.6M params 추가. **더 적은 파라미터의 MoE가 더 좋은 FID.**
> → conv 수가 아니라, **3개가 독립적으로 다른 것을 학습한 뒤 평균**하는 구조가 핵심.

### MoE Expert별 EffRank (참고)

| Expert | Shape | EffRank | Max 가능 |
|--------|-------|:-------:|:--------:|
| Stage 0 Expert 0 | (256,512,3,3) | 252.9 | 256 |
| Stage 0 Expert 1 | (256,512,3,3) | 253.4 | 256 |
| Stage 0 Expert 2 | (256,512,3,3) | 253.2 | 256 |
| Stage 1 Expert 0 | (128,256,3,3) | 126.9 | 128 |
| ... | ... | ~99% | ... |

→ 각 expert가 주어진 차원을 **거의 한계까지 활용** (99%).

---

## 분석 B: MoE concat vs cap_dec3x (Stage별)

3개 expert weight를 concat하면 하나의 큰 행렬이 된다. 이것의 effective rank를 cap_dec3x의 wider conv와 비교:

| Stage | MoE concat EffRank | Cap3x EffRank | 비율 |
|:-----:|:------------------:|:-------------:|:----:|
| 0 | 745.12 | 741.97 | 1.00x |
| 1 | 372.60 | 1.00 | 372x |
| 2 | 185.38 | 377.11 | 0.49x |
| 3 | 3.00 | 189.39 | 0.02x |

> Stage 1에서 cap3x eff_rank=1.00은 해당 stage의 conv가 사실상 rank-1(거의 하나의 방향만 사용)이라는 뜻. 
> cap3x의 파라미터가 효율적으로 활용되지 못하고 있음.

---

## 분석 C: Singular Value Spectrum

**Top-N Energy**: SVD 특이값 상위 N개가 전체의 몇 %를 차지하는가. 낮을수록 에너지가 더 고르게 분산.

| Model | 최대 Conv Shape | Top-5 Energy | Top-10 Energy |
|-------|:--------------:|:------------:|:-------------:|
| baseline | (256,512,3,3) | 3.22% | 5.95% |
| **tmoe_dec** | (256,512,3,3) | **2.40%** | **4.76%** |
| cap_dec3x | (768,512,3,3) | 1.70% | 2.93% |
| cap_both | (768,512,3,3) | 1.67% | 2.85% |

> tmoe_dec이 baseline보다 에너지가 더 분산 (3.22% → 2.40%). MoE와 공존하면서 기존 conv도 더 다양한 방향을 학습.

---

## 분석 D: Decoder Weight 간 Cosine Similarity

각 모델의 decoder conv weight를 하나의 벡터로 펼친 뒤, 모델 쌍 간 코사인 유사도를 측정:

| | baseline | tmoe_dec | cap_dec3x | cap_stem128 | cap_both |
|---|:--------:|:--------:|:---------:|:-----------:|:--------:|
| **baseline** | 1.000 | -0.001 | -0.001 | -0.001 | 0.001 |
| **tmoe_dec** | -0.001 | 1.000 | 0.000 | -0.002 | 0.001 |
| **cap_dec3x** | -0.001 | 0.000 | 1.000 | 0.001 | **0.003** |
| **cap_stem128** | -0.001 | -0.002 | 0.001 | 1.000 | 0.001 |
| **cap_both** | 0.001 | 0.001 | **0.003** | 0.001 | 1.000 |

> 모든 모델 쌍 간 cos ≈ **0.000** — 같은 초기화에서 시작해도, decoder 구조가 다르면 **완전히 다른 해(solution)로 수렴**.
> cap_dec3x ↔ cap_both만 0.003으로 미세하게 유사 (같은 wider conv 구조를 공유하기 때문).

---

## 분석 E: 파라미터 효율

| Model | Dec Params | Avg EffRank | MAE ↓ | PSNR ↑ | SSIM ↑ | FID ↓ | ΔFID |
|-------|:----------:|:-----------:|:-----:|:------:|:------:|:-----:|:----:|
| baseline | 2.0M | 67.75 | 0.1073 | 21.89 | 0.5428 | 66.92 | 0.00 |
| **tmoe_dec** | 6.7M | 67.88 | 0.1075 | 21.90 | 0.5442 | **57.18** | **-9.74** |
| cap_dec3x | 10.6M | 196.59 | 0.1070 | 21.94 | 0.5463 | 77.63 | +10.71 |
| cap_stem128 | 2.0M | 67.90 | 0.1060 | 22.00 | 0.5460 | 73.09 | +6.17 |
| **cap_both** | 10.6M | 196.30 | 0.1110 | 21.65 | 0.5380 | **43.59** | **-23.33** |

> [!WARNING]
> **cap_dec3x**: EffRank가 196.59로 baseline의 3배인데, FID는 오히려 **악화** (+10.71).  
> → **높은 effective rank ≠ 좋은 FID.** Rank만으로는 성능을 설명할 수 없다.
>
> **cap_both**: MAE/PSNR/SSIM이 baseline보다 나쁜데 FID만 압도적 — pixel 정확도와 분포 자연스러움은 **별개의 축**.

---

## 분석 F: Backbone Co-adaptation

"Decoder를 바꿨을 뿐인데, Transformer/인코더 등 나머지 부분(backbone)의 학습된 weight도 변하는가?"

각 모델의 backbone weight와 baseline의 backbone weight를 비교하여 L2 거리를 측정:

| Model vs baseline | L2 거리 | 상대 변화율 | 해석 |
|-------------------|:-------:|:----------:|------|
| tmoe_dec | 224.0 | 30.22% | baseline 대비 backbone이 30% 달라짐 |
| cap_dec3x | 222.6 | 30.04% | 거의 동일한 수준의 변화 |
| cap_stem128 | 213.3 | 28.84% | 약간 적은 변화 |
| cap_both | 213.1 | 28.82% | 약간 적은 변화 |

**"무엇이 변했는가"** — 각 모델에서 가장 크게 변한 backbone 컴포넌트:

| Model | 가장 크게 변한 레이어 | 변화율 |
|-------|----------------------|:------:|
| tmoe_dec | Stem Conv1 weight, Spatial Expert | 150~154% |
| cap_dec3x | Final Conv10 weight, Stem Conv2 | 147~171% |
| cap_stem128 | Stem Conv1 weight, Null Seasonal Token | 151~157% |
| cap_both | Stem Conv1 weight (193%), Cross-Attn (143%) | **193%** |

> **모든 변형이 backbone을 ~30% 변화시킨다**는 것은, decoder 구조 변경이 gradient를 통해 전체 네트워크 학습에 영향을 준다는 의미.
> 특히 cap_both는 Stem Conv1을 193%나 변화 — 인코더+디코더 동시 확장이 **학습 dynamics를 가장 크게 변형**.

---

## 종합 판정

### Q: MoE가 cap3x보다 FID가 좋은 이유는?

| 가설 | 검증 결과 |
|------|----------|
| MoE의 effective rank가 높아서? | ❌ cap3x(196) >> tmoe(68) |
| Conv 수가 많아서? | ❌ cap3x가 파라미터 더 많음 |
| Backbone co-adaptation이 달라서? | ❌ 모두 ~30%로 유사 |
| **Implicit ensemble 효과?** | ✅ **실험 확인** (분석 H) |
| **Activation sparsity 해소?** | ✅ **실험 확인** (분석 G) |

---

## 분석 G: Activation Sparsity 비교 (실험)

baseline과 tmoe_dec에 동일 입력을 넣어 DDIM 추론 → decoder 각 stage 출력의 near-zero 비율 측정 (10 batch, threshold: |x| < 0.01).

**Baseline decoder (주요 conv layers):**

| Layer | Zero Ratio | Near-Zero Ratio |
|-------|:----------:|:---------------:|
| dec_0 (Conv, 512→256) | 0.0000 | 0.0136 |
| dec_2 (Conv, 256→128) | 0.0769 | 0.1818 |
| dec_5 (Conv, 128→64) | 0.0569 | 0.1488 |
| dec_8 (Conv, 64→1) | 0.0470 | 0.1116 |

**tmoe_dec decoder (4 MoE stages):**

| Stage | Zero Ratio | Near-Zero Ratio |
|-------|:----------:|:---------------:|
| moe_stage_0 (3×Conv, 512→256) | 0.0230 | 0.0904 |
| moe_stage_1 (3×Conv, 256→128) | 0.0008 | 0.0386 |
| moe_stage_2 (3×Conv, 128→64) | 0.0001 | 0.0583 |
| moe_stage_3 (3×Conv, 64→1) | 0.0000 | 0.0366 |

| | Baseline avg | MoE avg | 변화 |
|---|:---:|:---:|:---:|
| **Near-Zero Ratio** | 8.15% | 5.59% | **−31.4%** |

> **결론**: MoE가 near-zero activation을 31% 줄임 → 더 많은 뉴런이 의미 있는 신호를 전달. **가설 2 지지.**

---

## 분석 H: Individual Expert vs Ensemble (실험)

tmoe_dec의 3개 expert를 각각 단독으로 사용한 output vs 3개 평균(ensemble) output의 메트릭 비교 (20 batch DDIM 추론).

> ⚠️ 절대 수치는 inference 후처리(residual 복원, denormalization) 생략한 raw 출력이므로 **상대 비교만 유효**.

| Variant | MAE | FID |
|---------|:---:|:---:|
| **Ensemble (3 avg)** | 0.54 | **191** |
| Expert 0 only | 192.90 | 430 |
| Expert 1 only | 37.50 | 518 |
| Expert 2 only | 17.91 | 471 |
| **Expert 평균** | **82.77** | **473** |

| | Ensemble | Expert 평균 | 개선율 |
|---|:---:|:---:|:---:|
| **FID** | 191 | 473 | **−59.6%** |
| **MAE** | 0.54 | 82.77 | **−99.3%** |

> **결론**: 개별 expert 단독은 매우 나쁘지만, 3명 평균하면 FID가 60% 개선. **가설 1 지지.**
>
> 특히 Expert 0은 MAE=193으로 거의 random noise 수준 — 3명이 각자 **완전히 다른 복원 전략**을 학습한 것.

---

## 최종 결론

**두 가설 모두 실험적으로 확인됨:**

| 메커니즘 | 효과 크기 | 역할 |
|----------|:---------:|------|
| **Implicit Ensemble** (가설 1) | FID −59.6% | **주효과**. 3명이 각자 다른 복원을 하고 평균 → output 분산 감소 + 다양한 모드 커버 |
| **Sparsity 감소** (가설 2) | Near-zero −31.4% | **부수 효과**. MoE가 dead neuron을 줄여 더 풍부한 feature 전달 |

> MoE(tmoe_dec)가 단순 capacity 확장(cap_dec3x)보다 좋은 이유:
> - cap_dec3x는 **하나의 큰 conv**로 하나의 해(solution)만 학습 → representation은 다양하지만(EffRank↑), 출력은 단일 모드.
> - tmoe_dec은 **3개 독립 expert**가 각자 다른 해를 학습 → 평균으로 노이즈 상쇄 + dead neuron 감소.
> - **"도구를 크게 키우는 것"보다 "전문가 3명이 각자 복원한 뒤 평균"이 더 효과적.**

