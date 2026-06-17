# PhysInput × Decoupled K/V Meta-Bias — 직교성 분석

> **분석 대상**:
> - **Stage18 PhysInput** (L40S 서버): V+SG condition injection → [report](file:///data/student1/docker_Solar/ST-DiT2/stage18_bundle/stage18_bundle/reports/report_stage18_PhysInput_success_analysis_v1.md)
> - **Decoupled K/V Meta-Bias** (H200 서버, 본 연구): cross-attn K/V 분리 conditioning → [novelty v3](file:///home/student1/.gemini/antigravity-ide/brain/921a79fc-46b9-442a-9092-725b75f38bef/ctx_meta_novelty_v3.md)

---

## 1. 두 연구의 비교

| 축 | Stage18 PhysInput | Decoupled K/V Meta-Bias |
|---|---|---|
| **무엇을** | 물리 피처(V, SG)를 input channel에 concat | 메타 정보(seasonal, lead, dynamics)를 K/V에 bias 주입 |
| **어디에** | **Input layer** (stem 이전, channel 확장) | **Cross-attention layer** (K/V projection 후) |
| **어떻게** | Raw pixel에 추가 channel concat → 모든 layer가 처리 | K-bias, V-bias로 attention routing/content 분리 조절 |
| **정보 유형** | Pixel-level physics (V=∂I/∂t, SG=∇I) | Global statistics (mean velocity, spatial std 등) |
| **Granularity** | Per-pixel (H×W 해상도) | Per-token 또는 global (1개 스칼라 → 1개 bias) |
| **Backbone** | ConvDiTUNet (D=4, no MoE) | ST-DiT + Spatial Expert MoE (8 experts) |
| **Training data** | 3년 (2021-2023) | 1년 (2021) |

### 핵심 차이

```
PhysInput:     물리 피처를 "보이는 것"에 추가 (input augmentation)
K/V Meta-Bias: 메타 정보를 "보는 방식"에 추가 (attention conditioning)
```

> [!IMPORTANT]
> **직교성 판정: ✅ 완전 직교**
> - PhysInput은 **what the model sees** (입력 데이터 풍부화)
> - K/V Meta-Bias는 **how the model looks** (attention 메커니즘 조절)
> - 서로 다른 computational stage에서 작동하므로 간섭 없음

---

## 2. 통합 가능성 분석

### 통합 방안 A: PhysInput + K+V Token Prepend (현재 구현)

```python
# PhysInput: input channel 확장
x_context = concat([irradiance, V_map, SG_map], dim=channel)  # (B, T, C+2, H, W)
x_context = stem(x_context)  # patch embedding

# K+V Prepend: meta tokens을 context에 prepend
context_seq = cat([meta_tokens, context_patches], dim=1)
# meta_tokens: seasonal, lead, dynamics
```

- **장점**: 둘 다 이미 구현됨, 즉시 결합 가능
- **예상 효과**: PhysInput이 per-pixel physics 제공 + meta token이 global context 제공 → 보완적
- **리스크**: PhysInput 보고서에서 "MoE와 negative interaction" 발견 → 우리 MoE 환경에서도 확인 필요

### 통합 방안 B: PhysInput + Decoupled K/V Bias (다음 단계)

```python
# PhysInput: input에서 pixel-level physics
x_context = concat([irradiance, V_map, SG_map], dim=channel)

# K/V Bias: attention에서 meta conditioning
K = W_k @ context_patches + k_bias(seasonal_emb, lead_emb)   # routing
V = W_v @ context_patches + v_bias(dynamics_emb, persist_emb)  # content
```

- **장점**: 가장 완전한 통합 — input physics + attention meta
- **리스크**: 복잡도 증가, PhysInput + MoE 간섭 가능성

### 통합 방안 C: PhysInput의 V/SG를 K/V Bias의 소스로 활용

```python
# V/SG의 global statistics를 dynamics token 대신 사용
v_global_stats = [V.mean(), V.std(), V.amax()]      # (B, 3)
sg_global_stats = [SG.mean(), SG.std(), SG.amax()]  # (B, 3)

# 이것을 K-bias 또는 V-bias의 소스로 활용
k_bias = W_k_meta @ v_global_stats   # V가 routing을 steering
v_bias = W_v_meta @ sg_global_stats  # SG가 content를 enrich
```

- **장점**: 새로운 피처 계산 없이 PhysInput의 정보를 재활용
- **논문 narrative**: "pixel-level physics(input) + global physics summary(attention) = multi-scale physics injection"

---

## 3. 주의점 / 리스크

### ️ 주의점 1: PhysInput의 seed fragility

| | seed=16 | seed=42 | 평균 |
|---|---|---|---|
| PhysInput Δ MSE | **−2.21%** | +1.62% | −0.27% |

- seed=16에서 큰 개선이지만, seed=42에서 역효과
- 평균 −0.27%는 seed variance(3.4%)의 8% — noise floor 내
- **FID는 두 seed 모두 개선 (−23%, −5.8%, 평균 −14%)** → FID 효과는 진짜

> 우리 실험(H200)에서도 seed 의존성 확인 필요. 현재 64개 실험은 단일 seed.

### ️ 주의점 2: PhysInput × MoE negative interaction

Stage18 보고서의 핵심 발견:
```
PhysInput + MoE (topk=2): MSE +4.00% (악화)
PhysInput + Dense softmix: MSE +6.23% (악화)
PhysInput 단독: MSE −2.21% (개선)
```

우리 모델은 **Spatial Expert MoE (8 experts, rank 16)**를 사용 중.
PhysInput을 그대로 적용하면 Stage18과 동일한 negative interaction 가능성.

> [!WARNING]
> **PhysInput은 "작은 모델에 강한 inductive bias" 전략.**
> 우리 모델(MoE + deep stem + deep decoder)은 이미 capacity가 큰 편.
> PhysInput의 fragile한 prior가 MoE routing과 충돌할 수 있음.

### ️ 주의점 3: Backbone 차이

| | L40S (Stage18) | H200 (우리) |
|---|---|---|
| Backbone | ConvDiTUNet | ST-DiT (vanilla DiT 기반) |
| Stem | Conv UNet | Deep Stem (hierarchical) |
| Decoder | Conv UNet | Deep Decoder |
| MoE | 없음 | Spatial Expert 8개 |
| Loss | CRPS + MinSNR | Charbonnier |
| Data | 3yr (2021-2023) | 1yr (2021) |

**직접 비교 불가** — backbone이 완전히 다르므로 PhysInput 효과가 동일할 보장 없음.

---

## 4. 추천 실험 계획

| 우선순위 | 실험 | 목적 | 예상 시간 |
|---|---|---|---|
| **P0** | 64개 CTX ablation 완료 | K+V prepend의 유효 토큰 확인 | 진행 중 (~30h 남음) |
| **P1** | `--use_phys_input --phys_features V SG` 단독 | PhysInput이 우리 backbone에서도 유효한지 | ~35min |
| **P2** | PhysInput + CTX best (S_L_D) | 입력 물리 + attention meta 시너지 | ~35min |
| **P3** | PhysInput 단독 (MoE OFF) | MoE 간섭 분리 | ~35min |
| **P4** | Decoupled K/V Bias 구현 | Phase 2 핵심 | 코딩 + ~3.5h 실험 |

---

## 5. 논문 Narrative에서의 위치

```
§3 Method
  §3.1 Physics-Informed Input Augmentation (PhysInput)
       → "what the model sees": V, SG channel concat
  §3.2 Decoupled K/V Meta-Bias
       → "how the model looks": K-bias (routing), V-bias (content)
  §3.3 Multi-Scale Physics Integration
       → PhysInput(pixel-level) + Meta-Bias(global summary) = 
          two complementary scales of physical prior injection
```

> [!TIP]
> **Two-scale physics injection**: PhysInput이 per-pixel 물리를 input에서 제공하고, K/V Meta-Bias가 global 물리 요약을 attention에서 제공하는 **multi-scale 구조**로 framing하면, 두 연구를 하나의 coherent contribution으로 통합 가능.
