# Multi-Lead Decoder 개선안 설계 문서

## 1. 현재 구조와 한계

현재 Joint Temporal 파이프라인:

```
Stem (독립)  →  Transformer (cross-frame attention)  →  Decoder (독립)
[B*T,C,H,W]     [B, T*N, D] — 시점 간 mix              [B*T,N,D] → [B*T,C,H,W]
```

Transformer 내부에서 시점 간 상호작용이 발생하지만, decoder 출구에서 per-frame 독립 처리. 두 가지 잠재적 개선 방향이 존재한다.

---

## 2. 방향 A: Cross-Frame Decoder (시점 간 일관성)

### A1. Temporal Conv Post-Processing

Pangu-Weather (Bi et al., 2023)의 multi-step 예측 구조에서 영감. 각 frame을 독립 decode한 후, pixel space에서 temporal 1D conv를 적용.

```
per-frame decode → [B, T, 1, H, W] → Conv1d(kernel=3, along T) → [B, T, 1, H, W]
```

| 항목 | 내용 |
|:---|:---|
| 파라미터 | conv1d: `1×3` kernel, ~3 params per pixel (극소) |
| 장점 | 구현 단순, pixel-level temporal smoothing 직접 보장, 기존 decoder 수정 불필요 |
| 단점 | local smoothing만 가능 (kernel 크기 제한), 장거리 temporal dependency 불가 |
| 문헌 | MCVD (Voleti et al., 2022) — video prediction에서 temporal post-conv 사용 |

### A2. Latent-Space Temporal Attention (before decode)

Video Diffusion Models (Ho et al., 2022)의 temporal attention 구조 차용. Decoder 진입 전, latent token 수준에서 temporal self-attention 수행.

```
[B, T*N, D] → split → T개의 [B, N, D]
  → 같은 spatial position끼리 temporal attention
  → [B, T*N, D] → per-frame decode
```

구체적으로 spatial position `i`에 대해 T개 frame의 token `{x_30[i], x_60[i], x_90[i], x_120[i]}`끼리 self-attention 수행.

| 항목 | 내용 |
|:---|:---|
| 파라미터 | QKV projection: `3 × D × D` + LayerNorm, 약 0.8M (D=512 기준) |
| 장점 | latent space에서 temporal coherence 학습, 장거리 dependency 가능, Video DiT 문헌에서 검증된 패턴 |
| 단점 | 연산 비용 증가 (N개 position × T² attention), 학습 불안정 가능성 |
| 문헌 | Video Diffusion Models (Ho et al., 2022), Latte (Ma et al., 2024) — temporal self-attention layer |

### A3. 3D Conv Decoder (Joint Decode)

NowcastNet (Zhang et al., 2023)의 multi-lead precipitation decoder에서 착안. Decoder 자체를 3D conv로 교체하여 시공간 동시 처리.

```
[B, T*N, D] → reshape [B, D, T, H_g, W_g] → Conv3d stack → [B, 1, T, H, W]
```

| 항목 | 내용 |
|:---|:---|
| 파라미터 | Conv3d layers: kernel `(3,3,3)` × hidden_channels, 약 2-5M |
| 장점 | 시공간을 동시에 처리하여 가장 강력한 temporal coherence |
| 단점 | 기존 HierarchicalDeepDecoder와 호환 불가 (전면 교체 필요), 학습 비용 대폭 증가, T=4라는 작은 temporal dimension에 3D conv가 과도할 수 있음 |
| 문헌 | NowcastNet (Zhang et al., 2023), DGMR (Ravuri et al., 2021) |

### A 방향 비교 및 추천

| 기준 | A1 (Post-Conv) | A2 (Temporal Attn) | A3 (3D Conv) |
|:---|:---:|:---:|:---:|
| 구현 난이도 | 낮음 | 중간 | 높음 |
| 기존 decoder 호환 | O | O | X |
| Temporal coherence 강도 | 약 (local) | 강 (global) | 강 (global) |
| 추가 파라미터 | ~100 | ~0.8M | ~3M |
| 추가 연산 비용 | 무시 | 중간 | 높음 |

> **추천: A2 (Latent-Space Temporal Attention)**
>
> 근거: Video Diffusion Models (Ho et al., 2022)과 Latte (Ma et al., 2024)에서 검증된 패턴. 기존 per-frame decoder 앞에 1개 layer만 삽입하면 되므로 구현 부담이 적고, T=4 정도의 짧은 시퀀스에서 self-attention은 연산 부담도 적다. 3D Conv는 T=4에 비해 과도한 구조.
> 
> A1은 baseline 비교군으로 함께 실험하면 A2의 이점을 증명하는 데 유용하다.

---

## 3. 방향 B: Per-Lead Specialized Decoder

### B1. 독립 Decoder Head (Per-Lead Separate)

GenCast (Price et al., 2023)의 variable-specific output head 패턴. Lead time별 별도의 decoder weight.

```
30분 tokens → Decoder_30(·) → pred_30
60분 tokens → Decoder_60(·) → pred_60
...
```

| 항목 | 내용 |
|:---|:---|
| 파라미터 | 기존 decoder × T배 (현재 ~2M × 4 = ~8M 추가) |
| 장점 | 각 lead time의 분포 차이(30분: sharp, 120분: blurry)를 별도 weight로 완전 특화 가능 |
| 단점 | 파라미터 4배 증가, per-lead 학습 데이터가 동일하므로 overfitting 위험, weight 간 지식 공유 없음 |
| 문헌 | ClimaX (Nguyen et al., 2023) — variable-specific heads |

### B2. 공유 Decoder + Lead-Time FiLM Conditioning

FiLM (Perez et al., 2018) 기법. 단일 decoder weight를 공유하되, lead-time embedding으로 scale/shift 조절.

```
lead_emb = Embedding(4, D)  # [30분→0, 60분→1, 90분→2, 120분→3]
γ, β = MLP(lead_emb)        # FiLM parameters
decode(x) → x * γ + β       # 각 layer에서 lead-time 조건부 변조
```

| 항목 | 내용 |
|:---|:---|
| 파라미터 | Embedding(4, D) + 소형 MLP per layer, 약 0.1M 추가 |
| 장점 | 파라미터 효율적, decoder weight 공유로 일반화 유지, lead-time별 미세 조정 |
| 단점 | FiLM의 표현력이 충분한지 불확실 (scale/shift만으로 120분의 blurriness를 보정할 수 있는가) |
| 문헌 | FiLM (Perez et al., 2018), Adaptive Instance Norm (Huang et al., 2017) |

### B3. LoRA-Style Per-Lead Adaptation

LoRA (Hu et al., 2022) 패턴. 공유 decoder base에 per-lead low-rank delta를 추가.

```
Decoder_shared(x) + LoRA_lead_t(x)
  where LoRA_lead_t(x) = x @ A_t @ B_t  (rank r=4~8)
```

| 항목 | 내용 |
|:---|:---|
| 파라미터 | per-lead: `D × r + r × D_out` × T × num_layers, r=4 기준 약 0.2M |
| 장점 | 파라미터 효율과 특화 능력의 균형, 학습 시 공유 weight는 freeze 가능 |
| 단점 | LoRA rank 선정이 추가 하이퍼파라미터, 구현 복잡도 중간 |
| 문헌 | LoRA (Hu et al., 2022), 기상 분야에서는 아직 미검증 |

### B4. Decoder MoE (Router-Based)

현재 HierarchicalDeepDecoder의 temporal MoE와 유사하지만, routing 기준을 lead-time index로 고정하는 변형.

```
gate = one_hot(lead_time_idx)  # deterministic routing
output = Σ gate[k] * Expert_k(x)
```

| 항목 | 내용 |
|:---|:---|
| 파라미터 | T개 expert, 기존 decoder와 동일 구조, ~2M × T 또는 light expert |
| 장점 | 기존 MoE 코드 재사용 가능, hard routing으로 명확한 특화 |
| 단점 | B1과 본질적으로 동일 (hard routing = separate head), MoE의 soft routing 이점 상실 |

### B 방향 비교 및 추천

| 기준 | B1 (Separate) | B2 (FiLM) | B3 (LoRA) | B4 (MoE) |
|:---|:---:|:---:|:---:|:---:|
| 구현 난이도 | 낮음 | 낮음 | 중간 | 중간 |
| 파라미터 추가 | ~8M | ~0.1M | ~0.2M | ~8M |
| 특화 능력 | 최고 | 중간 | 중-고 | 최고 |
| 일반화 | 낮음 | 높음 | 높음 | 낮음 |

> **추천: B2 (FiLM Conditioning)**
>
> 근거: 현재 데이터에서 4개 lead time의 target 분포는 "시간이 멀어질수록 persistence baseline 대비 residual이 커진다"는 단조 경향을 보인다. 이 경향은 scale/shift 변조만으로 충분히 표현 가능하다. B1/B4는 파라미터 4배 증가에 비해 학습 데이터가 동일하므로 overfitting 위험이 크다. B3(LoRA)는 후속 실험으로 B2가 부족할 때 시도.

---

## 4. 통합 실험 설계 (추천)

현재 baseline(공유 독립 decoder)에 대해 3단계로 검증:

| 단계 | 구성 | config flag | 검증 목표 |
|:---|:---|:---|:---|
| Step 0 | 현재 (공유 독립 decode) | baseline | 기준선 확립 |
| Step 1 | A2 (Temporal Attn) 단독 | `--decoder_temporal_attn` | cross-frame coherence 효과 |
| Step 2 | B2 (FiLM) 단독 | `--decoder_lead_film` | per-lead 특화 효과 |
| Step 3 | A2 + B2 결합 | 둘 다 활성화 | 상승효과 검증 |

---

## 5. 관점별 분석

### Architecture 관점
- A2는 Video DiT 계열의 표준 temporal attention 패턴이며, spatial attention과 temporal attention을 분리하는 "factored attention" 구조(Arnab et al., 2021 — ViViT)와 일관됨
- B2는 conditioning mechanism의 표준 (Diffusion 모델의 t_emb도 본질적으로 FiLM)

### 나의 도메인 관점 (기상 예측)
- 기상 예측에서 lead time이 증가할수록 예측 불확실성이 단조 증가하는 것은 물리적 사실 (Lorenz, 1963)
- 이 단조 경향을 FiLM의 scale factor가 학습하면, 먼 시점일수록 residual의 scale을 키우는 방향으로 자동 조절 가능
- Temporal smoothness는 대기 역학의 연속성 (advection equation)과 부합

### Engineering 관점
- A2: 1개 `nn.MultiheadAttention` layer 삽입, 기존 decoder 수정 없음
- B2: `nn.Embedding(4, D)` + `nn.Linear(D, 2*D_decoder)` 추가, decoder의 각 layer에서 `x = x * gamma + beta` 1줄 삽입
- 둘 다 기존 `MULTI_LEAD=False` 경로에 영향 없음 (하위 호환 보장)

### 나의 검토자 관점
- "왜 per-frame 독립 decode인가?"라는 질문에 대한 답변으로 A2+B2 ablation 결과를 Table로 제시 가능
- Step 0→1→2→3 순차 ablation이 논문의 contribution을 체계적으로 증명
- 문헌 근거가 명확하여 novelty 주장보다 "합리적 통합"으로 positioning 가능
