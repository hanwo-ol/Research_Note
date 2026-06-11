# Backbone Bottleneck — 남아있는 개선 여지 분석

## 이미 건드린 것 (현 파이프라인)

| 카테고리 | 구현체 | 역할 |
|---|---|---|
| **Tokenization** | Deep Stem (4-stage hierarchical conv) | P=16 패치 내부 intra-patch reasoning |
| **Reconstruction** | Deep Decoder + Dynamic Decoder (SPADE) | Motion-aware 역패치화 |
| **Block 내부** | Spatial Expert v2 (LoRA MoE + routing) | MLP 병렬 expert residual |
| **Loss** | Charbonnier + DWT HF + Persistence Residual + Freq-Aware + TGS | 다면적 supervision |
| **Conditioning** | t-gate (stem/decoder), seasonal/t routing (expert) | timestep/계절 적응 |

---

## 아직 건드리지 않은 병목 5개

### 1. Depth Scaling (depth=4 → 8+)

**현 상태**: depth=4는 ViT 기준 매우 얕음 (ViT-S/B: 12, DiT-XL: 28). 4블록이면 self-attn 4회 + cross-attn 4회로, 복잡한 공간 관계를 학습할 representational capacity가 제한적.

**기대 효과**:
- 더 깊은 feature hierarchy → high-level spatial pattern 포착 개선
- DiT 논문에서 depth 증가가 FID에 가장 큰 영향 (Figure 9)

**비용**: depth 2배 → 학습 시간 ~1.8-2배, VRAM ~1.5배. H200에서 bs=16 유지 가능 여부 확인 필요.

**위험**: depth=4에서 이미 overfitting 징후가 없는지 먼저 확인 필요. 504 test sample(y21q)에서의 FID 불안정이 시사하듯 **데이터가 모델 크기에 비해 작을 수** 있음. depth 증가 시 regularization(dropout, weight decay) 동반 필요.

> **권장**: depth=8을 y21f에서 시도. 기존 pareto_base 명령에 `--depth 8`만 추가. VRAM 초과 시 `--batch_size 8 --grad_accum_steps 2`로 effective batch 유지.

---

### 2. Attention Quality (Self-Attention 자체 개선)

**현 상태**: 표준 full self-attention + absolute positional embedding. P=16, 512×512 → 32×32 grid → 1024 tokens. attention complexity O(N²) = O(1M)으로 가벼움. **attention 자체가 병목은 아니지만**, 주의를 기울이는 방식에 개선 여지.

**아이디어**:

| 방식 | 설명 | 기대 효과 |
|---|---|---|
| **RoPE** (이미 구현됨) | `--pos_embed_type ast_rope` | 상대 위치 인코딩. 공간 구조 더 잘 포착. **이미 코드 있으니 flag만 바꾸면 됨** |
| **SwiGLU MLP** (이미 구현됨) | `--use_swiglu_mlp` | SD3/FLUX 표준. 동일 파라미터에서 표현력 ↑. **이미 코드 있으니 flag만 바꾸면 됨** |
| **Gate Zero-Init** (이미 구현됨) | `--use_gate_zero_init` | DiT 원 논문 권장. 학습 초기 identity mapping. |

> **권장**: `--pos_embed_type ast_rope --use_swiglu_mlp --use_gate_zero_init`를 한 번에 켜서 y21f ablation. 이 세 가지는 **이미 구현되어 있지만 현재 실험에서 사용하지 않고 있음.** 코딩 비용 0.

---

### 3. Context Utilization (과거 프레임 활용 방식)

**현 상태**: 5개 context frame → patchify → concat → cross-attention의 KV로 사용. 모든 frame이 동등하게 취급됨.

**문제**: 시간적으로 가까운 frame(t-1)이 먼 frame(t-5)보다 더 정보적이지만, 현재 구조는 이를 구분하지 않음.

**아이디어**:

| 방식 | 설명 | 구현 비용 |
|---|---|---|
| **Temporal Attention Bias** | cross-attn에서 최근 frame에 attention score bonus | 낮음 (bias 텐서 추가) |
| **Shared Context KV** (이미 구현됨) | `--use_shared_context_kv` | 블록 간 KV 공유로 context encoding 일관성 |
| **ContextEncoder** (이미 구현됨) | `--use_ctx_encoder --ctx_encoder_depth 2` | context frame 전용 self-attn → 더 좋은 KV |

> **권장**: `--use_ctx_encoder --ctx_encoder_depth 2`를 켜보는 것만으로도 cross-attn 품질 향상 가능. 이미 코드 존재.

---

### 4. Training Recipe 최적화

**현 상태**: AdamW, cosine LR, weight_decay=0.0, batch_size=16, 100 epochs.

**아직 시도 안 한 것**:

| 기법 | 현 상태 | 기대 효과 | 비용 |
|---|---|---|---|
| **Weight Decay** | 0.0 | 0.01~0.05 시 일반화 개선 (ViT 표준) | 0 |
| **EMA** | off | 학습 안정성, test metric 개선 (일반적으로 +0.5~1.0 FID) | ~10% 메모리 |
| **Longer Training** | 100 epochs | 200~300 epochs에서 추가 개선 가능 (특히 y3f 같은 큰 데이터) | 시간 |
| **Gradient Accumulation** | 1 | bs=16에서 이미 충분하지만, depth 증가 시 bs 줄이고 accum으로 보완 | 0 |

> **권장**: `--weight_decay 0.01 --use_ema --ema_decay 0.9999`를 기본으로 추가. 특히 EMA는 diffusion model에서 거의 표준.

---

### 5. Inference Quality (추론 측)

**현 상태**: DDIM 10 steps, guidance_scale=1.0 (CFG off), delta_alpha=1.0.

| 기법 | 현 상태 | 기대 효과 |
|---|---|---|
| **DDIM eta > 0** | eta=0 (deterministic) | stochastic sampling으로 HF texture 회복. `--ddim_eta 0.3~0.5` |
| **더 많은 steps** | 10 steps | 20~50 steps에서 quality 개선 (FID 감소). 비용 선형 증가 |
| **DPM-Solver++** | DDIM | `--sampler dpmpp` — 같은 step수에서 더 높은 품질 |
| **Multi-seed ensemble** | 1 seed | `--ensemble_seeds 5` — 5번 샘플 후 latent 평균. FID 개선 |

> **권장**: 먼저 `--sampler dpmpp --inference_timesteps 20`을 학습 없이 기존 ckpt에서 test. 학습 비용 0.

---

## 종합: 비용 대비 효과 우선순위

| 순위 | 개선 방향 | 추가 학습 필요? | 구현 비용 | 기대 FID 개선 |
|---|---|---|---|---|
| **1** | 추론 개선 (dpmpp + steps 20) | ❌ | 0 | ~5-15% |
| **2** | 기존 구현 flag 활성화 (RoPE, SwiGLU, gate_zero_init) | ✅ (fresh train) | 0 | ~5-10% |
| **3** | Training recipe (weight_decay + EMA) | ✅ (fresh train) | 0 | ~3-8% |
| **4** | Depth scaling (4 → 8) | ✅ (fresh train) | 낮음 | ~10-20% |
| **5** | Context Encoder | ✅ (fresh train) | 0 | ~3-5% |

> [!IMPORTANT]
> **순위 1은 학습 없이 바로 테스트 가능합니다.** 기존 best ckpt로 `--sampler dpmpp --inference_timesteps 20`만 바꿔서 test_only.py 실행하면 됩니다. 이것만으로도 유의미한 개선이 나올 수 있습니다.

> [!NOTE]
> **순위 2~3은 코드 수정 없이 CLI flag만 추가하면 됩니다.** 이미 구현되어 있지만 현재 실험 config에서 비활성 상태인 기능들입니다. 단, 새로운 학습이 필요하고 기존 ckpt와 비호환입니다.
