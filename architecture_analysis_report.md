# ST-DiT2 아키텍처 구성요소 분석 보고서

> **대상 모델**: Solar Irradiance Nowcasting용 ST-DiT2 (Stage4_Resid paradigm)
> **분석 관점**: @reviewer · @domain · @arch · @engineer

---

## 전체 아키텍처 개요

```
Input (B,1,512,512)
    │
    ▼
┌─────────────────────┐
│  ① Deep Stem        │  Conv3×3 stride=2 × 5단 (P=32)
│  [32→64→128→256→512]│  ~1.57M params
└────────┬────────────┘
         │ tokens (B, 256, 512)
         ▼
┌─────────────────────┐
│  ② ST-DiT Bottleneck│  4 × STDiTBlock (SA + CA + MLP)
│  + Spatial Expert    │  + LoRA MoE r16n8 per block
│  + adaLN conditioning│  hidden=512, heads=8
└────────┬────────────┘
         │ tokens (B, 256, 512)
    ┌────┴────┐
    │         │ .detach()
    ▼         ▼
┌────────┐ ┌──────────────┐
│③ Deep  │ │④ Dynamic Head│  2ch velocity/accel 예측
│Decoder │ │  (~0.3M)     │  stage features → SPADE
│(mirror)│ └──────┬───────┘
│~1.57M  │        │ motion features
│        │◄───────┘ DecoderSPADE modulation
│        │         (1+γ)·x + β per stage
└────┬───┘
     │ (B, 1, 512, 512)
     ▼
  Residual Output (x̂₀)
```

---

# ① Deep Stem — `HierarchicalDeepStem`

> 소스: [stem.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/stem.py#L102-L236)

## 1.1 @arch — 설계 원리 분석

| 항목 | 설명 |
|------|------|
| **핵심 동기** | P=32 단일 Conv는 kernel=32×32, hidden=512이면 **134M params** (본체보다 큼). 점진 다운샘플로 **1.57M** (1.2%)에 동일 RF 확보 |
| **구조** | `Conv3×3 stride=2 + GELU` × log₂(P) 단. P=32 → 5단 |
| **채널 스케줄** | `start = max(H // 2^(D-1), 16)` → 매 stage 2배 → cap at H. [32→64→128→256→512] |
| **RF (Receptive Field)** | 2P-1 = **63×63** pixels. CNNStem RF(=P=32)의 약 2배 |
| **정규화** | **없음** (BN/GN 금지). Diffusion noise 분포 보존 정책 |
| **Init** | xavier_uniform + zero bias. 학습 초기 noise scale 보존 |

**설계 영감**: ConvNeXt / Swin V2의 점진 다운샘플 패턴을 DiT 프레임워크에 이식. 단, normalization 전면 제거가 차별점.

## 1.2 @domain — 태양광 도메인 적합성

1. **INS 이미지 특성**: 512×512 해상도, 구름 경계(edge)와 맑은 영역(flat)이 혼재. 5단 점진 압축은 **구름 텍스처의 다중 스케일 정보**를 계층적으로 포착.
2. **Night pixel 처리**: 야간 영역(INS≈0)은 flat → Conv bias로 충분히 표현. Deep Stem의 비선형 계층이 주간 cloud texture에 집중하여 **day/night 분리 자연스러움**.
3. **P=32 정당성**: 512/32 = 16×16 = 256 tokens. Attention O(N²)에서 N=256은 충분히 tractable하면서 각 패치가 32×32 pixel 영역의 풍부한 local 정보를 보유.

## 1.3 @reviewer — 비판적 검토

1. **GELU vs SiLU 불일치**: CNNStem은 SiLU, Deep Stem은 GELU(tanh approx). Bottleneck의 adaLN도 SiLU 사용. 활성화 함수 일관성 부족. 영향은 미미하겠지만 ablation 없음.
2. **Normalization 완전 제거의 근거 불충분**: "noise 분포 보존"이 근거이나, x₀-prediction에서 stem 입력은 noised sample이 아닌 **clean target image**. 학습 안정성을 위해 stem 내부에 GroupNorm이 유익할 수 있으나, 이 가설은 실험으로 검증되지 않음.
3. **Dual-Scale 미사용**: `dual_scale` 옵션이 구현되어 있으나 현재 champion config에서 **비활성** (`USE_DUAL_SCALE_STEM=False`). 활용되지 않는 dead code.

## 1.4 @engineer — 구현 품질

1. **Checkpoint 호환 설계 우수**: `expose_features=False`(default)는 Sequential path, True는 ModuleList path. `state_dict` key 분리로 기존 ckpt 100% 호환.
2. **Power-of-2 검증**: `patch_size & (patch_size - 1) != 0` → 비트 연산 유효성 검사 정확.
3. **Init 루프 범위**: `for m in self.modules()` — 하위 모듈 포함 전체 순회. dual_scale의 `fine_proj`도 xavier로 덮어쓰인 뒤 다시 zero-init → 순서 의존적이지만 **정확히 동작** (zero-init이 뒤에 위치).
4. **개선 제안**: `_StemStage`가 `expose_features` 전용인데 docstring에만 기술. `if not use_stages_path` 분기에서도 동일 클래스 사용으로 통합 가능.

---

# ② ST-DiT Bottleneck — `STDiTBlock × 4`

> 소스: [blocks.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/blocks.py#L34-L301), [spatial_expert.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/spatial_expert.py)

## 2.1 @arch — 설계 원리 분석

| 항목 | 설명 |
|------|------|
| **블록 구조** | `adaLN → Self-Attn → adaLN → Cross-Attn → adaLN → MLP` (3-stage, 9-chunk modulation) |
| **Depth=4** | 극도로 얕음 (ViT-B=12, DiT-XL=28). **Stem+Decoder의 deep hierarchy가 depth 부족을 보상**하는 설계 |
| **adaLN 9-chunk** | shift/scale/gate × 3 stages. Gate로 residual 강도를 timestep-adaptive 조절 |
| **Spatial Expert** | LoRA residual (rank=16, K=8 experts). `delta = Σ_k w_k · up_k · down_k · x`. **MLP 출력에 합산** |
| **Cross-Attention** | Context = 5 past frames × 256 tokens = 1280 KV tokens. Temporal history 주입 |

### Spatial Expert 상세 (LoRA MoE)

```
Input x (B,N,512)
    │
    ├──→ Router: Linear(512→8) → softmax → weights (B,N,8)
    │
    ├──→ Expert k: x @ down_k(512,16) @ up_k(16,512) → e_k
    │    (k = 0..7, loop 방식 — cudagraph 호환)
    │
    └──→ delta = Σ_k w_k · e_k
         mlp_out += delta
```

- **Init**: down=kaiming, up=**zero** → 초기 delta=0 (baseline 동등 시작)
- **Params per block**: 8 × (512×16 + 16×512) + router(512×8) = **135K**
- **전체 (4 blocks)**: ~**540K** (trunk 대비 ~2%)

## 2.2 @domain — 태양광 도메인 적합성

1. **Depth=4의 충분성**: Solar nowcasting은 ImageNet 분류와 달리 **temporal extrapolation** 과제. Semantic hierarchy보다 **motion/trend 포착**이 핵심 → shallow trunk + deep stem/decoder 조합이 합리적.
2. **Cross-Attention 5-frame context**: 10분 간격 × 5 = 50분 history. 구름 이동 속도 고려 시 **적정 temporal window**. 더 긴 history는 persistence-residual 패러다임에서 marginal gain.
3. **8-expert routing**: 태양광 이미지는 (맑음/부분흐림/전면흐림/일출/일몰/야간/구름경계/그림자) 등 다양한 spatial pattern → 8 expert가 이들을 분리 학습 가능.

## 2.3 @reviewer — 비판적 검토

1. **Depth=4 극단적 설정**: 선행연구에서 depth↓ 시 representation capacity 급감이 일반적. **Stem/Decoder가 이를 보상**한다는 가설은 depth ablation(D=2,4,6,8)으로만 검증 가능하나, 보고서에 D=4 외 실험 기록 없음.
2. **Expert 수 K=8 과다 의심**: KI 문서(0081-v2)에 "r16n2로 충분"이라 기술되어 있으나, champion config는 **n=8**. n=2 vs n=8 비교 실험 결과가 명시되지 않음.
3. **Load Balance Loss 비활성**: `lb_weight=0.0` (default). Expert collapse 위험이 있으나 모니터링/방어 없음. `last_router_weights` 진단 슬롯은 있으나 **실제 로깅 여부 불명**.
4. **einsum → loop 전환 근거**: cudagraph 호환을 위해 expert loop으로 변경. K=8이면 loop 8회 → K=2보다 **4× overhead**. K=8 유지라면 batched matmul이 더 효율적일 수 있음.

## 2.4 @engineer — 구현 품질

1. **adaLN chunk 확장 설계 우수**: `use_expert_gate=True` 시 9→10 chunk으로 확장. Gate3와 독립된 expert gate 분리 — **깔끔한 확장**.
2. **cudagraph 호환 고려**: einsum 대신 plain matmul loop. `torch.compile` 친화적 설계.
3. **Forward signature 비대**: `STDiTBlock.forward`에 **13개 인자** (x, context, cond, num_special_tokens, coords×2, x_prev, pos_emb, m_dem, h/w_shape, t_emb, cached_cross_kv). dataclass나 context object 도입 권장.
4. **`inspect.signature` 런타임 호출** (L256): MoE 분기에서 `inspect` 사용 — **매 forward마다 reflection**. 성능 저하 가능. `hasattr` 또는 flag로 대체 권장.

---

# ③ Deep Decoder — `HierarchicalDeepDecoder`

> 소스: [decoder.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/decoder.py#L120-L393)

## 3.1 @arch — 설계 원리 분석

| 항목 | 설명 |
|------|------|
| **핵심 원리** | Deep Stem의 **정확한 mirror**. Bilinear↑ + Conv3×3 × log₂(P) 단 |
| **채널 스케줄** | H → H/2 → H/4 → ... → out_channels. [512→256→128→64→1] |
| **Upsample** | **Bilinear** (ConvTranspose의 checkerboard artifact 회피) |
| **마지막 Conv** | **zero-init** → 학습 초기 출력 = 0 → persistence residual baseline과 동등 시작 |
| **GELU 정책** | 중간 stage에만 적용. **마지막 stage GELU 제거** (출력 unbounded 보장) |

### SPADE Modulation 인터페이스

```
Deep Decoder Stage i 출력 x
    │
    ▼
DecoderSPADE(motion_feat_i, x):
    shared = ReLU(Conv3×3(motion_feat_i))
    γ = Conv3×3(shared)     ← zero-init
    β = Conv3×3(shared)     ← zero-init
    return (1 + γ) · x + β
    │
    ▼ (dyn_t_schedule=alpha_bar 시)
    x_final = g(t) · x_mod + (1-g(t)) · x
    g(t) = √ᾱ(t): t=0 → 1 (full SPADE), t→T → 0 (identity)
```

## 3.2 @domain — 태양광 도메인 적합성

1. **Bilinear upsample 적합**: Solar irradiance 이미지는 smooth gradient가 많음 (맑은 하늘). Bilinear이 ConvTranspose보다 자연스러운 복원.
2. **Zero-init 마지막 Conv**: Persistence-residual 패러다임에서 **초기 예측 = persistence** (변화 없음 가정). 모델이 점진적으로 "변화량"을 학습 → 물리적으로 합리적.
3. **5단 점진 복원**: 32×32 → ... → 512×512. 구름 경계의 **multi-scale sharpness**를 계층적으로 복원.

## 3.3 @reviewer — 비판적 검토

1. **U-Net Skip 폐기 후 정보 병목**: Stem→Bottleneck→Decoder에서 256-token bottleneck이 유일한 정보 통로. Skip이 폐기(0082-0083)되어 **Stem의 풍부한 중간 feature가 전부 소실**. Bottleneck 용량이 충분한지 정량 검증 필요.
2. **Sequential path에서 SPADE 삽입 로직 취약**: L377-389에서 `isinstance(layer, nn.Conv2d)`로 stage 완료 판단 — **Sequential 내부 구조에 의존하는 brittle 로직**. 구조 변경 시 조용히 깨질 위험.
3. **dyn_t_schedule buffer 공유**: skip과 dyn이 같은 `_alpha_bar_for_skip` buffer 사용. 현재는 동일 스케줄이므로 문제없으나, 독립 스케줄 시 분리 필요.

## 3.4 @engineer — 구현 품질

1. **Forward shape 검증 철저**: N != gh*gw, H != hidden_size 시 명시적 ValueError. 디버깅 친화적.
2. **`dynamic_spade_layers` 외부 attach 패턴**: st_dit.py에서 ModuleList를 생성해 decoder에 부착 → **모듈 간 결합도 높음**. Decoder 자체가 SPADE 빌드를 내장하는 게 더 clean.
3. **zero_init_last 안전장치**: `use_skip` path와 Sequential path 모두 마지막 Conv를 정확히 찾아 zero-init. `RuntimeError` fallback도 구현.
4. **t_gate 범용 인터페이스**: `_compute_t_gate` 메서드가 skip/dyn 양쪽 스케줄을 통합 처리 — 확장성 우수.

---

# ④ Dynamic Decoder — `DynamicHead` + `DecoderSPADE`

> 소스: [decoder.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/decoder.py#L449-L554) (DynamicHead), [decoder.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/decoder.py#L396-L446) (DecoderSPADE)

## 4.1 @arch — 설계 원리 분석

| 항목 | 설명 |
|------|------|
| **영감** | NowcastNet (Nature 2023) 경량 변종. Content-Motion 분리 |
| **출력** | 2ch: velocity `I(t)-I(t-1)`, acceleration `v(t)-v(t-1)` |
| **채널 스케줄** | H → H//4 → ... → 2. Content Decoder보다 **4× 빠른 축소** (경량) |
| **Integration** | Stage features → DecoderSPADE로 Content Decoder modulation |
| **Gradient 차단** | 입력에 `.detach()` — Transformer trunk gradient 오염 방지 (0084 교훈) |
| **Loss** | Charbonnier(pred_vel, GT_vel) + Charbonnier(pred_accel, GT_accel), λ=10 |

### 데이터 흐름 상세

```
Transformer tokens (B, 256, 512)
    │
    │ .detach()  ← trunk gradient 차단
    ▼
DynamicHead:
    reshape → (B, 512, 16, 16)
    Stage 0: Bilinear↑ + Conv3×3(512→128) + GELU  → feat_0
    Stage 1: Bilinear↑ + Conv3×3(128→32)  + GELU  → feat_1
    Stage 2: Bilinear↑ + Conv3×3(32→8)    + GELU  → feat_2
    Stage 3: Bilinear↑ + Conv3×3(8→2)     + GELU  → feat_3
    Stage 4: Bilinear↑ + Conv3×3(2→2)              → feat_4 (dynamics)
    │
    ├── stage_features = [feat_0, ..., feat_4]  → Content Decoder SPADE 입력
    └── dynamics (B, 2, 512, 512)              → velocity/accel loss 계산
```

## 4.2 @domain — 태양광 도메인 적합성

1. **Velocity/Acceleration 물리적 의미**: `v(t) = I(t) - I(t-1)` = 10분간 일사량 변화. `a(t) = v(t) - v(t-1)` = 변화의 가속도. **구름 이동/생성/소멸을 직접 모델링**.
2. **Content-Motion 분리 합리성**: 맑은 영역(static)과 구름 이동 영역(dynamic)의 예측 전략이 본질적으로 다름. SPADE modulation이 **motion 영역에서만 Content를 변조**하는 효과.
3. **IP(Input Perturbation) 시너지**: FID 27.79(단독) → **12.27**(IP 결합). **직교 시너지** — IP가 exposure bias 해소, Dynamic이 motion 구조 주입 → 독립적 개선이 곱셈적 효과.

## 4.3 @reviewer — 비판적 검토

1. **`.detach()` 양날의 검**: Trunk gradient 오염 방지는 맞으나, Dynamic Head가 Transformer representation 개선을 **역전파로 유도할 수 없음**. End-to-end 최적화가 아닌 **2-stage 최적화** 효과.
2. **λ=10 고정**: Dynamic loss weight가 매우 높음 (base loss의 10배). Schedule이나 adaptive weighting 없이 고정 → 초기 학습에서 dynamic loss가 **gradient를 지배**할 위험.
3. **InstanceNorm 생략 리스크**: 원본 SPADE는 InstanceNorm + affine modulation. 본 구현은 norm 없이 `(1+γ)·x + β` → **feature magnitude 불안정** 가능성. Norm 제거 근거는 diffusion noise 보존이나, decoder 출력은 이미 denoised representation.
4. **Acceleration 채널 활용도**: GT acceleration은 2차 미분이라 noise 증폭됨. 실제로 acceleration이 content decoder에 유의미한 modulation signal을 제공하는지 **ablation 미확인**.

## 4.4 @engineer — 구현 품질

1. **Zero-init 마지막 Conv**: 학습 초기 dynamics=0 → Content Decoder만 동작. 안전한 warm start.
2. **`stage_features` 속성 저장**: `self.stage_features = stage_features` — forward에서 instance variable에 저장. **메모리 누수 위험** (gradient graph 유지). 다만 `.detach()` 입력이므로 graph 크기 제한적.
3. **해상도 불일치 자동 처리**: `DecoderSPADE.forward`에서 `F.interpolate`로 motion_feat↔x 해상도 자동 맞춤 — 견고한 설계.
4. **dyn_input_noise 학습 시만 적용**: `self.training` 체크 정확. 추론 시 깨끗한 입력 보장.

---

# 종합 평가 매트릭스

| 구성요소 | Params | 설계 완성도 | 도메인 적합성 | 코드 품질 | 핵심 리스크 |
|----------|--------|------------|--------------|----------|------------|
| ① Deep Stem | ~1.57M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Norm 부재, activation 불일치 |
| ② ST-DiT Bottleneck | ~8.5M | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | D=4 극소, K=8 정당성, forward 비대 |
| ③ Deep Decoder | ~1.57M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Skip 부재 병목, SPADE 삽입 취약 |
| ④ Dynamic Decoder | ~0.3M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | .detach() 제약, λ=10 고정, accel 활용 불명 |

## 핵심 권고사항

> [!IMPORTANT]
> 1. **Depth ablation 필요**: D=4가 최적인지 D=6,8과 비교 실험 시급
> 2. **Expert 수 정리**: K=8 vs K=2 실험 결과 문서화. LB loss 활성화 검토
> 3. **Forward signature 리팩터링**: 13개 인자 → context dataclass 도입
> 4. **Sequential SPADE 삽입 로직 개선**: stage 명시 indexing으로 전환

> [!TIP]
> 현재 champion (FID 4.48, MAE 0.1193)의 핵심 비결은 개별 구성요소의 우수성보다 **E2E training + cosine LR + Input Perturbation의 시너지 효과**. 아키텍처 변경보다 **학습 레시피 최적화**가 더 큰 도약을 만들어냈음 (FID 12.27 → 4.78).
