# ST-DiT2 아키텍처 구성요소 분석 보고서

> **대상 모델**: Solar Irradiance Nowcasting용 ST-DiT2 (Stage4_Resid paradigm)
> **분석 관점**: 아키텍처 설계, 도메인 적합성, 비판적 검토, 그리고 구현 품질에 대한 종합적 분석

---

## 전체 아키텍처 개요

```
Input (B,1,512,512)
    │
    ▼
┌─────────────────────┐
│  1. Deep Stem       │  Conv3×3 stride=2 × 5단 (P=32)
│  [32→64→128→256→512]│  ~1.57M params
└────────┬────────────┘
         │ tokens (B, 256, 512)
         ▼
┌─────────────────────┐
│  2. ST-DiT Bottleneck│  4 × STDiTBlock (SA + CA + MLP)
│  + Spatial Expert    │  + LoRA MoE r16n8 per block
│  + adaLN conditioning│  hidden=512, heads=8
└────────┬────────────┘
         │ tokens (B, 256, 512)
    ┌────┴────┐
    │         │ .detach()
    ▼         ▼
┌────────┐ ┌──────────────┐
│3. Deep │ │4. Dynamic Head│  2ch velocity/accel 예측
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

# 1. Deep Stem — `HierarchicalDeepStem`

> 소스: [stem.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/stem.py#L102-L236)

## 1.1 아키텍처 설계 분석

| 항목 | 설명 |
|------|------|
| **핵심 동기** | P=32 단일 Conv는 kernel=32×32, hidden=512이면 **134M params** (본체보다 큼). 점진 다운샘플로 **1.57M** (1.2%)에 동일 RF 확보 |
| **구조** | `Conv3×3 stride=2 + GELU` × log₂(P) 단. P=32 → 5단 |
| **채널 스케줄** | `start = max(H // 2^(D-1), 16)` → 매 stage 2배 → cap at H. [32→64→128→256→512] |
| **RF (Receptive Field)** | 2P-1 = **63×63** pixels. CNNStem RF(=P=32)의 약 2배 |
| **정규화** | **없음** (BN/GN 금지). Diffusion noise 분포 보존 정책 |
| **Init** | xavier_uniform + zero bias. 학습 초기 noise scale 보존 |

**설계 분석**: ConvNeXt / Swin V2의 점진 다운샘플 패턴을 DiT 프레임워크에 이식했습니다. 다만, 정규화(normalization)를 전면 제거한 점이 차별점입니다.

## 1.2 태양광 도메인 적합성 검토

1. **INS 이미지 특성**: 512×512 해상도에서 구름 경계(edge)와 맑은 영역(flat)이 혼재합니다. 이 5단 점진 압축은 **구름 텍스처의 다중 스케일 정보**를 계층적으로 포착하는 데 적합하다고 판단했습니다.
2. **Night pixel 처리**: 야간 영역(INS≈0)은 flat하므로 Conv bias로 충분히 표현 가능합니다. Deep Stem의 비선형 계층이 주간 cloud texture에 집중하도록 유도하여 **day/night 분리가 자연스럽게** 이루어지도록 설계했습니다.
3. **P=32 정당성**: 512/32 = 16×16 = 256 tokens입니다. Attention O(N²) 연산량 관점에서 N=256은 충분히 다루기 쉬우며(tractable), 각 패치가 32×32 pixel 영역의 풍부한 local 정보를 보유할 수 있는 구조입니다.

## 1.3 비판적 검토 및 분석

1. **GELU vs SiLU 불일치**: CNNStem은 SiLU를 사용하고 Deep Stem은 GELU(tanh approx)를 사용하는 반면, Bottleneck의 adaLN은 다시 SiLU를 사용하고 있어 활성화 함수 간의 일관성이 부족하다고 봅니다. 비록 영향은 미미하겠지만 관련 ablation 실험이 부재합니다.
2. **Normalization 완전 제거의 근거 불충분**: "noise 분포 보존"을 근거로 제시했으나, x₀-prediction에서 stem 입력은 noised sample이 아닌 **clean target image**입니다. 학습 안정성을 위해 stem 내부에 GroupNorm을 적용해 볼 것을 제안하지만, 현재로서는 이 가설이 실험으로 검증되지 않았습니다.
3. **Dual-Scale 미사용**: `dual_scale` 옵션이 구현되어 있으나 현재 champion config에서 **비활성화** (`USE_DUAL_SCALE_STEM=False`)되어 있습니다. 실질적으로 활용되지 않는 dead code로 판단됩니다.

## 1.4 구현 품질 검수

1. **Checkpoint 호환 설계 우수**: `expose_features=False`(기본값)는 Sequential 경로를, True는 ModuleList 경로를 사용하도록 분리하여 기존 체크포인트와의 호환성을 100% 유지하도록 깔끔하게 설계했습니다.
2. **Power-of-2 검증**: `patch_size & (patch_size - 1) != 0` 비트 연산을 통한 유효성 검사가 정확하게 구현되어 있습니다.
3. **Init 루프 범위**: `for m in self.modules()`를 통해 하위 모듈을 포함한 전체를 순회하고 있습니다. dual_scale의 `fine_proj`가 xavier로 덮어쓰인 뒤 다시 zero-init되는 구조인데, 순서 의존적이지만 **정확하게 동작**하고 있음을 확인했습니다.
4. **개선 제안**: `_StemStage`가 `expose_features` 전용으로 설계되었으나 docstring에만 기술되어 있습니다. `if not use_stages_path` 분기에서도 동일한 클래스를 사용하도록 통합하면 코드가 더 깔끔해질 것 같습니다.

---

# 2. ST-DiT Bottleneck — `STDiTBlock × 4`

> 소스: [blocks.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/blocks.py#L34-L301), [spatial_expert.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/spatial_expert.py)

## 2.1 아키텍처 설계 분석

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

- **Init**: down=kaiming, up=**zero**로 초기화하여 초기 delta=0이 되도록 설정했으며, baseline과 동등한 상태에서 학습을 시작합니다.
- **Params per block**: 8 × (512×16 + 16×512) + router(512×8) = **135K**
- **전체 (4 blocks)**: ~**540K** (trunk 대비 ~2%)

## 2.2 태양광 도메인 적합성 검토

1. **Depth=4의 충분성**: Solar nowcasting은 일반적인 ImageNet 분류와 달리 **temporal extrapolation**이 과제입니다. Semantic hierarchy보다 **motion/trend 포착**이 핵심이므로, shallow trunk와 deep stem/decoder의 조합이 합리적이라고 판단했습니다.
2. **Cross-Attention 5-frame context**: 10분 간격 × 5프레임 = 50분의 history를 활용합니다. 구름의 이동 속도를 고려했을 때 **적정 수준의 temporal window**이며, 더 긴 history는 persistence-residual 패러다임 하에서 한계 효용(marginal gain)이 낮을 것으로 분석됩니다.
3. **8-expert routing**: 태양광 이미지는 (맑음/부분흐림/전면흐림/일출/일몰/야간/구름경계/그림자) 등 다양한 spatial pattern을 보입니다. 8개의 expert가 이러한 패턴들을 효과적으로 분리 학습하도록 설계했습니다.

## 2.3 비판적 검토 및 분석

1. **Depth=4 극단적 설정**: 일반적으로 depth가 낮아지면 representation capacity가 급격히 저하됩니다. **Stem/Decoder가 이를 충분히 보상**한다는 가설은 depth ablation(D=2,4,6,8)을 통해 정량적으로 검증되어야 하지만, 현재 보고서에 D=4 외의 실험 결과가 명시되어 있지 않아 추가 검증이 필요하다고 봅니다.
2. **Expert 수 K=8 과다 의심**: KI 문서(0081-v2)에는 "r16n2로 충분"하다고 기술되어 있으나, champion config는 여전히 **n=8**을 채택하고 있습니다. n=2와 n=8 간의 정밀한 비교 실험 결과가 누락되어 있어 최적화 여부가 불분명합니다.
3. **Load Balance Loss 비활성**: `lb_weight=0.0`(기본값) 설정으로 인해 expert collapse의 위험이 있으나 이에 대한 방어 메커니즘이 보이지 않습니다. `last_router_weights` 진단 슬롯은 마련되어 있지만 실제 로깅 여부가 확인되지 않습니다.
4. **einsum → loop 전환의 비효율성**: cudagraph 호환을 위해 expert loop으로 변경한 구조입니다. 하지만 K=8일 때 loop를 8회 실행하므로 K=2 대비 **4배의 overhead**가 발생합니다. K=8을 유지한다면 batched matmul을 도입하는 것이 성능상 효율적이라고 생각합니다.

## 2.4 구현 품질 검수

1. **adaLN chunk 확장 설계 우수**: `use_expert_gate=True`일 때 9단에서 10단 chunk으로 확장되도록 설계하고, Gate3와 독립된 expert gate를 분리하여 **깔끔한 확장성**을 확보했습니다.
2. **cudagraph 호환성 확보**: einsum 대신 plain matmul loop 방식을 사용하여 `torch.compile`에 친화적으로 설계했습니다.
3. **Forward signature 비대**: `STDiTBlock.forward` 메서드가 **13개의 인자**를 받고 있어 인터페이스가 지나치게 무겁습니다. 향후 dataclass나 context object를 도입하여 인자 전달 방식을 단순화할 것을 제안합니다.
4. **inspect.signature 런타임 호출 개선 필요**: L256 라인의 MoE 분기에서 `inspect` 라이브러리를 사용하여 매 forward마다 reflection 연산을 수행하므로 성능 저하가 우려됩니다. 이를 `hasattr`이나 static flag 방식으로 대체할 것을 권장합니다.

---

# 3. Deep Decoder — `HierarchicalDeepDecoder`

> 소스: [decoder.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/decoder.py#L120-L393)

## 3.1 아키텍처 설계 분석

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

## 3.2 태양광 도메인 적합성 검토

1. **Bilinear upsample 적합성**: Solar irradiance 이미지는 맑은 하늘 영역 등 부드러운 그래디언트(smooth gradient)가 많이 존재하므로, Bilinear 보간이 ConvTranspose보다 시각적으로 더 자연스러운 복원 결과를 낸다고 봅니다.
2. **Zero-init 마지막 Conv**: Persistence-residual 패러다임 하에서 **초기 예측치 = persistence**(변화 없음)로 상정하는 것이 타당합니다. 모델이 점진적으로 변화량(residual)을 학습해 나가는 구조는 도메인 특성에 잘 부합합니다.
3. **5단 점진 복원**: 32×32에서 512×512까지 단계적으로 복원하면서 구름 경계의 **다중 스케일 선명도(multi-scale sharpness)**를 계층적으로 잘 복원하고 있습니다.

## 3.3 비판적 검토 및 분석

1. **U-Net Skip 폐기로 인한 정보 병목**: Stem→Bottleneck→Decoder로 이어지는 흐름에서 256개 토큰의 bottleneck이 유일한 정보 전달 통로입니다. Skip connection이 폐기되면서 **Stem 단계에서 수집된 풍부한 중간 feature들이 유실**되고 있습니다. Bottleneck 용량이 이를 감당하기에 충분한지 정량적인 검증이 요구됩니다.
2. **SPADE 삽입 로직의 취약성**: L377-389 라인의 Sequential path에서 `isinstance(layer, nn.Conv2d)`를 검사하여 stage 완료를 판정하는 방식은 **Sequential 내부 구조에 의존적이어서 구조 변경 시 오동작할 위험**이 있습니다.
3. **dyn_t_schedule buffer 공유 문제**: skip과 dyn이 동일한 `_alpha_bar_for_skip` buffer를 공유하고 있어, 향후 스케줄을 독립적으로 가져가야 할 경우 코드를 분리할 필요가 있습니다.

## 3.4 구현 품질 검수

1. **Forward shape 검증 설계**: N != gh*gw 또는 H != hidden_size 조건에 대해 명시적으로 ValueError를 발생시켜 디버깅이 용이하도록 설계했습니다.
2. **dynamic_spade_layers 결합도 완화 필요**: st_dit.py 파일에서 ModuleList를 외부에서 직접 생성하여 decoder에 부착하는 구조는 **모듈 간의 결합도를 높입니다**. Decoder 내부에 SPADE 빌드 로직을 내장하는 방향이 설계상 더 깔끔할 것입니다.
3. **zero_init_last 안정장치 구현**: `use_skip` 경로와 Sequential 경로 모두에서 마지막 Conv 레이어를 정확히 찾아 zero-init을 수행하며, 예외 상황에 대비한 `RuntimeError` fallback 로직까지 안전하게 구성했습니다.
4. **t_gate 범용 인터페이스**: `_compute_t_gate` 메서드가 skip 및 dyn 양쪽 스케줄을 통합하여 유연하게 처리할 수 있도록 설계했습니다.

---

# 4. Dynamic Decoder — `DynamicHead` + `DecoderSPADE`

> 소스: [decoder.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/decoder.py#L449-L554) (DynamicHead), [decoder.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/decoder.py#L396-L446) (DecoderSPADE)

## 4.1 아키텍처 설계 분석

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

## 4.2 태양광 도메인 적합성 검토

1. **Velocity/Acceleration 물리적 의미**: `v(t) = I(t) - I(t-1)`는 10분간의 일사량 변화량이며, `a(t) = v(t) - v(t-1)`는 그 변화의 가속도입니다. **구름의 이동 및 생성, 소멸 과정을 물리적으로 직접 모델링**하고자 시도했습니다.
2. **Content-Motion 분리 합리성**: 정적인(static) 영역과 구름이 이동하는 동적인(dynamic) 영역의 예측 전략은 달라야 합니다. SPADE 변조를 통해 **motion이 발생하는 영역에서만 선택적으로 content를 변조**하는 메커니즘을 구축했습니다.
3. **IP(Input Perturbation)와의 시너지**: 단독 적용 시 FID 27.79에서 **IP와 결합할 경우 12.27로 개선**되었습니다. IP가 exposure bias를 완화하고 Dynamic branch가 motion 구조 정보를 주입하는 상호 보완적(orthogonal) 효과가 있음을 확인했습니다.

## 4.3 비판적 검토 및 분석

1. **`.detach()` 제약에 따른 한계**: Trunk gradient 오염을 방지하는 설계는 타당하나, 이로 인해 Dynamic Head가 Transformer representation의 학습 방향을 **역전파를 통해 직접 개선하도록 유도할 수 없다는 한계**가 있습니다. 즉, End-to-end 최적화가 아닌 2-stage 최적화에 가까운 동작 방식입니다.
2. **손실 함수 가중치 λ=10 고정 문제**: Dynamic loss의 가중치가 base loss 대비 10배로 매우 높은 편입니다. 스케줄링이나 adaptive weighting 없이 고정되어 있어, 학습 초기 단계에서 dynamic loss가 **gradient 흐름을 지배**할 리스크가 있다고 봅니다.
3. **InstanceNorm 생략에 따른 리스크**: 원본 SPADE는 InstanceNorm과 affine modulation을 결합하여 사용하지만, 본 구현에서는 정규화 없이 바로 `(1+γ)·x + β`를 적용합니다. 이로 인해 **feature magnitude가 불안정해질 위험**이 있어 검토가 필요합니다.
4. **Acceleration 채널 활용도 검증 부족**: GT acceleration은 2차 미분 값이므로 노이즈가 크게 증폭되는 특성을 보입니다. 실제로 acceleration이 content decoder에 유의미한 변조 신호를 주입하고 있는지에 대한 **ablation 분석이 누락**되어 있습니다.

## 4.4 구현 품질 검수

1. **Zero-init 마지막 Conv**: 학습 초기 dynamics 값을 0으로 출발시켜 Content Decoder만 정상 동작하도록 안전하게 설계했습니다.
2. **stage_features의 인스턴스 저장에 따른 누수 위험**: `self.stage_features = stage_features`와 같이 forward 연산 결과를 인스턴스 변수에 보관하고 있는데, **메모리 누수(stale reference 및 gradient graph 유지) 위험**이 있어 보입니다. 다만 입력단에서 `.detach()`를 수행하므로 실질적인 그래프 크기는 제한적입니다.
3. **해상도 불일치 자동 처리**: `DecoderSPADE.forward` 내부에서 `F.interpolate`를 통해 motion_feat과 x의 해상도를 자동으로 맞추도록 견고하게 구현했습니다.
4. **dyn_input_noise 적용 조건**: `self.training` 여부를 올바르게 체크하여 추론 단계에서는 깨끗한 입력이 보장되도록 작성했습니다.

---

# 종합 평가 매트릭스

| 구성요소 | Params | 설계 완성도 | 도메인 적합성 | 코드 품질 | 핵심 리스크 |
|----------|--------|------------|--------------|----------|------------|
| 1. Deep Stem | ~1.57M | 4/5 | 5/5 | 4/5 | Norm 부재, activation 불일치 |
| 2. ST-DiT Bottleneck | ~8.5M | 3/5 | 4/5 | 3/5 | D=4 극소, K=8 정당성, forward 비대 |
| 3. Deep Decoder | ~1.57M | 4/5 | 5/5 | 4/5 | Skip 부재 병목, SPADE 삽입 취약 |
| 4. Dynamic Decoder | ~0.3M | 4/5 | 5/5 | 4/5 | .detach() 제약, λ=10 고정, accel 활용 불명 |

## 핵심 권고사항

> [!IMPORTANT]
> 1. **Depth ablation 필요**: D=4가 최적인지 D=6,8과 비교하는 실험을 시급히 진행해야 합니다.
> 2. **Expert 수 정리**: K=8 vs K=2 실험 결과를 문서화하고, LB loss 활성화를 검토해야 합니다.
> 3. **Forward signature 리팩터링**: 13개의 개별 인자를 context dataclass 형태로 묶어 가독성을 높일 것을 추천합니다.
> 4. **Sequential SPADE 삽입 로직 개선**: stage 인덱스를 명시적으로 지정하여 조작하는 방식으로 전환해야 합니다.

> [!TIP]
> 현재 champion (FID 4.48, MAE 0.1193) 모델의 핵심 성공 비결은 아키텍처 자체의 우수성보다는 **E2E training + cosine LR + Input Perturbation의 종합적 시너지 효과**에 가깝습니다. 즉, 복잡한 아키텍처 수정보다는 **학습 프로세스 레시피의 최적화**가 실질적인 도약을 만들어냈다고 해석됩니다 (FID 12.27 → 4.78).
