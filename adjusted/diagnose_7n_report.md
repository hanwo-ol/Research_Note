# 7N Variance Loss 진단 리포트

## 진단 대상
- **Baseline**: `4_FreqMoE_TrueMoE_STAwareRouter_Cap1.5_CosineLR` (SSIM=0.500)
- **7N**: `7N_VarLossLog_NoPixelLoss` — Variance Loss(log) 단독, pixel loss 제거 (SSIM=0.431)

---

## 1. 레이어 유형별 가중치 변화량

| Group | Mean |ΔW|/|W| | Max |ΔW|/|W| | Count |
|-------|:---:|:---:|:---:|
| **MoE Experts** | **1.376** | 1.480 | 128 |
| **Shared Expert** | **1.387** | 1.453 | 32 |
| Embedder | 1.341 | 1.510 | 19 |
| Other (interaction 등) | 1.297 | **1.573** | 116 |
| Attention | 1.289 | 1.471 | 64 |
| Router | 1.020 | 1.497 | 40 |
| Final Layer | 0.935 | 1.315 | 6 |
| Normalization | 0.824 | 1.560 | 32 |

> [!IMPORTANT]
> **전체 레이어의 상대 변화량이 0.8~1.4x**입니다. 이는 7N이 baseline과 **완전히 다른 수렴 지점**에 도달했음을 의미합니다. Variance loss가 학습 신호를 근본적으로 바꿔서 baseline과 호환되지 않는 가중치 공간으로 이동한 것입니다.

### 가장 많이 변한 레이어 Top 5

| Layer | |ΔW|/|W| | Params |
|-------|:---:|:---:|
| B6.interaction.up_proj.weight | **1.573** | 262K |
| B6.high_expert.norm.bias | **1.560** | 512 |
| B5.interaction.up_proj.weight | **1.540** | 262K |
| B7.interaction.up_proj.weight | **1.538** | 262K |
| x_embedder_low.proj.weight | **1.510** | 33K |

> **Interaction (저고주파 통신) 모듈이 가장 크게 변함** → Variance loss가 채널 간 가중치를 재분배하면서 저고주파 상호작용 패턴이 완전히 달라짐.

---

## 2. 전문가 분화 (Expert Specialization) 분석

전문가 쌍 간 Cosine Similarity (낮을수록 분화가 잘 된 것):

| Block | Branch | BL Mean | 7N Mean | Δ | 판정 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | low | 0.0106 | 0.0047 | -0.006 | → 약간 분화 |
| 1 | low | 0.0130 | 0.0043 | -0.009 | → 약간 분화 |
| 2 | low | 0.0116 | 0.0042 | -0.007 | → 약간 분화 |
| 3 | low | 0.0113 | 0.0041 | -0.007 | → 약간 분화 |
| 4 | low | 0.0135 | 0.0059 | -0.008 | → 약간 분화 |
| 5 | low | 0.0173 | 0.0062 | **-0.011** | ✅ 분화 증가 |
| 6 | low | 0.0181 | 0.0075 | **-0.011** | ✅ 분화 증가 |
| 7 | low | 0.0292 | 0.0135 | **-0.016** | ✅ 분화 증가 |

> [!NOTE]
> **7N에서 전문가 분화는 오히려 개선됨** (특히 심층 블록 5-7). 즉, variance loss 자체가 라우팅을 붕괴시킨 것은 아닙니다.

---

## 3. 핵심 진단 결론

### Variance Loss 단독 사용이 실패한 근본 원인

1. **구조 가이드 부재**: Variance loss는 `log(1 + σ_ch) * MSE_ch` 형태로 채널별 **분산 크기에 비례한 가중치**만 적용합니다. 이것은 "어느 채널에 더 집중할지"만 알려주지, "올바른 구조를 복원하라"는 신호는 제공하지 못합니다.

2. **Gradient 노이즈 증폭**: Min-SNR이 이미 timestep별로 최적 가중치를 제공하는데, variance loss가 **중복 가중치**를 추가하여 gradient signal의 SNR을 오히려 저하시킵니다.

3. **Interaction 모듈 과도한 변형**: 상대 변화량 Top 3이 모두 `interaction.up_proj`(저고주파간 정보 교환) — variance loss의 채널별 가중치가 저고주파 밸런스를 깨트렸습니다.

### 대조적으로 7M (L1 + VarLoss)이 성공한 이유

7M에서 variance loss는 **L1 pixel loss의 보조 항**으로 작동했습니다. L1이 구조적 복원 신호를 제공하고, variance loss가 CP6(채널 가중치 편향)를 보정하는 역할 분담이 이루어진 것입니다.

### Per-Channel Min-SNR (7O)이 상위 호환인 이유

| 비교 항목 | Variance Loss | Per-Channel Min-SNR |
|-----------|:---:|:---:|
| 가중치 적용 위치 | auxiliary loss (추가 항) | **diffusion loss 자체** |
| 이론적 근거 | 경험적 분산 비례 | **Fisher information 최적해** |
| gradient 간섭 | Min-SNR과 중복 | Min-SNR의 **자연스러운 확장** |
| 구조 복원 신호 | ❌ 없음 | ✅ 내장 |
