# Spatial MoE for Raw Pixel Pipeline (v3)

> Raw pixel(1ch x 512x512, P=16, 1024 tokens) 환경에서
> DWT 없이 지역별 전문화를 달성하는 경량 MoE 설계안.
>
> v2: 코드 리뷰 반영 (LoRA init fix, einsum 최적화, 3-stage 통합, CLS 처리)
> v3: 학술/도메인/시스템/구현 4관점 리뷰 반영

---

## 1. 동기

DWT 파이프라인에서 MoE가 효과적이었던 이유는 **LF/HF 분리 구조** 위에서 라우터가 주파수별 전문화를 형성할 수 있었기 때문이다. Raw pixel에는 이 구조가 없으므로, **공간적 내용(구름 가장자리 vs 맑은 영역 vs 균일 구름)** 기반으로 전문화를 유도해야 한다.

### 선행연구

LoRA를 MoE expert로 사용하는 접근은 LLM 분야에서 선례가 있다:
- **LoRAMoE** (Dou et al., 2024): 여러 LoRA adapter를 expert로 구성하고 soft routing으로 task별 전문화 달성.
- **MoLoRA** (Zadouri et al., 2023): LoRA의 low-rank 구조를 MoE expert로 활용하여 파라미터 효율적 전문화.

본 설계는 이 패턴을 vision transformer의 **공간적 전문화**(spatial specialization)로 확장한다. LLM의 task/domain 축 대신 **토큰의 공간적 내용**(edge vs flat) 축에서 전문화를 유도하는 것이 차별점.

### 목표

- 토큰의 내용(밝기, 기울기, 복잡도)에 따라 서로 다른 처리 경로 활성화
- 파라미터/연산 추가 최소화 (현행 대비 5% 이내)
- torch.compile 호환

---

## 2. 설계안 비교

### Option A: Shared FFN + LoRA Expert Residual (권장)

```
토큰 x: (B, 1024, 512)
    │
    ▼
┌─────────────────────────────────┐
│  Shared FFN (기존 MLP 그대로)   │  ← 전체 토큰 공통 기본 연산
│  h = FFN(x)                     │     파라미터: 2.1M (기존)
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Router: x → softmax(W·x)      │  ← 토큰별 expert 선택 (soft)
│  weights: (K,) per token        │     파라미터: 512 × K (2개면 1K)
│  (K=2: sigmoid와 수학적 등가)   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Expert Residuals (LoRA-style)  │  ← K개 경량 보정 모듈
│  e_k(x) = up_k · (down_k · x)  │     down_k: (512, r), up_k: (r, 512)
│  r = 16 (rank)                  │     expert당: 512×16×2 = 16.4K params
│                                 │     K=2 → 32.8K params 추가 (~0.4%)
│  delta = Σ_k w_k · e_k(x)      │     soft weighted sum
└─────────────────────────────────┘
    │
    ▼
  output = h + delta              ← shared 결과 + 영역별 보정
```

**장점:**
- Shared FFN이 기본 연산 담당 → 학습 안정성 유지
- LoRA init(down=kaiming, up=zero) → 초기 delta=0, baseline 동등 보장 + gradient 활성
- Soft routing → torch.compile 완전 호환 (dynamic shape 없음)
- 파라미터 추가: 0.4% (K=2, r=16, 4블록)

**단점:**
- LoRA rank가 낮으면 표현력 제한 (r=8/16/32 ablation 가능)
- Soft routing이 sharp specialization을 형성하기 어려울 수 있음

---

### Option B: Soft Spatial FiLM

```
토큰 x → Router: region_score = sigmoid(W · x)  ← (B, 1024, 1) scalar
FFN(x) → output = FFN(x) * (1 + α · gamma) + β
          where gamma, beta = f(region_score)
```

- 파라미터: ~1.5K per block (전체의 ~0.01%)
- 장점: 극경량
- 단점: scale/shift만으로는 전문화 수준이 낮음

---

### Option C: 2-Expert Sparse MoE (표준)

```
토큰 x → Router: top-1 선택 → Expert 0 또는 Expert 1
각 Expert = 독립 FFN (512 → 2048 → 512)
```

- 파라미터: 2 × 2.1M = 4.2M per block (100% 증가)
- 장점: 완전한 전문화 가능
- 단점: 파라미터 2배, sparse routing은 CUDA Graphs 비호환

---

## 3. 권장안: Option A 상세 설계

### 3.1 Router 설계

```
router_input = x                    # (B, N, D) = (B, 1024, 512)
router_logits = Linear(D, K)(x)     # (B, N, K) = (B, 1024, 2)
router_weights = softmax(logits)    # (B, N, K), dim=-1
```

K=2 expert의 기대 역할:
- Expert 0: "구조 보존" (맑은 영역, 균일 구름) → 평탄 영역의 값 유지 강화
- Expert 1: "디테일 보정" (구름 가장자리, 급변 영역) → edge 정밀도 강화

> 이 역할 분화는 **가설**이며, 학습 데이터와 loss gradient에 의해 실제로 형성되는지는 실험적 검증이 필요하다. 라우터 weight 시각화(per-token routing 분포 vs spatial gradient 상관)로 사후 확인.

> K=2일 때 softmax(a, b)는 sigmoid(a-b)와 수학적 등가. 전문화는 라우터 자체가 아닌 expert A/B 행렬의 차이에서 발생.

### 3.2 Expert Residual (LoRA)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialExpertResidual(nn.Module):
    """Shared FFN과 병렬로 동작하는 LoRA Expert Residual.

    K개 expert가 각각 low-rank transform을 수행하고,
    router가 토큰별 soft weight로 합산하여 영역별 보정 delta 생성.

    선행연구: LoRAMoE (Dou et al., 2024), MoLoRA (Zadouri et al., 2023)의
    LoRA-as-expert 패턴을 vision spatial specialization으로 확장.
    """
    def __init__(self, dim: int = 512, rank: int = 16, num_experts: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.rank = rank

 # Router: per-token soft expert selection
        self.router = nn.Linear(dim, num_experts)

 # LoRA down/up per expert
 # down: kaiming init (gradient 활성 보장)
 # up: zero init (초기 delta=0 → baseline 동등)
        self.down = nn.Parameter(torch.empty(num_experts, dim, rank))
        self.up = nn.Parameter(torch.zeros(num_experts, rank, dim))

 # Per-expert 2D slice init (3D 텐서 직접 kaiming 시 fan 계산 오류 방지)
        for k in range(num_experts):
            nn.init.kaiming_uniform_(self.down[k], a=math.sqrt(5))  # (D, r)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D) spatial 토큰 (CLS 제외 상태)
        Returns:
            delta: (B, N, D) expert 보정 잔차
            lb_loss: scalar, load balance loss (0 if not needed)
        """
        weights = F.softmax(self.router(x), dim=-1)           # (B, N, K)

 # Step 1: down projection
        mid = torch.einsum('bnd,kdr->bnkr', x, self.down)    # (B, N, K, r)

 # Step 2: weight 적용 (K축 contract 전에 곱하여 (B,N,K,D) 중간 텐서 제거)
        mid = mid * weights.unsqueeze(-1)                     # (B, N, K, r)

 # Step 3: up projection + K축 contract
        delta = torch.einsum('bnkr,krd->bnd', mid, self.up)  # (B, N, D)

 # Load balance loss: expert별 평균 weight의 균등성
 # f_k = mean(weights[:,:,k]), 목표: 모든 f_k ≈ 1/K
        f = weights.mean(dim=(0, 1))                          # (K,)
        lb_loss = self.num_experts * (f * f).sum()             # scalar

        return delta, lb_loss
```

> [!IMPORTANT]
> **LoRA init 규칙**: `down`은 per-expert 2D slice별 `kaiming_uniform_`, `up`은 zero init.
> (1) 양쪽 모두 zero이면 gradient가 영원히 0 → expert 영구 dead.
> (2) 3D 텐서에 직접 kaiming 적용 시 fan_in/fan_out 계산이 반전됨 → per-expert 2D slice 필수.

> [!TIP]
> **einsum 메모리 최적화**: weight를 두 번째 einsum 전에 적용하여 `(B, N, K, D)` 중간 텐서 제거.
> B=4, N=1024, K=2, D=512 기준: 64MB → 0.5MB (128배 절감).

> [!NOTE]
> **torch.compile 주의**: `torch.einsum`은 compile 시 내부적으로 matmul로 분해된다.
> K=2 고정이면 `x @ down[0]`, `x @ down[1]`로 풀어쓰는 것이 compile 친화적일 수 있다.
> 단, K 확장성을 포기하게 되므로 우선 einsum으로 구현 후 벤치마크 결과로 판단.

### 3.3 STDiTBlock 3-stage 통합

우리 STDiTBlock은 3-stage 구조: self-attn → cross-attn → MLP. gate는 stage별로 존재.
Expert는 **stage 3 (MLP)와 병렬**, gate3로 한번에 스케일링.

MLP는 1회만 호출하고, delta만 spatial 토큰에 적용:

```python
# STDiTBlock.forward — stage 3 (MLP)
x_mod3 = modulate(self.norm3(x), shift3, scale3)

mlp_out = self.mlp(x_mod3)                         # MLP 1회 호출 (전체 토큰)

if self.spatial_expert is not None:
    if num_special > 0:
 # CLS 제외, spatial 토큰만 expert 적용
        spatial_input = x_mod3[:, num_special:]      # (B, H*W, D)
        delta, lb_loss = self.spatial_expert(spatial_input)
        mlp_out[:, num_special:] = mlp_out[:, num_special:] + delta
    else:
        delta, lb_loss = self.spatial_expert(x_mod3)
        mlp_out = mlp_out + delta

x = x + gate3.unsqueeze(1) * mlp_out
```

- Cross-attn은 변경하지 않음
- MLP forward 1회 통일 (이전 v2의 이중 호출 제거)
- CLS는 spatial expert 우회 → Swin 설계와 일관
- `lb_loss`는 model forward의 `aux_loss`에 합산하여 기존 MoE 인프라 재활용

### 3.4 블록별 적용 전략

| 전략 | CLI 값 | 설명 | 비용 |
|:---|:---|:---|:---:|
| 전 블록 | `--spatial_expert_blocks all` | 모든 블록에 적용 | 0.4% |
| **후반만** | `--spatial_expert_blocks last_half` | coarse 블록은 공통, fine만 전문화 **(기본)** | 0.2% |
| 마지막만 | `--spatial_expert_blocks last_only` | 최종 출력 직전 블록만 | 0.1% |

> **기본값: `last_half`**. 초기 블록(0-1)의 토큰 임베딩은 patchify 직후라 raw 패치 평균에 가까워, 라우터가 edge/flat 분류를 형성하기 어렵다. self-attention 1회 이상 통과 후의 블록(2-3)에서 인접 토큰 정보가 통합되어 라우터 입력 품질이 향상됨.

---

## 4. 연산 비용 분석

D=512, N=1024, r=16, K=2 기준.

| 항목 | FLOPs per block | 비율 |
|:---|:---:|:---:|
| Shared FFN | 4,194,304 (4.2M) | 100% |
| Router | 1,024 × 512 × 2 = 1.05M | 0.025% |
| LoRA down (einsum) | 1,024 × 512 × 16 × 2 = 33.6M | 0.8% |
| LoRA up (einsum) | 1,024 × 16 × 512 × 2 = 33.6M | 0.8% |
| Weighted sum | 1,024 × 2 × 16 = 32.8K | ~0% |
| **Expert 총 추가** | **~69M** | **~1.6%** |

`last_half` (2블록) 합산: ~138M FLOPs 추가 (전체 모델 ~27G 대비 **~0.5%**)

**메모리**: expert 파라미터 2블록 x 33K = 66K params (0.25MB), VRAM 무시 가능.
einsum 최적화 적용 시 중간 텐서 최대 크기: (B, N, K, r) = 4x1024x2x16 = 0.5MB.

---

## 5. Ablation 계획

최소 3축 ablation으로 각 설계 선택의 기여도를 분리:

### 5.1 Ablation Grid

| 축 | 수준 | 기본값 |
|:---|:---|:---:|
| LoRA rank (r) | 8, **16**, 32 | 16 |
| Expert 수 (K) | **2**, 3, 4 | 2 |
| 적용 블록 | last_only, **last_half**, all | last_half |

총 3 x 3 x 3 = 27 조합이나, 교호작용이 낮으므로 각 축 독립 sweep으로 축소 가능:
- r sweep: r={8, 16, 32}, K=2, blocks=last_half → 3 runs
- K sweep: r=16, K={2, 3, 4}, blocks=last_half → 3 runs
- blocks sweep: r=16, K=2, blocks={last_only, last_half, all} → 3 runs
- **총 7 runs** (기본값 1회 공유)

### 5.2 분석 항목

| 분석 | 방법 |
|:---|:---|
| 전문화 형성 여부 | Per-token routing weight 시각화 vs spatial gradient 상관 |
| Edge/flat 분류 품질 | GT edge mask와 router weight의 IoU |
| 성능 기여 | MSE, MAE, SSIM 비교 (baseline vs spatial expert) |

---

## 6. 구현 계획

### 파일 구조

| 파일 | 역할 | 규모 |
|:---|:---|:---:|
| `stdit/models/spatial_expert.py` [NEW] | SpatialExpertResidual (per-expert init + einsum 최적화 + lb_loss 반환) | ~80줄 |
| `stdit/models/st_dit.py` [MODIFY] | 블록 생성 분기, MLP stage에 delta + CLS split, aux_loss 집계 | ~30줄 |
| `stdit/config.py` [MODIFY] | USE_SPATIAL_EXPERT / RANK / NUM / BLOCKS / LB config | ~10줄 |
| `main.py` [MODIFY] | argparse | ~10줄 |

### CLI / Config

```
--use_spatial_expert                    # 활성화
--spatial_expert_rank 16                # LoRA rank (기본: 16, ablation: 8/32)
--spatial_expert_num 2                  # expert 수 (기본: 2)
--spatial_expert_blocks last_half       # 적용 범위 (all / last_half / last_only)
--use_spatial_expert_lb_loss            # load balance loss (선택적)
--spatial_expert_lb_weight 0.01         # LB loss 가중치
```

### 호환 검증

| 조합 | 호환 | 비고 |
|:---|:---:|:---|
| `--use_spatial_expert` + `--use_raw_pixel` | ✅ | 주 사용 대상 |
| `--use_spatial_expert` + `--use_dwt` | ✅ | 가능하나 BiDir MoE와 중복 |
| `--use_spatial_expert` + `--use_freq_moe` | ⚠️ | MLP 경로에 두 MoE 중복, mutex 권장 |
| `--use_spatial_expert` + torch.compile | ✅ | dense 연산, dynamic shape 없음 |
| `--use_spatial_expert` + `--use_persistence_noise` | ✅ | orthogonal (noise vs routing) |
| test_only.py (ckpt 로딩) | ⚠️ | `strict=False` 확인 필요 (비활성 시 weight mismatch) |

---

## 7. 예상 효과와 위험

### 기대

| 항목 | 근거 |
|:---|:---|
| Edge 영역 정밀도 향상 | Edge 전담 expert가 gradient 급변 패턴 학습 (가설, 실험 검증 필요) |
| 평탄 영역 값 보존 | Structure expert가 불필요한 변동 억제 (가설) |
| SSIM 개선 | 영역별 최적 처리 → 국소적 구조 유사도 상승 (가설) |

### 위험

| 항목 | 대응 |
|:---|:---|
| Specialization 미형성 | Kaiming/zero init으로 baseline 동등 보장. 악화 없음. routing 시각화로 사후 확인 |
| 라우터 collapse | `--use_spatial_expert_lb_loss`로 load balance 강제. lb_loss 반환 패턴 통일 |
| torch.compile 충돌 | Soft routing (dense 연산), dynamic shape 없음 |
| 초기 블록 라우터 무효 | 기본값 `last_half`로 self-attn 통과 후 블록에만 적용 |
| USE_FREQ_MOE와 중복 | validator에서 mutex 처리 |

---

## 8. Persistence Noise와의 시너지

| 모듈 | 역할 | 적용 위치 |
|:---|:---|:---|
| Persistence Noise | Edge 영역에 더 강한 forward noise → 학습 신호 집중 | Diffusion forward |
| Spatial Expert | Edge 영역에 전담 expert → 처리 능력 집중 | Transformer block |

두 모듈은 **서로 다른 축에서 edge 영역 처리를 강화**한다:
- Persistence noise: "얼마나 세게 학습시킬 것인가" (noise 수준)
- Spatial expert: "무엇으로 처리할 것인가" (전문가 경로)

Orthogonal하므로 결합 가능. 다만 두 모듈 동시 투입 시 변수가 2개이므로, 단독 효과 확인 후 결합이 실험 설계상 바람직함.
