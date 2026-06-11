# ST-DiT Backbone 병목 분석 및 확장 전략

> [!NOTE]
> 작성일: 2026-05-21 | 근거: Stem V2/V3 실험 전체 결과

---

## 병목 진단: 왜 PSNR이 안 올라가는가

### 실험적 증거

| 개선 대상 | FID 변화 | PSNR 변화 | 결론 |
|----------|---------|---------|------|
| Stem V2 (Res) | -38% ✅ | ≈ 0 | Stem은 FID 경로 |
| Stem V3 (t-gate only) | -71% ✅✅ | **-0.56 dB** ❌ | Stem t는 FID-PSNR trade-off |
| Decoder t-gate | -33% ✅ | ≈ 0 | Decoder도 FID 경로 |
| Both t-gate | ≈ | ≈ 0 | 양쪽 대칭 시 안정 |

**일관된 패턴**: Stem/Decoder를 아무리 개선해도 PSNR(pixel accuracy)은 올라가지 않는다.

### 구조적 원인

```
128×128 (16,384 pixels)
    ↓ [Stem: stride-2 ×4]
8×8 = 64 tokens (each token ≈ 256 pixels)
    ↓ [4× Transformer Block (AdaLN, Self-Attn, FFN)]
64 tokens
    ↓ [Decoder: upsample ×4]
128×128 prediction
```

**핵심 제약**:
- **Token 수 = 64**: 128×128 이미지를 64개 토큰으로 압축. 토큰당 16×16=256 pixel 담당
- **Depth = 4**: DiT-XL(28), U-ViT(13) 대비 극도로 얕음
- **단일 해상도**: U-Net과 달리 multi-scale 처리 없음. 모든 추론이 8×8 해상도에서 수행

PSNR의 상한은 **"64개 토큰이 128×128 이미지를 얼마나 잘 표현할 수 있는가"**에 의해 결정된다.

---

## (참고) Depth/Patch 스케일링 — 엔지니어링 레벨

> [!WARNING]
> **Depth 증가, Patch 감소는 하이퍼파라미터 스케일링이지 방법론적 기여가 아님.**
> 이전 실험에서 depth 증가는 64 토큰의 정보 용량 상한을 넘지 못해 효과 제한적이었음.
> 참고용으로만 기록.

| Config | 변화 | 비고 |
|--------|------|------|
| `--depth 8` | 블록 4→8 | 같은 64 토큰, 더 깊은 처리. 이미 포화 의심 |
| `--patch_size 8` | 토큰 64→256 | B2+grad_ckpt ≈ 20GB. H200 가능 |

---

## 전략 3: Multi-Scale / Hierarchical (U-ViT 스타일)

### 원리

현재 ST-DiT는 **단일 해상도**(8×8)에서 모든 처리를 수행:

```
[현재] 단일 해상도
64 tokens → [Block1] → [Block2] → [Block3] → [Block4] → 64 tokens
```

U-ViT (Bao et al., 2023)나 MDT (Gao et al., 2023)는 **U-Net처럼 다운→업 구조**를 사용:

```
[U-ViT 스타일] 다중 해상도
256 tokens (16×16) → [Block1] → [Block2]
                        ↓ downsample (token merge)
                    64 tokens (8×8) → [Block3] → [Block4]
                        ↓ downsample
                    16 tokens (4×4) → [Block5] (bottleneck)
                        ↑ upsample (token split)
                    64 tokens → [Block6] + skip → [Block7]
                        ↑ upsample
                    256 tokens → [Block8] + skip → output
```

### 수학적 의미

- **고해상도 단계** (256 tokens): 세밀한 spatial detail 처리 (high-frequency)
- **저해상도 단계** (16 tokens): 전역적 구조/맥락 파악 (low-frequency)
- **Skip connection**: U-Net과 동일 — 고해상도 정보 보존

이는 wavelet decomposition과 유사한 구조:
- 다운샘플 = low-pass filter (coarse 정보 추출)
- 스킵 = high-pass residual (detail 보존)
- 업샘플 + skip = 재구성

### Token Merge/Split 방법

```python
# Token Merge (downsample): 2×2 인접 토큰 → 1 토큰
# (B, N, D) → reshape → (B, H//2, 2, W//2, 2, D) → (B, H//2, W//2, 4D) → Linear(4D, D)

# Token Split (upsample): 1 토큰 → 2×2 토큰
# (B, N, D) → Linear(D, 4D) → reshape → (B, 2H, 2W, D)
```

### 기존 구현 현황

현재 코드베이스에 **구현되어 있지 않음**. 새로운 Transformer backbone 설계가 필요.

### 장단점

| 장점 | 단점 |
|------|------|
| **Multi-resolution 처리** (U-Net의 핵심 장점) | 새로운 backbone 설계 필요 (대규모 리팩터링) |
| 고해상도에서 detail, 저해상도에서 context | 기존 Spatial Expert, MoE 등과의 호환성 불확실 |
| Skip connection으로 정보 보존 | 구현 복잡도 높음 |
| PSNR + FID 동시 개선 가능 | 검증에 시간 소요 |

> [!WARNING]
> **구현 규모가 크다.** 기존 `STDiTBlock`의 forward 루프를 완전히 재설계해야 하며, token merge/split 레이어, skip connection 관리, AdaLN 조정 등이 필요. 2~3일 규모의 작업.

---

## 전략 4: Cross-Scale Attention (기구현)

### 원리

Main path는 P=16 (64 tokens)을 유지하면서, **별도 fine encoder**가 P=8 (256 tokens)으로 인코딩한 토큰을 KV로 제공:

```
[Main Path]   P=16, 64 tokens (Q)
                ↕ Cross-Attention
[Fine Encoder] P=8, 256 tokens (K, V) — context frame 인코딩
```

- **Q = main path tokens** (coarse, 64개): denoising 대상
- **K, V = fine tokens** (fine, 256개): 고해상도 참조 정보
- Main Self-Attention cost는 **불변** (64×64)
- Cross-Attention cost: 64×256 (affordable)

### 기존 구현 분석

[ir_cross_attn.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/ir_cross_attn.py)에 구현됨:

```python
class IRCrossAttn:
    # Q = x (main block feature, 64 tokens)
    # K, V = fine_tokens (256 tokens from fine encoder)
    # out_proj: zero-init → baseline 동등 시작
    # 매 block에서 x = x + CrossAttn(Q=x, KV=fine_tokens)
```

[st_dit.py L700-719](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/st_dit.py#L700-L719):

```python
# Fine encoder: context frame을 P=8로 인코딩
self.cs_fine_encoder = HierarchicalDeepStem(
    in_channels=1, hidden_size=512, patch_size=cs_patch,
)
# 각 block에서 cross-attention 적용
self.cs_cross_attns = nn.ModuleList([
    CrossScaleAttn(512, 8) if i in layer_indices else None
    for i in range(depth)
])
```

### U-ViT와의 차이

| | U-ViT Multi-Scale | Cross-Scale Attention |
|---|---|---|
| **Main path** | multi-resolution (변동) | 단일 해상도 (64 tokens 고정) |
| **Fine 정보** | skip connection (자기 자신) | 별도 encoder (context frame) |
| **Computation** | backbone 전체 변경 | Cross-Attn 추가만 |
| **구현 규모** | 대규모 리팩터링 | **이미 구현됨** |
| **Main Self-Attn cost** | 증가 (고해상도 단계) | **불변** |

### VRAM 영향

- Fine encoder: HierarchicalDeepStem(P=8) ≈ 추가 ~2GB
- Cross-Attn per block: 64×256 attention ≈ 무시 가능
- **총 추가: ~3-5 GB** (현재 25GB → ~30GB)

### 장단점

| 장점 | 단점 |
|------|------|
| **이미 구현되어 있음** | Fine encoder는 context frame만 인코딩 (noisy x_t가 아님) |
| Main Self-Attn cost 불변 | Fine tokens가 clean signal 기반 → noisy denoising과 불일치 가능 |
| VRAM 추가 최소 | Context frame 의존 → 새로운 시간대 예측에 한계 |
| zero-init 안전 도입 | 실험 검증 미완료 |

> [!IMPORTANT]
> Cross-Scale은 **가장 경제적인 접근**이지만, fine encoder가 context frame (과거 관측)을 인코딩하는 구조라 "미래 예측"에서의 효과는 제한적일 수 있다. noisy x_t 자체를 fine encoding하는 변형이 더 효과적일 수 있음.

---

## 전략 5: 전문가 구조 (Spatial Expert) 재설계

### 현재 구현

두 가지 전문가 시스템이 공존:

**A. SpatialExpertResidual** ([spatial_expert.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/spatial_expert.py))
- K개 LoRA expert (rank=16), soft routing
- `delta = Σ_k w_k · (up_k · down_k · x)` → shared FFN에 residual 추가
- 현재 baseline에서 사용 중: `--spatial_expert_rank 16 --spatial_expert_num 8 --spatial_expert_blocks all`

**B. Hetero Experts** ([hetero_experts.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/hetero_experts.py))
- 8가지 이종 전문가: Global(7×7), Axial(1×5+5×1), Local(3×3), Pure(1×1), Dilated(3×3 d=2), MultiScale(3+5), Strip(1×11+11×1), GatedSpatial
- 블록별 서로 다른 전문가 타입 배정 (Depth-Hetero)

### 현재 구조의 근본적 문제

> [!CAUTION]
> **8×8 grid에서 7×7 Conv는 거의 전체 이미지를 커버한다.** 현재 P=16에서 spatial 해상도는 8×8. 이 해상도에서 수용장 계층(3×3, 5×5, 7×7)의 차별성이 거의 없다.

| Expert | Kernel | 8×8에서의 커버리지 |
|--------|--------|:---:|
| Local | 3×3 | 37.5% |
| Dilated | 3×3 d=2 (eff 5×5) | 62.5% |
| Global | 7×7 | **87.5%** |
| Strip | 11×1 | **전체 초과** |

**8×8 grid에서 3×3과 7×7의 차이는 미미하다.** 수용장 기반 전문가 분화가 의미를 가지려면 **더 높은 spatial 해상도**(P=8 → 16×16)가 필요.

### LoRA Expert의 한계

```python
# 현재: rank=16 LoRA expert
delta = Σ_k w_k · (up_k · down_k · x)  # rank-16 low-rank residual
```

- rank=16으로 512차원 공간에서 표현 가능한 변형이 제한적 (16/512 = 3.1%)
- 모든 expert가 동일한 rank → 복잡한 패턴과 단순한 패턴에 동일 용량
- Soft routing (softmax)은 hard routing 대비 expert 특화가 약함

### 학술적으로 의미 있는 개선 방향

#### 5-A. t-conditioned Expert Routing

##### 동기

현재 router는 token content만으로 expert를 선택한다:

```python
# 현재
weights = softmax(Linear(x))  # x: (B, N, D) → (B, N, K)
```

그러나 diffusion에서 **동일한 x라도 t에 따라 전혀 다른 처리가 필요하다:**

| t 범위 | x_t의 상태 | 필요한 처리 | 적합한 expert |
|--------|----------|-----------|:---:|
| t ≈ 999 | 거의 순수 noise | 전역 구조 복원 (coarse) | Global / Structure |
| t ≈ 500 | 구조 + noise 혼재 | 중간 스케일 디테일 정제 | Axial / Multi-scale |
| t ≈ 0 | 거의 clean | 미세 보정 (fine-grained) | Local / Gated |

DDPM U-Net은 모든 레이어에 t가 주입되어 자연스럽게 이 분화가 이루어진다. 그러나 현재 ST-DiT의 expert router에는 **t 정보가 전혀 없다** — AdaLN이 Transformer block 자체에는 t를 주지만 router는 blind.

##### 수학

```
현재:  w_k = softmax(W_router · x_n)
제안:  w_k = softmax(W_router · (x_n + γ(t)))
       γ(t) = Linear(t_emb)  ∈ R^D     (t_emb은 기존 sinusoidal embedding)
```

또는 FiLM 스타일:

```
제안 (FiLM):  w_k = softmax(W_router · (scale(t) ⊙ x_n + shift(t)))
              scale(t), shift(t) = MLP(t_emb).chunk(2)
```

##### 기대 효과

```
              ┌─ t≈999: Global expert 지배 (구조 복원)
x_n + γ(t) → │─ t≈500: Mixed routing (중간 스케일)
              └─ t≈0:   Local expert 지배 (디테일 보정)
```

- **FID 개선**: noise level별 최적 expert 선택 → 생성 품질 향상
- **PSNR 개선**: low-t에서 Local expert에 집중 → pixel 정확도 향상
- t-gate (Stem/Decoder)와 자연스러운 확장: "t-aware pipeline"의 완성

##### 구현 (현재 코드 기준)

```python
# spatial_expert.py 수정
class SpatialExpertResidual(nn.Module):
    def __init__(self, dim, rank=16, num_experts=2, lb_weight=0.0,
                 use_t_routing=False, t_emb_dim=None):
        ...
        self.router = nn.Linear(dim, num_experts)
        
        # [NEW] t-conditioned routing
        self.use_t_routing = use_t_routing
        if use_t_routing:
            self.t_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim or dim, dim),
            )
            nn.init.zeros_(self.t_proj[-1].weight)
            nn.init.zeros_(self.t_proj[-1].bias)  # zero-init → 초기 baseline 동등

    def _compute_delta(self, x, t_emb=None):
        if self.use_t_routing and t_emb is not None:
            # t_emb: (B, D) → (B, 1, D) broadcast
            router_input = x + self.t_proj(t_emb).unsqueeze(1)
        else:
            router_input = x
        weights = F.softmax(self.router(router_input), dim=-1)
        ...
```

- zero-init으로 시작 → 학습 초기 t routing은 비활성, 기존과 동일
- 학습이 진행되며 γ(t)가 커지면서 t-dependent routing이 활성화
- **기존 `t_emb`는 이미 STDiTBlock에서 계산됨** → forward에서 넘겨주기만 하면 됨

##### 구현 비용: **~0.5일** (router에 t_proj 추가 + forward 인자 전달)

---

#### 5-C. P=16에서 유효한 Expert 재설계

##### 현재 문제 재확인

8×8 grid에서 Conv 기반 수용장 분화는 무의미하다. 하지만 **expert의 차별화 축을 "수용장"에서 다른 것으로 바꾸면 P=16에서도 의미 있다.**

##### 차별화 축 대안

Conv kernel size 대신 **연산 유형**으로 expert를 차별화:

| Expert 유형 | 연산 | 포착하는 것 | P=16 유효? |
|------------|------|-----------|:---:|
| **Channel-Group** | 채널을 그룹으로 나눠 독립 처리 | 채널 간 다른 물리량 | ✅ |
| **Token-Interaction** | 토큰 간 linear mixing | 전역적 공간 관계 | ✅ |
| **Spectral** | FFT → frequency domain 처리 | 주파수 특성 직접 조작 | ✅ |
| **Statistical** | Global pooling → channel attention | 전역 통계 기반 조건화 | ✅ |
| ~~Conv 수용장~~ | ~~3×3, 5×5, 7×7 DWConv~~ | ~~국소/전역 공간 패턴~~ | ❌ (8×8에서 무의미) |

##### 구체적 설계

**Expert A — Channel-Group Expert**

```python
class ChannelGroupExpert(nn.Module):
    """채널을 G 그룹으로 분할, 그룹별 독립 변환 후 재조합.
    
    INS(직달일사), DHI(산란일사) 등 물리적으로 다른 정보가 
    채널 차원에 혼재할 때, 그룹별 분리 처리로 inter-channel 간섭 방지.
    """
    def __init__(self, dim, num_groups=4, mlp_ratio=4.0):
        super().__init__()
        group_dim = dim // num_groups
        self.num_groups = num_groups
        self.group_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(group_dim, int(group_dim * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(group_dim * mlp_ratio), group_dim),
            ) for _ in range(num_groups)
        ])
    
    def forward(self, x):  # (B, N, D)
        groups = x.chunk(self.num_groups, dim=-1)
        return torch.cat([ffn(g) for ffn, g in zip(self.group_ffns, groups)], dim=-1)
```

**Expert B — Token-Interaction Expert (gMLP 변형)**

```python
class TokenInteractionExpert(nn.Module):
    """토큰 간 직접 상호작용: N×N learned mixing.
    
    Self-Attention은 Q·K 유사도 기반이지만, 이 expert는
    position-dependent linear mixing으로 고정된 공간 패턴(위도별 일사량 분포 등)을 학습.
    8×8 = 64 tokens이므로 64×64 mixing matrix는 충분히 작음.
    """
    def __init__(self, dim, num_tokens=64, mlp_ratio=4.0):
        super().__init__()
        self.spatial_proj = nn.Linear(num_tokens, num_tokens)  # N×N mixing
        nn.init.zeros_(self.spatial_proj.weight)  # zero-init
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
    
    def forward(self, x):  # (B, N, D)
        # (B, N, D) → transpose → (B, D, N) → spatial mix → (B, D, N) → transpose
        mixed = self.spatial_proj(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x + mixed)
        return self.ffn(x)
```

**Expert C — Spectral Expert (FFT)**

```python
class SpectralExpert(nn.Module):
    """주파수 도메인에서 처리: 2D FFT → learnable filter → IFFT.
    
    8×8 grid의 2D FFT는 8×8 = 64 주파수 bin.
    Conv와 달리 전체 주파수 스펙트럼에 대한 독립적 필터링 가능.
    구름의 주기적 패턴, 위성 스캔라인 아티팩트 등 주파수 특성에 직접 작용.
    """
    def __init__(self, dim, h=8, w=8):
        super().__init__()
        # learnable complex-valued filter: (D, H, W//2+1)
        self.freq_filter = nn.Parameter(torch.ones(dim, h, w // 2 + 1))
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
        )
    
    def forward(self, x, h_shape=8, w_shape=8):  # (B, N, D)
        B, N, D = x.shape
        # spatial → 2D FFT
        x_2d = x.transpose(1, 2).reshape(B, D, h_shape, w_shape)
        x_freq = torch.fft.rfft2(x_2d, norm='ortho')
        # learnable frequency-domain filtering
        x_freq = x_freq * self.freq_filter.unsqueeze(0)
        # IFFT back
        x_filtered = torch.fft.irfft2(x_freq, s=(h_shape, w_shape), norm='ortho')
        x_out = x_filtered.flatten(2).transpose(1, 2)  # (B, N, D)
        return self.ffn(self.norm(x + x_out))
```

**Expert D — Statistical Conditioning Expert (SE-Net 변형)**

```python
class StatCondExpert(nn.Module):
    """전역 통계로 채널별 가중치 학습 (Squeeze-and-Excitation).
    
    64 토큰의 global average → channel attention → channel reweighting.
    '현재 이미지 전체가 밝은가/어두운가/구름이 많은가'에 따라
    채널별 중요도를 동적으로 조절.
    """
    def __init__(self, dim, reduction=8, mlp_ratio=4.0):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
    
    def forward(self, x):  # (B, N, D)
        # Global average pooling → channel attention
        stats = x.mean(dim=1)  # (B, D)
        gate = self.se(stats).unsqueeze(1)  # (B, 1, D)
        return self.ffn(x * gate)
```

##### P=16에서의 의미

| Expert | 차별화 축 | 8×8에서 유효 | 직교성 |
|--------|----------|:---:|:---:|
| Channel-Group | 채널 분리 처리 | ✅ | 다른 물리량 독립 처리 |
| Token-Interaction | 고정 공간 패턴 | ✅ | Attention과 보완적 |
| Spectral | 주파수 도메인 | ✅ | 공간/주파수 이중 관점 |
| Stat-Cond | 전역 통계 조건화 | ✅ | Content-adaptive |

**핵심**: "수용장 크기"가 아닌 **"연산 유형"**으로 분화하면 P=16의 8×8에서도 각 expert가 진정으로 다른 일을 한다.

##### 구현 비용: **~1~2일** (4개 expert 클래스 + router 통합)

---

#### 5-D. Adaptive Rank Expert

##### 현재 문제

```python
# 현재: 모든 K=8 expert가 동일한 rank=16
self.down = nn.Parameter(torch.empty(K, D, r))   # K×D×r
self.up   = nn.Parameter(torch.zeros(K, r, D))    # K×r×D
```

모든 expert가 동일한 표현 용량(rank=16, D=512 중 3.1%) → **복잡한 패턴도, 단순한 패턴도 동일한 capacity**.

이는 비효율적이다:
- Structure expert: 전역적 구조 복원에는 높은 rank 필요 (다양한 방향/스케일)
- Detail expert: 국소 보정은 낮은 rank로 충분 (방향 1~2개)

##### 설계

```python
# Adaptive Rank: expert별 다른 rank
# 총 파라미터 예산: K × D × r_avg × 2 (고정)
# 예산 내에서 rank 재분배

# 예시: K=4, r_avg=16, budget=4×512×16×2 = 65,536
rank_schedule = {
    'structure': 32,   # 복잡한 전역 패턴 → 높은 capacity
    'direction': 24,   # 방향성 패턴 → 중간-높은 capacity  
    'detail':    12,   # 국소 보정 → 중간 capacity
    'identity':   4,   # 배경/무변환 → 최소 capacity
}
# 총: (32+24+12+4) × 512 × 2 = 73,728 ≈ 기존 (16×4×512×2 = 65,536)
```

##### 수학적 근거

LoRA의 **rank = 표현 가능한 변형 공간의 차원**:

```
rank  = r  →  expert가 표현 가능한 변형:  {W·x : W ∈ R^{D×D}, rank(W) ≤ r}
                                          = Grassmannian G(r, D) 위의 r차원 부분공간
```

- r=32: 512차원 중 32차원 부분공간 (6.25%) → 다양한 변형 가능
- r=4: 512차원 중 4차원 부분공간 (0.78%) → 극도로 제한된 변형

**복잡한 패턴에 높은 rank를 할당하는 것은 표현력의 "정보 이론적 최적 배분"**이다.

##### 구현

```python
class AdaptiveRankExpert(nn.Module):
    def __init__(self, dim, rank_schedule, lb_weight=0.0):
        """rank_schedule: List[int], expert별 rank 리스트."""
        super().__init__()
        self.num_experts = len(rank_schedule)
        
        # Expert별 다른 rank의 down/up
        self.downs = nn.ParameterList([
            nn.Parameter(torch.empty(dim, r)) for r in rank_schedule
        ])
        self.ups = nn.ParameterList([
            nn.Parameter(torch.zeros(r, dim)) for r in rank_schedule
        ])
        for down in self.downs:
            nn.init.kaiming_uniform_(down, a=math.sqrt(5))
        
        self.router = nn.Linear(dim, self.num_experts)
    
    def _compute_delta(self, x):
        weights = F.softmax(self.router(x), dim=-1)
        delta = torch.zeros_like(x)
        for k in range(self.num_experts):
            mid = x @ self.downs[k]      # (B, N, r_k)
            e_k = mid @ self.ups[k]       # (B, N, D)
            delta = delta + weights[..., k:k+1] * e_k
        return delta
```

##### t-conditioned routing과의 조합

5-A와 5-D는 **직교적으로 조합 가능**:

```python
# t-aware routing + adaptive rank
class TCondAdaptiveExpert(nn.Module):
    def __init__(self, dim, rank_schedule, t_emb_dim):
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, dim))
        self.router = nn.Linear(dim, len(rank_schedule))
        # ... adaptive rank downs/ups ...
    
    def _compute_delta(self, x, t_emb=None):
        router_in = x + self.t_proj(t_emb).unsqueeze(1) if t_emb else x
        weights = F.softmax(self.router(router_in), dim=-1)
        # adaptive rank loop ...
```

이렇게 하면:
- **t≈999**: structure expert (rank=32)에 높은 weight → 전역 구조 복원에 최대 capacity 사용
- **t≈0**: detail expert (rank=12)에 높은 weight → 적은 capacity로 미세 보정

##### 구현 비용: **~0.5일** (ParameterList 변환 + rank config 추가)

---

## 전략 비교 요약 (학술적 기여 관점)

| | Multi-Scale | Cross-Scale | t-conditioned Routing | P16 Expert 재설계 | Adaptive Rank |
|---|:---:|:---:|:---:|:---:|:---:|
| **학술적 기여** | ✅✅ 아키텍처 혁신 | ✅ 어텐션 혁신 | ✅ Diffusion+MoE 결합 | ✅ 연산유형 분화 | ✅ 정보론적 최적배분 |
| **PSNR 개선 기대** | **높음** | 중간 | 중간 | 중간~높음 | 중간 |
| **구현 복잡도** | 높음 (2~3일) | **없음** (구현됨) | **낮음** (0.5일) | 중간 (1~2일) | **낮음** (0.5일) |
| **조합 가능** | 독립 | 독립 | ⟷ 5-D와 직교 조합 | 독립 | ⟷ 5-A와 직교 조합 |
| **논문 스토리** | Hierarchical ViT | Asymmetric attention | Noise-adaptive routing | Operation-type MoE | Capacity-optimal MoE |

---

## 대기 중인 실험

- [ ] Both t-gate 3-seed 검증 (P0, [명령어](file:///home/student1/.gemini/antigravity-ide/brain/921a79fc-46b9-442a-9092-725b75f38bef/experiment_status.md))
