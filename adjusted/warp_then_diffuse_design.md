# Warp-then-Diffuse 설계도

**ST-DiT2 Deep Series — Motion-Aware Prediction Paradigm**  
Phase 설계 문서 | 2026-05 (rev.2 — 0084 교훈 반영)

---

## 0. 0084 OF Stem 실패 교훈 (필독)

> [!CAUTION]
> 이 문서는 0084 Optical Flow Stem 실패를 **출발점**으로 작성되었다.  
> 하기 실패 원인을 구현 전 반드시 숙지할 것.

| 0084 OF Stem 실패 원인 | 본 설계의 대응 |
|---|---|
| OF를 Stem(입력단)에서 학습 → trunk gradient 간섭 | FlowHead는 **마지막 블록 출력에 `.detach()`** 후 입력 (§3.1) |
| Flow bound 없음 → 학습 후반 flow 폭주 | `tanh × 32px` soft cap 강제 (§2.3, §7) |
| Flow + residual이 단일 head → 역할 혼재 | FlowHead(warp) / DiT(residual) **완전 분리** (§3.3) |
| 다중 paradigm 동시 학습 → gradient 분산 | **1차: FlowHead 단독** + DynamicDecoder OFF (§6) |

**vs 기존 `USE_OPTICAL_FLOW` 구현의 차이**:
- 기존: `pred = warp(flow) + α·residual` — flow/residual이 같은 forward path
- 신규: `warp_prior = warp(FlowHead(x.detach()))`, `pred = warp_prior + DiT_sample()·σ` — **diffusion이 warp_error만 담당**

---

## 1. 동기 및 목표

### 현재 챔피언 (tempgradSQ + DynamicDecoder)의 한계

```
context frames → STDiT → diffusion residual → + last_ctx → prediction
```

- Diffusion이 **위치 정렬 없이** 전체 잔차를 학습
- 고모션 구름 이동 시: 이동 전/후 위치를 동시에 처리 → **이중 모드 분포** → 평균화 = blur
- tempgradSQ는 방향 신호만 주입; **실제 warp 수행 없음**

### Warp-then-Diffuse의 핵심 아이디어

```
context frames → FlowHead(x.detach()) → flow field → warp(last_ctx) = aligned prior
                                                               ↓
                          aligned prior + diffusion(warp_error) → final prediction
```

Diffusion이 다루는 것: **이동 후 warp 오차 (위치 정렬된 잔차)**  
→ 이중 모드 문제 해소 → 구름 경계선 선명도 향상 기대

---

## 2. 수학적 정식화

### 2.1 심볼 정의

| 심볼 | 설명 |
|---|---|
| $I_{t-1}$ | last context frame (persistence baseline) |
| $I_t$ | target frame (GT) |
| $\hat{F}$ | FlowHead 예측 flow field: $(B, 2, H, W)$, pixel units |
| $\hat{I}_{warp}$ | warp된 prior: $\mathcal{W}(I_{t-1}, \hat{F})$ |
| $r_{warp}$ | warp residual: $I_t - \hat{I}_{warp}$ |
| $\hat{r}$ | diffusion 예측 residual |

### 2.2 학습 목표

**학습 시:**
```
x_spatial = transformer_output.detach()              # trunk gradient 차단
flow      = FlowHead(x_spatial)                      # pixel displacement (B,2,H,W)
warp_prior = warp_frame(last_ctx, flow)              # aligned prior
target = (GT - warp_prior) / σ_resid                 # warp residual 정규화
loss = L_diff + flow_lambda * L_photo
     + lambda_smooth * |Sobel(flow)|
     + lambda_mag   * |flow|
```

**추론 시:**
```
x_spatial  = transformer_output.detach()
flow       = FlowHead(x_spatial)
warp_prior = warp_frame(last_ctx, flow)   # step 1: motion compensation
residual   = DiT_sample() * σ_resid       # step 2: refine warp error
prediction = warp_prior + residual         # step 3: compose
```

### 2.3 Flow Field 설계

```
F: (B, 2, H, W) — pixel units (dx, dy)
W(I, F)[y, x] = I[y + F[1,y,x], x + F[0,y,x]]   (bilinear grid_sample)
```

> [!IMPORTANT]
> **flow_cap = `tanh(raw_F) × 32.0` 권장 (0084 폭주 방지)**  
> zero-init만으로 학습 초기 안정성 확보되나, **학습 후반 bound 없으면 폭주 위험**.  
> default: `flow_cap = 32.0` (1 patch = 32px). CLI `--flow_cap 0`으로 무제한 가능하나 비권장.

```python
# FlowHead 마지막 출력에 적용
if self.flow_cap is not None and self.flow_cap > 0:
    flow = torch.tanh(raw_flow) * self.flow_cap  # (B, 2, H, W), pixel units
else:
    flow = raw_flow  # 무제한 (0084 위험, 비권장)
```

---

## 3. 아키텍처 설계

### 3.1 FlowHead 모듈

```python
class FlowHead(nn.Module):
    """Spatial tokens → pixel-level flow field.

    DynamicHead와 동일한 계층적 Bilinear-upsample 구조.
    out_channels=2 (dx, dy in pixel units). 마지막 Conv zero-init.

    ★ 입력: x_spatial = transformer_output.detach()
      → trunk gradient 완전 차단 (0084 핵심 교훈).

    차이점:
    - DynamicHead: velocity/acceleration → SPADE condition (픽셀 예측 보조)
    - FlowHead:    pixel displacement    → warp_frame 입력 (motion compensation)

    forward 입력:  (B, hidden_size, grid_size, grid_size)  ← reshape 후 전달
    forward 출력:  (B, 2, H_full, W_full)                  ← tanh * flow_cap 적용
    """
    def __init__(self, hidden_size: int, patch_size: int,
                 grid_size: int, flow_cap: float = 32.0):
 # 채널 스케줄: hidden_size → h//4 → ... → 2
 # flow_cap: tanh soft cap (pixel units). 0 = 무제한.
```

**채널 스케줄**: `hidden_size → h//4 → ... → 2`  
**zero-init**: 마지막 Conv → 학습 초기 flow=0 → warp=identity → persistence baseline 동등  
**flow_cap**: `tanh(·) × flow_cap` — 기본 32px (0084 폭주 방지)

### 3.2 Warp 함수 (정정된 2D meshgrid 버전)

```python
def flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    """flow (pixel units, B,2,H,W) → normalized sampling grid (B,H,W,2) in [-1,1].

    ★ 이전 버전(1D arange)은 grid_sample 에러 유발 — 2D meshgrid 필수.
    """
    B, _, H, W = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=flow.device, dtype=flow.dtype),
        torch.arange(W, device=flow.device, dtype=flow.dtype),
        indexing='ij',
    )
    src_x = xx.unsqueeze(0).expand(B, -1, -1) + flow[:, 0]  # (B, H, W)
    src_y = yy.unsqueeze(0).expand(B, -1, -1) + flow[:, 1]  # (B, H, W)
    grid_x = 2.0 * src_x / max(W - 1, 1) - 1.0
    grid_y = 2.0 * src_y / max(H - 1, 1) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)


def warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Bilinear warp. frame: (B,C,H,W), flow: (B,2,H,W) pixel units → (B,C,H,W).

    NOTE: gaussian.py의 warp_persistence를 이 함수로 통합 예정.
    """
    grid = flow_to_grid(flow)
    return F.grid_sample(frame, grid, mode='bilinear',
                         align_corners=True, padding_mode='border')
```

### 3.3 STDiT 통합 위치 (`.detach()` 필수)

```python
# st_dit.py forward() — transformer blocks 통과 후

# .detach()로 trunk gradient 완전 차단 (0084 핵심 교훈)
x_spatial = x.transpose(1, 2).reshape(B, hidden_size, grid_size, grid_size).detach()
flow_field = self.flow_head(x_spatial)           # (B, 2, H_full, W_full)
warp_prior = warp_frame(last_ctx_pixel, flow_field)  # (B, 1, H, W)
# → train.py에서: target = (GT - warp_prior) / σ_resid
# → inference.py에서: pred = DiT_output * σ_resid + warp_prior
```

**통합 플래그**: `USE_WARP_DIFFUSE`

| 플래그 | 역할 | Warp-then-Diffuse 시 |
|---|---|---|
| `USE_PERSISTENCE_RESIDUAL` | residual 학습 | ✅ 유지 (baseline → warp_prior로 대체) |
| `USE_DYNAMIC_DECODER` | velocity/accel SPADE | ⚠️ **1차: OFF** (gradient 분산 방지) |
| `USE_TEMPORAL_GRADIENT_SQUARE` | context에 delta² 주입 | ✅ 유지 (입력단, 독립) |
| `USE_OPTICAL_FLOW` | 기존 OF 구현 | ❌ mutex (같은 목적) |

---

## 4. 학습 전략

### 4.1 2단계 Curriculum (명확화)

**Stage A — warm-up (epoch 1~20)**:
- FlowHead last Conv zero-init → flow=0 → warp=identity → persistence residual로 시작
- `flow_lambda = 0.0` (photometric loss OFF, diffusion loss만)
- FlowHead gradient: diffusion loss backprop 자연 유입

**Stage B — ramp + plateau (epoch 21~100)**:
```python
if epoch <= 20:
    flow_lambda = 0.0                          # Stage A: warm-up
elif epoch <= 50:
    flow_lambda = (epoch - 20) / 30 * 0.1     # linear ramp 0.0 → 0.1
else:
    flow_lambda = 0.1                          # Stage B plateau
```

### 4.2 손실 함수 전체 정식화

```python
# warp supervision (직접 flow 학습)
L_photo  = F.mse_loss(warp_prior, target_pixel)           # 정렬 품질

# diffusion residual (핵심)
L_diff   = charbonnier(DiT_pred - (GT - warp_prior) / σ)  # warp_error 학습

# flow regularization (0084 stability guard)
grad_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]
grad_y = flow[:, 1:] - flow[:, :-1]
L_smooth = (grad_x.abs().mean() + grad_y.abs().mean())    # field 변동성 억제
L_mag    = flow.abs().mean()                               # clear sky → flow=0 회귀

L_total = L_diff \
        + flow_lambda    * L_photo   \  # default ramp (§4.1)
        + lambda_smooth  * L_smooth  \  # default 0.01
        + lambda_mag     * L_mag        # default 0.001
```

> [!NOTE]
> L_smooth / L_mag 은 시작 시 default로 ON (안전망).  
> `--lambda_smooth 0 --lambda_mag 0` 으로 ablation 가능.

### 4.3 Dynamic Decoder와의 통합 순서

> [!IMPORTANT]
> **§10.5 챔피언 chain 교훈**: "점진적 ft은 suboptimal — 단변수 isolate 우선"

```
1차 실험: USE_WARP_DIFFUSE + USE_DYNAMIC_DECODER=OFF
          → FlowHead 단독 효과 측정

2차 실험: 결과 확인 후 USE_DYNAMIC_DECODER=ON 결합
          → warp_prior ← flow_head(x.detach())
             SPADE cond ← dynamic_head(x.detach())  (동일 detach 패턴)
```

두 head 모두 같은 transformer 출력 토큰 → **동일한 `.detach()` 패턴** 필수.

---

## 5. 구현 파일 목록

```
[NEW]  stdit/models/flow_head.py      — FlowHead(hidden_size, patch_size, grid_size, flow_cap)
[NEW]  stdit/diffusion/warp.py        — flow_to_grid, warp_frame  (gaussian.py 통합)
[MOD]  stdit/models/st_dit.py         — FlowHead init + forward: x.detach() → flow → warp_prior
[MOD]  stdit/engine/train.py          — target = (GT - warp_prior) / σ + L_photo + L_smooth + L_mag
[MOD]  stdit/engine/inference.py      — pred = DiT_output * σ + warp_prior
[MOD]  stdit/engine/periodic_eval.py  — 동일
[MOD]  main.py / test_only.py         — --use_warp_diffuse, --flow_lambda, --flow_cap
                                        --lambda_smooth, --lambda_mag
[MOD]  stdit/config.py                — 매핑 + mutex validator (OF / WarpDiffuse)
```

---

## 6. 단계적 실험 로드맵

```
현재 위치
    │
    ├─ [A] per-patch σ 결과 확인 (진행 중 — perpatch005)
    │
    ├─ [B1] Warp-then-Diffuse 단독 (DynamicDecoder OFF)
    │         • flow_lambda=0, lambda_smooth=0.01, lambda_mag=0.001
    │         • 30 epoch → 진단: warp_MAE vs persistence_MAE
    │         • 로그: flow magnitude 통계, warp quality 시각화
    │
    ├─ [B2] flow_lambda ramp 활성화 (epoch 21~)
    │         • L_photo curriculum 가동
    │         • 100 epoch full run
    │
    ├─ [C] tempgradSQ + Warp-then-Diffuse 결합
    │         • context embedder delta² 주입 + FlowHead 동시
    │
    ├─ [D] DynamicDecoder + FlowHead 결합 ablation
    │         • 둘 다 .detach() 패턴
    │
    └─ [E] 최종: per-patch σ + tempgradSQ + Warp-then-Diffuse + DynamicDecoder
               → 앙상블 후보
```

---

## 7. 설계 결정 사항 (확정 + 미결)

### 확정

| 항목 | 결정 |
|---|---|
| `.detach()` | ✅ 필수 — trunk gradient 차단 (0084 교훈) |
| flow_cap | ✅ `tanh × 32.0` (1 patch) — 무제한 비권장 |
| Warp 함수 | ✅ 2D meshgrid `flow_to_grid` → `grid_sample` |
| DynamicDecoder 1차 | ✅ OFF — isolate 우선 |
| L_smooth / L_mag | ✅ default ON (0.01 / 0.001) |
| curriculum | ✅ Stage A (0~20): lambda=0, Stage B (21~50): ramp, (51+): 0.1 plateau |

### 미결

> [!IMPORTANT]
> **warp_prior를 context embedder에도 주입할지?**
> ```
> Option A: warp_prior는 target 조정에만 사용 — 단순, 1차 권장
> Option B: warp_prior를 추가 context frame으로 주입
>           → 모델이 warp 품질도 파악 가능, 단 구조 변경 필요
> ```
> → **1차: Option A** (isolate). B2 이후 ablation.
