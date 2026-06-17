# Seq2Seq Open Questions — 전문가 검토

## Q1. Per-frame σ (Residual 정규화)

### 문제

현재 single-lead:
```python
residual = target - last_ctx           # (B, 1, H, W)
target_normalized = residual / σ       # ≈ N(0, 1)
```

Multi-lead에서 4장의 target은 last_ctx로부터 **서로 다른 시간 거리**:
```
Frame 0 (t+30):  σ₀ ≈ 0.37  (30분 거리 → 작은 분산)
Frame 1 (t+60):  σ₁ ≈ ?     (60분 거리 → 더 큰 분산)
Frame 2 (t+90):  σ₂ ≈ ?     (90분 거리 → 더 큰 분산)
Frame 3 (t+120): σ₃ ≈ ?     (120분 거리 → 가장 큰 분산)
```

**단일 σ를 쓰면**: 30분 residual은 over-normalized (너무 작아짐), 120분은 under-normalized → diffusion 학습 불안정.

### 검토

| Option | 방법 | 장점 | 단점 |
|:---:|:---|:---|:---|
| **(a)** | 전체 공통 σ | 단순 | scale 불균형 → 30분 학습 불리 |
| **(b)** | frame별 독립 σ 측정 | 정확 | 측정 4회 필요 (실은 1 pass로 가능) |
| **(c)** | σᵢ = σ₀ × √(i+1) | 물리적 근사 | 실제 분포와 괴리 가능 |

### 추천: (b-lite) — 기존 `residual_auto_stats`를 4-frame으로 확장

> [!TIP]
> 현재 `residual_auto_stats`는 이미 1024 샘플을 순회하며 σ를 측정합니다.
> 이 루프에서 **frame별 σ를 동시에 측정**하면 추가 비용 ≈ 0.

```python
# 현재 (train.py L246-247):
sigma_resid = float(res_stats["sigma"])  # 단일 σ

# 변경:
sigma_resid = [float(res_stats[f"sigma_{i}"]) for i in range(T)]
# 학습 시작 시 1회 측정, 이후 고정

# 적용:
for i in range(T):
    target[:, i] = (target[:, i] - last_ctx[:, 0]) / sigma_resid[i]
```

**예상 σ 패턴** (2021년 data 기준 추정):
```
σ₀(30min)  ≈ 0.37  (kine_v2에서 측정됨)
σ₁(60min)  ≈ 0.52  (≈ σ₀ × √2 ≈ 0.37 × 1.41)
σ₂(90min)  ≈ 0.64  (≈ σ₀ × √3)
σ₃(120min) ≈ 0.74  (≈ σ₀ × √4)
```

실제 측정 후 √t 근사와의 괴리를 확인하면 domain insight도 얻을 수 있음.

---

## Q2. Loss 가중치

### 문제

4장의 target에 대한 loss를 어떻게 합산?

### 검토

| Option | 가중치 | 근거 |
|:---:|:---|:---|
| **(a)** | 균등 [1/4, 1/4, 1/4, 1/4] | 단순, per-frame σ 정규화 후 scale 동일 |
| **(b)** | 근미래 강조 [0.4, 0.3, 0.2, 0.1] | 실용적 가치 기반 |
| **(c)** | 학습 가능 (uncertainty) | 자동 조절, 파라미터 추가 |

### 추천: (a) 균등

> [!IMPORTANT]
> **Per-frame σ 정규화 (Q1)가 올바르게 적용되면, 각 frame의 loss scale은 이미 ≈ 동일합니다.**
>
> `target_normalized[i] ≈ N(0, 1)` → `MSE(pred[i], noise[i])` 의 기댓값이 frame 무관하게 ≈ 1.
>
> 따라서 추가적인 loss weighting은 불필요하며, 오히려 하이퍼파라미터를 늘릴 뿐.

**단, 추후 ablation 필요 시** (b)를 시도할 수 있도록 `--seq2seq_loss_weights` CLI 옵션을 기본값 "uniform"으로 추가.

근미래 가중이 필요한 경우는:
- 30분 예측 품질이 120분보다 **현저히 나빠진 경우** (multi-frame 학습이 30분을 희생하는 경우)
- 이때만 (b)로 전환 의미 있음

---

## Q3. Kinematic Loss — Inter-frame Chain Velocity (Option b)

### 사용자 선택: (b) 연쇄 inter-frame consistency

### 수식 정리

**개념**: 예측된 프레임 간의 **30분 속도**가 과거 관측 속도와 유사해야 한다.

```
v_past = context[-1] - context[-2]    (관측된 30분 속도)

v_pred[0] = x̂₀(t+30) - context[-1]   (context → 첫 예측)
v_pred[1] = x̂₀(t+60) - x̂₀(t+30)    (예측 간 속도)
v_pred[2] = x̂₀(t+90) - x̂₀(t+60)
v_pred[3] = x̂₀(t+120) - x̂₀(t+90)
```

**Loss**:
```
L_kine = Σᵢ wᵢ · MSE(v_pred[i], v_past)
```

각 v_pred[i]가 v_past에 가까워야 함 = **등속 직선 운동 귀납 편향 (inertia prior)**.

### 아키텍처 검토

> [!NOTE]
> **주의점**: `x̂₀(t+k)`는 diffusion의 x0 prediction이므로, noise level `t`에 따라 품질이 다름.
> - 높은 `t` (많은 noise): x0 prediction이 불정확 → v_pred 노이즈 큼
> - 낮은 `t` (적은 noise): x0 prediction이 정확 → v_pred 신뢰 가능
>
> 따라서 기존과 동일하게 **SNR gating**을 적용해야 함.

### 구현 설계

```python
# train.py — multi-lead kinematic loss
if use_kinematic_loss and num_pred_frames > 1:
    v_past = (context[:, -1] - context[:, -2]).detach()  # (B, C, H, W)
    
 # x0_pred_latent: (B, T*C, H, W) → per-frame split
    x0_frames = x0_pred_latent.reshape(B, T, C, H, W)  # (B, T, C, H, W)
    
    kine_loss = 0.0
    for i in range(T):
        if i == 0:
            v_pred_i = (x0_frames[:, 0] * sigma_resid[0]) - last_ctx[:, 0]
        else:
            v_pred_i = (x0_frames[:, i] * sigma_resid[i]) - (x0_frames[:, i-1] * sigma_resid[i-1])
        kine_loss += F.mse_loss(v_pred_i, v_past)
    
    kine_loss /= T  # per-frame 평균
    loss += kine_w * kine_loss
```

### 물리적 의미

이 정규화는 모델에 다음을 강제합니다:

```
x̂(t+30) ≈ ctx[-1] + v_past              (등속 외삽)
x̂(t+60) ≈ x̂(t+30) + v_past ≈ ctx[-1] + 2·v_past
x̂(t+90) ≈ x̂(t+60) + v_past ≈ ctx[-1] + 3·v_past
x̂(t+120) ≈ x̂(t+90) + v_past ≈ ctx[-1] + 4·v_past
```

**이것이 단순 persistence와 다른 점**:
- Persistence: x̂(t+k) = ctx[-1] (정지)
- Chain velocity: x̂(t+k) = ctx[-1] + k·v_past (**움직임 유지**)

diffusion 모델은 이 inertia prior **위에서** 비선형 보정을 학습하므로,
변화 추세를 유지하면서도 급격한 변동에 대응할 수 있음.

> [!WARNING]
> **가중치 주의**: `kinematic_loss_weight`이 너무 크면 모든 예측이 **등속 외삽**에 수렴.
> 현재 single-lead에서 `weight=0.05`이므로, multi-lead에서도 `0.05` 유지 후 ablation.
> chain 길이가 4배이므로 실효 기여가 커질 수 있어 **0.03 정도로 시작** 추천.

---

## 최종 추천 요약

| 질문 | 추천 | 근거 |
|:---|:---:|:---|
| **Q1. Per-frame σ** | **(b-lite)** | 기존 auto_stats 루프에서 4-frame σ 동시 측정 (추가 비용 ≈ 0) |
| **Q2. Loss 가중치** | **(a) 균등** | σ 정규화 후 scale 동일 → 추가 weighting 불필요 |
| **Q3. Kinematic** | **(b) 연쇄** ✅ | inter-frame velocity consistency, weight=0.03 시작 |
