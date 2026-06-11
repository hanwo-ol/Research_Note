# STDiTBlock 내부 검수 보고서

> 대상 파일: `stdit/models/blocks.py` (237 lines)  
> 검수 시점: 2026-05-15  
> 현재 챔피언: `tempgradSQ` — MAE 0.1194 / FID 12.06

---

## 블록 구조 개요

```
cond → adaLN_modulation → 9 × hidden_size
       → (shift1/scale1/gate1, shift2/scale2/gate2, shift3/scale3/gate3)

x → norm1 → modulate → Self-Attention  → × gate1 → residual
  → norm2 → modulate → Cross-Attention → × gate2 → residual
  → norm3 → modulate → MLP (+SpatialExpert) → × gate3 → residual
```

파라미터 수 (hidden=512, mlp_ratio=4, num_heads=8):
- Self-Attn QKV: 3 × 512² = 786K
- Cross-Attn Q+KV: 512² + 2×512² = 786K  
- MLP: 512×2048 + 2048×512 = 2.1M
- adaLN: 512 × (9×512) = 2.36M
- **블록 합계 ≈ 6.0M × 4 blocks = 24M** (전체 모델의 주요 부분)

---

## 발견사항 (7건)

### 🔴 F1. adaLN gate zero-init 누락

**위치**: L83–86, L105

```python
self.adaLN_modulation = nn.Sequential(
    nn.SiLU(),
    nn.Linear(hidden_size, 9 * hidden_size, bias=True),  # ← 랜덤 init
)
shift1, scale1, gate1, shift2, scale2, gate2, shift3, scale3, gate3 = \
    self.adaLN_modulation(cond).chunk(9, dim=1)
```

**문제**: gate1/2/3 파라미터가 랜덤 init. 학습 초기 gate가 1에 가까운 큰 값을 가지면 residual 통로가 억제됨.

**원 논문 (DiT)**: gate에 해당하는 출력 파라미터를 **zero-init**하여 학습 초기에는 pure residual(`x + 0 * attn_out`)로 동작 — identity mapping 보장.

**수정 방법**:
```python
# __init__ 마지막에 추가
nn.init.zeros_(self.adaLN_modulation[-1].weight[6*hidden_size:])   # gate1,2,3의 weight
nn.init.zeros_(self.adaLN_modulation[-1].bias[6*hidden_size:])     # gate1,2,3의 bias
```

> ⚠️ 기존 checkpoint 비호환. 새 실험에서만 적용.

**개선 기대**: 학습 초기 안정성. 수렴 속도 +5~15%.

---

### 🔴 F2. MLP 활성함수: GELU → SwiGLU 미사용

**위치**: L72–77

```python
self.mlp = nn.Sequential(
    nn.Linear(hidden_size, mlp_hidden_dim),
    nn.GELU(approximate="tanh"),          # ← 구형 표준
    nn.Linear(mlp_hidden_dim, hidden_size),
)
```

**현황**: GELU는 ViT/GPT-2 시대의 표준. 2023년 이후 고성능 DiT 계열 (SD3, FLUX, DiT-XL fine-tune)은 **SwiGLU** 또는 **GeGLU** 채택.

**SwiGLU 수식**:
```
SwiGLU(x) = SiLU(W₁x) ⊙ (W₂x)
```
- 파라미터 중립을 위해 hidden_dim을 `int(mlp_ratio × H × 2/3)`으로 조정
- 동일 파라미터 수에서 표현력 ↑ (gating mechanism 추가)

**드롭인 구현**:
```python
class SwiGLUMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        # 파라미터 동등: 기존 2×in×4H → 3×in×(8/3)H (≈동등)
        gate_dim = int(hidden_dim * 2 / 3)
        self.fc1 = nn.Linear(in_dim, gate_dim * 2)  # gate + value
        self.fc2 = nn.Linear(gate_dim, in_dim)
    def forward(self, x):
        gate, val = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * val)
```

**CLI**: `--use_swiglu_mlp`

**개선 기대**: MAE -1~3% (문헌 기반). FID 중립~소폭 개선.

---

### 🟡 F3. SpatialExpert gate 공유 문제

**위치**: L229–235

```python
if self.spatial_expert is not None:
    delta = self.spatial_expert(x_mod3, num_special_tokens=num_special_tokens)
    mlp_out = mlp_out + delta

x = x + gate3.unsqueeze(1) * mlp_out   # gate3이 MLP + Expert 합산에 공동 적용
```

**문제**: Expert의 `delta`가 MLP 출력과 합산된 후 `gate3` 하나로만 스케일링됨. Expert의 기여도를 독립적으로 조절 불가.

**현재 회피**: `spatial_expert_rank=16, num=8`의 LoRA expert는 `delta` 규모가 작아 큰 영향은 없음. 하지만 expert가 커지거나 다수가 될 경우 제어 복잡도 ↑.

**개선 방향**:
```python
# Expert에 독립 gate 부여
x = x + gate3.unsqueeze(1) * mlp_out + gate3_expert.unsqueeze(1) * delta
```
→ adaLN 출력을 10×H로 확장 필요. **현재 ckpt 비호환**.

**우선도**: 현재 성능에 미치는 영향 작음. 후순위.

---

### 🟡 F4. DenseMoEFFN 경로의 `special_out` 잠재 버그

**위치**: L191–211

```python
if hasattr(self.mlp, 'num_experts') and getattr(self, 'num_moe_experts', 0) > 0:
    special = x_mod3[:, :num_special_tokens]
    spatial = x_mod3[:, num_special_tokens:]
    
    spatial_out = self.mlp(spatial, h_shape=H, w_shape=W)
    special_out = self.mlp(special)          # ← h_shape/w_shape 없이 호출
```

**문제**: `self.mlp`가 `DenseMoEFFN`인 경우 `h_shape`, `w_shape` 없이 호출. `DenseMoEFFN.forward`의 이 인자 처리 여부에 따라 **silent wrong output** 가능.

**확인 방법**:
```bash
grep -n "def forward" stdit/models/moe_blocks.py
grep -n "h_shape" stdit/models/moe_blocks.py
```

**현재 실험에서의 영향**: `USE_SPATIAL_EXPERT`는 별도 LoRA expert이고 DenseMoEFFN과 다른 경로 → 현재 실험에서 충돌 없음. 두 기능을 동시에 ON할 경우 요주의.

---

### 🟢 F5. Cross-Attention에 위치 정보 부재

**위치**: L154–176

```python
# absolute 경로
cross_out, _ = self.cross_attn(x_mod2, context, context, need_weights=False)

# rope 경로
q, k = self.rope_module(q, k, coords_target, coords_context)  # ← rope 경로에만 존재
```

**현황**: absolute pos_embed 경로(`default`)에서는 Cross-Attn에 위치 정보 없음. Q(target)와 K(context) 간 공간 대응 관계를 모델이 암묵적으로만 학습.

**개선 가능성**: Cross-Attn에 상대 위치 bias (relative position bias) 추가. 하지만:
- absolute pos_embed를 입력에 미리 더했으므로 이미 Q/K에 위치 정보 포함
- 구현 복잡도 높음
- 현재 챔피언에서 이미 잘 동작

**결론**: 낮은 우선도. 향후 검토.

---

### 🟢 F6. Self-Attention Local Inductive Bias 없음

**위치**: L117–134

현재 Self-Attention은 **전역 attention** (모든 패치 쌍). 512×512 이미지를 32-patch로 나누면 16×16=256 토큰 — 전역 범위가 합리적. 단, **locality를 명시하는 bias 없음**.

`use_local_phase_preserver` (L69–70)의 depthwise conv가 이를 부분 보완하나 현재 미사용(`default=False`).

**개선 방향**: DW-Conv 활성화 (`--use_local_phase_preserver`). 파라미터 추가: `hidden_size × 3×3` depthwise = 512×9 = 4.6K (negligible).

**실험 가치**: ablation 간단 (기존 커맨드에 플래그 1개).

---

### 🟢 F7. Dropout 전무

MLP/Attention 모두 dropout=0. batch_size=16, 42 steps/epoch 환경에서 overfitting 리스크 존재하나, 현재 val loss가 train loss와 비슷하게 수렴 → 큰 문제 없음.

---

## 우선순위 종합표

| # | 발견사항 | 개선 효과 추정 | 구현 비용 | ckpt 호환 | 권장 순서 |
|---|---|---|---|---|---|
| F2 | **SwiGLU MLP** | MAE -1~3% | 낮음 | ❌ 재학습 | **1순위** |
| F1 | **gate zero-init** | 수렴 안정성 ↑ | 매우 낮음 | ❌ 재학습 | **2순위** |
| F6 | DW-Conv (phase_preserver) | 불확실 | 매우 낮음 | ❌ 재학습 | **3순위 (ablation)** |
| F4 | MoE special_out 버그 | 잠재 오류 방지 | 낮음 | ✅ | **수시 수정** |
| F3 | Expert 독립 gate | 정교한 제어 | 중간 | ❌ | 후순위 |
| F5 | Cross-Attn 위치 bias | 불확실 | 높음 | ❌ | 연구 단계 |
| F7 | Dropout | 현재 불필요 | — | ✅ | 무시 |

---

## 권장 실험 계획

### Phase A: 즉시 실험 (현재 warp 실험과 병렬)

**SwiGLU ablation** (warp_stageA 완료 후):
```bash
# tempgradSQ 기준 + SwiGLU
--use_swiglu_mlp \
--exp_name ..._warp_stageA_swiglu
```

### Phase B: gate zero-init + SwiGLU 결합
```bash
# 새 학습, gate zero-init 적용
--use_swiglu_mlp --gate_zero_init \
--exp_name ..._swiglu_gatezeroinit
```

### Phase C: DW-Conv local bias (빠른 ablation)
```bash
--use_local_phase_preserver \
--exp_name ..._localphase
```

---

## 참고: 개선 불필요 항목 (이유 포함)

| 항목 | 이유 |
|---|---|
| attention dropout | val/train loss 갭 작음 — overfitting 없음 |
| mlp_ratio 증가 | depth=4로 이미 얕음, ratio ↑보다 depth ↑이 효과적 (별도 ablation) |
| LayerNorm → RMSNorm | 현재 `elementwise_affine=False` adaLN이 이미 최적화됨 |
| Flash Attention | `F.scaled_dot_product_attention`이 이미 Flash Attention backend 선택 가능 |

---

*본 문서는 `blocks.py` v2026-05-15 기준. 구현 변경 시 재검수 필요.*
