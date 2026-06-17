# t-gated GroupNorm PoC 설계 리뷰

**리뷰 대상**: [stem_v3_tgate_design.md](file:///home/student1/.gemini/antigravity-ide/brain/921a79fc-46b9-442a-9092-725b75f38bef/stem_v3_tgate_design.md)  
**리뷰 역할**: 도메인 관점 설계자 관점 구현자 관점 검토자 관점

---

## Q1: context_embedder에도 t_gate를 적용할 것인가?

### 판정: 현재 설계(아니오) 올바름

| 관점 | 분석 |
|------|------|
| **도메인 관점** | Context 프레임은 clean 과거 관측. 노이즈 레벨 `t`와 무관. t-conditioning은 의미론적으로 부적절 |
| **설계자 관점** | `x_embedder`와 `context_embedder`는 **weight 비공유** (line 548 vs 554). 독립 모듈이므로 한쪽만 변경 가능 |
| **구현자 관점** | `context_embedder`는 for-loop 내에서 프레임별 호출 (line 1477-1480). t_emb 전달은 간단하나 불필요한 연산 추가 |
| **검토자 관점** | 단일 변수 실험 원칙: t-gated의 효과를 순수 측정하려면 target side만 변경이 올바름 |

> [!TIP]
> **후속 실험 가능성**: 만약 V3가 성공하면, context_embedder에도 t_emb를 전달하여 "context 표현을 target의 noise level에 맞게 적응" 하는 실험이 가능. 하지만 이는 별도 ablation으로 분리해야 함.

---

## Q2: Residual shortcut도 함께 적용할 것인가?

### 판정: 조건부 승인 — 근거 수정 필요 + 설계 결함 1건 발견

#### 설계서 근거 오류

> "AdaGN 추가로 depth가 실질적으로 깊어지므로 안전 장치로 유지"

이 진술은 **오류**. AdaGN은 channel-wise affine transform(`h * (1+s) + b`)으로, 네트워크 depth를 증가시키지 않음. V3 stage의 computational graph는 V2와 동일한 depth를 가짐:

```
V2: x → Conv → GN → GELU → (+shortcut)     # depth 동일
V3: x → Conv → GN → ×(1+s)+b → GELU → (+shortcut)  # 같은 depth, affine만 추가
```

#### 설계 분석 관점 — Residual의 진짜 역할

Residual shortcut이 필요한 **올바른 이유**:

> [!CAUTION]
> **초기화 시점에서 GroupNorm이 활성 상태로 출발하는 문제에 대한 안전망.**
>
> zero-init 시 `scale=0, shift=0` → `h * (1+0) + 0 = h` → **GroupNorm 출력이 그대로 통과**.
> 즉, 학습 첫 step에서 V3는 `GELU(GroupNorm(Conv(x))) + shortcut(x)` — 이것은 V2의 `use_norm=True, use_residual=True` 조합과 동일.
>
> 문제: GroupNorm 단독 실험 (`stem_v2_gnorm_p16_s42`)은 **FID 62.23으로 치명적 실패**. Residual이 없으면 V3도 GNorm 실패와 동일한 초기 loss landscape에서 출발.

Residual shortcut은 gradient가 GroupNorm 경로를 bypass하여 직접 흐를 수 있게 해주므로, GroupNorm이 초기에 해로워도 학습이 진행될 수 있음.

#### 구현 검수 관점 — 핵심 설계 결함 발견

> [!WARNING]
> **CRITICAL: GroupNorm의 `affine` 파라미터 충돌**
>
> 현재 설계에서 `nn.GroupNorm(groups, c_out)` — 기본값 `affine=True`.
> 이는 GroupNorm 자체가 learnable γ, β를 가진다는 의미.
>
> AdaGN이 별도로 scale/shift를 생성하므로 **이중 affine**:
> ```
> h = GroupNorm(h)  →  h = γ_gn * h_normed + β_gn   (GroupNorm 자체 affine)
> h = h * (1 + s_ada) + b_ada                         (AdaGN affine)
> ```
>
> DiT 원전의 AdaLN은 `LayerNorm(elementwise_affine=False)` 를 사용:
> ```python
> # DiT (facebook/DiT)
> self.norm = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
> ```
> GroupNorm에도 동일 원칙 적용 필요:
> ```python
> self.norm = nn.GroupNorm(groups, c_out, affine=False)
> ```

**수정안**:
```python
# _StemStageV3 내
self.norm = nn.GroupNorm(min(norm_groups, c_out), c_out, affine=False)
# ^^^^^^^^^^^^
# affine=False → γ,β 제거. 모든 affine 제어를 AdaGN에 위임.
```

이렇게 하면:
- GroupNorm은 순수 정규화만 수행 (zero-mean, unit-variance per group)
- AdaGN의 scale/shift가 유일한 affine 파라미터
- zero-init 시: `h_normed * (1+0) + 0 = h_normed` (정규화만 적용)
- 이중 affine 제거 → 파라미터 효율 + 깔끔한 gradient flow

#### 도메인 적합성 관점 — 초기화 전략 개선 제안

`affine=False` + zero-init 조합에서, 학습 초기는 `GroupNorm_only` (정규화만, affine 없음) 상태.
이것이 GNorm 단독 실패(FID 62.23)보다 안전한 이유:

1. `stem_v2_gnorm` 실험의 GroupNorm은 `affine=True` — learnable γ,β가 noise 통계와 충돌하며 학습
2. `affine=False`는 γ=1, β=0 **고정** — noise 통계를 억지로 학습하지 않음
3. AdaGN이 t에 따라 γ,β를 **동적으로** 생성 — noise-aware normalization

→ **GNorm 실패의 근본 원인(t-agnostic learnable affine)이 구조적으로 제거됨.**

### Residual 결론

| 조건 | 판정 |
|------|------|
| `affine=False` 수정 적용 시 | Residual **선택적** (있어도 좋지만 필수는 아님) |
| `affine=True` 유지 시 | Residual **필수** (GNorm 초기화 위험 완화) |
| **추천** | `affine=False` + `use_residual=True` (안전 + 깨끗한 설계) |

---

## Q3: t_emb의 차원은?

### 판정: 현재 설계 올바름

| 관점 | 분석 |
|------|------|
| **설계자 관점** | `TimestepEmbedder(hidden_size=512)` → 출력 `(B, 512)`. 표준적 |
| **구현자 관점** | Per-stage projection: `Linear(512, 2*c_out)`. Stage별 파라미터 합산 ~985K. 전체 모델 대비 무시 가능 |

#### 학술 검토 관점 — raw t_emb vs cond 선택 검증

```python
# line 1270: t_emb = self.t_embedder(t) → (B, 512), noise level만
# line 1278: cond = global_cond_mlp(cat(t_emb, seasonal_emb)) → (B, 512), noise + season + time
```

| 선택지 | 장점 | 단점 |
|--------|------|------|
| **raw t_emb** (현재) | 순수 noise-level 적응. 단일 변수 실험 | Season 정보 누락 |
| cond | Season/time 정보도 반영 | 실험 변수 2개 (t + season). 인과 해석 어려움 |

**PoC에는 raw t_emb이 올바른 선택.** 가설 "Stem이 t를 알면 token 품질 개선" 을 순수하게 검증 가능.

---

## 추가 발견: 설계 누락 사항

### ① expose_features / dual_scale 경로 비호환

> [!WARNING]
> **`HierarchicalDeepStem.forward` 의 V3 경로가 기존 분기들과 비호환.**
>
> 현재 forward (line 267-303)에는 4개 분기가 있음:
> 1. `dual_scale` path (line 277) — `stage(x)` 호출
> 2. `expose_features` path (line 289) — `stage(x)` 호출
> 3. `_use_v2` path (line 296) — `stage(x)` 호출
> 4. Sequential path (line 301) — `self.stem(x)` 호출
>
> V3 stage의 forward signature는 `stage(x, t_emb)`로 변경됨.
> **분기 1, 2에서도 `stage(x, t_emb)`을 호출해야 하지만, 설계서에 이 처리가 누락.**

**수정 필요**: `forward(self, x, t_emb=None)` 에서 모든 분기가 V3일 때 `stage(x, t_emb)` 호출하도록 통합.

```python
def forward(self, x, t_emb=None):
    def _call_stage(stage, x):
        if self._use_v3 and t_emb is not None:
            return stage(x, t_emb)
        return stage(x)
    
    if self.dual_scale:
        feats = []
        for stage in self.stages:
            x = _call_stage(stage, x)
            feats.append(x)
        ...
```

### ② FreqMoE 경로의 x_embedder_low 호출

> [!NOTE]
> FreqMoE 모드 (line 1329-1401)에서는 `self.x_embedder_low`를 사용하며, 이것은 `HierarchicalDeepStem`이 아닌 `LinearStem/CNNStem/OverlappingCNNStem` 계열.
> **V3는 baseline 모드 (line 1430) 한정이므로 FreqMoE 경로에는 영향 없음.** 이 점은 확인 완료.

### ③ torch.compile 호환성

> [!NOTE]
> 기존 ckpt에 `_orig_mod.` prefix가 있어 torch.compile 사용 확인됨.
> `_StemStageV3`는 표준 PyTorch 모듈이므로 torch.compile 호환에 문제 없음.
> 단, **기존 ckpt 로딩 시 V3 추가 키(adagn_proj 등)는 strict=False 필요.**

---

## 최종 리뷰 요약

| # | 심각도 | 항목 | 조치 |
|---|--------|------|------|
| 1 | 🔴 CRITICAL | GroupNorm `affine=True` 이중 affine 문제 | `affine=False` 로 수정 |
| 2 | 🟡 HIGH | expose_features/dual_scale path에서 V3 stage 호출 누락 | `_call_stage` helper로 통합 |
| 3 | 🟡 HIGH | 설계 근거 오류 ("depth 깊어짐") | 올바른 근거로 교체 (GNorm 초기화 안전망) |
| 4 |  OK | Q1: context 미적용 | ✅ 올바름 |
| 5 |  OK | Q3: raw t_emb 512차원 | ✅ 올바름 |
| 6 |  INFO | FreqMoE 경로 무영향 | ✅ 확인됨 |
| 7 |  INFO | torch.compile 호환 | ✅ strict=False 필요 주의 |
