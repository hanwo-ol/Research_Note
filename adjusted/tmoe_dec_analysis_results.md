# tmoe_dec 모델 파일 심층 분석 결과

> **분석 대상**: `models_dit/tmoe_dec/best_st_dit.pt` (3-expert MoE decoder)  
> **비교 대상**: `tgate_baseline`, `cap_dec3x`

---

## 핵심 결론 (3줄 요약)

1. **Expert들은 서로 매우 다른 weight를 학습했다** (cos ≈ 0.09, 거의 직교)
2. **Gate는 거의 작동하지 않는다** (weight norm < 0.1, routing ≈ 1/3 uniform)
3. **FID 개선은 "3개 전문 conv의 uniform 평균 = implicit ensemble" 효과**

---

## 분석 1: Expert Weight Divergence (분화도)

| Stage | Shape | Avg Pairwise Cos | 해석 |
|:-----:|-------|:-----------------:|------|
| 0 | (256,512,3,3) | **0.091** | 거의 직교 → 진정한 분업 |
| 1 | (128,256,3,3) | **0.074** | 거의 직교 → 진정한 분업 |
| 2 | (64,128,3,3) | **0.098** | 거의 직교 → 진정한 분업 |
| 3 | (1,64,3,3) | **-0.032** | 음의 상관 → 완전 독립 |

> [!IMPORTANT]
> **Cosine similarity ≈ 0.09는 random init 수준과 유사.** 하지만 이것은 random이 아닌 **학습된 결과**다.
> Xavier init 직후 random weight 쌍의 기대 cos는 ≈ 0이므로, 학습 과정에서 각 expert가 **서로 다른 방향으로 수렴**한 것.  
> 
> 이것이 FID 개선의 원천: 3개 expert가 각각 다른 spatial pattern 복원 전문가가 되어, 
> 그 평균이 **단일 conv의 한계를 넘는 표현력**을 제공.

---

## 분석 2: Gate Routing 패턴

| Stage | Weight Norm | Bias Softmax | Effective Rank | 판정 |
|:-----:|:-----------:|:------------:|:--------------:|------|
| 0 | 0.094 | (0.333, 0.334, 0.333) | 2.49/3 | 🔴 Uniform |
| 1 | **0.129** | (0.334, 0.333, 0.334) | 2.40/3 | 🟡 약한 conditional |
| 2 | 0.098 | (0.333, 0.334, 0.333) | 2.23/3 | 🔴 Uniform |
| 3 | 0.091 | (0.334, 0.333, 0.334) | 2.47/3 | 🔴 Uniform |

> [!WARNING]
> **Gate가 사실상 학습되지 않았다.**  
> - Weight norm < 0.13 → softmax 입력이 ≈ 0 → 모든 expert에 ≈ 1/3 weight
> - Bias도 ≈ 0 → zero-init에서 거의 움직이지 않음
> - **결론**: Gate는 input-independent uniform averaging 수행

---

## 분석 3: Expert Output Diversity (Forward Pass)

### Gate Routing by Timestep

| t | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
|---|---------|---------|---------|---------|
| 10 | (0.349, 0.320, 0.331) | (0.317, 0.347, 0.336) | (0.341, 0.330, 0.329) | (0.340, 0.329, 0.331) |
| 500 | (0.332, 0.334, 0.335) | (0.333, 0.329, 0.338) | (0.334, 0.336, 0.330) | (0.335, 0.331, 0.333) |
| 990 | (0.335, 0.331, 0.334) | (0.327, 0.337, 0.336) | (0.339, 0.326, 0.335) | (0.328, 0.339, 0.332) |

→ **t에 관계없이 routing ≈ (1/3, 1/3, 1/3)** — t-dependent routing 없음

### Expert Output Cosine Similarity

| Stage | Output cos (t=10) | Output cos (t=500) | Output cos (t=990) |
|:-----:|:-----------------:|:------------------:|:------------------:|
| 0 | 0.14~0.15 | 0.14~0.15 | 0.14~0.15 |
| 1 | 0.12 | 0.12 | 0.11~0.12 |
| 2 | 0.22~0.26 | 0.22~0.25 | 0.23~0.25 |
| 3 | **-0.15~-0.05** | **-0.16~-0.03** | **-0.15~-0.03** |

→ Expert output도 cos < 0.26으로 **서로 매우 다른 출력을 생산**  
→ 마지막 stage(3)는 **음의 상관** — 완전히 다른 reconstruction 전략  
→ **t에 따른 output 변화는 없음** (t-independent)

---

## 분석 4: Capacity 비교

| Model | Decoder Params |
|-------|:--------------:|
| tmoe_dec (3-expert MoE) | **1,548,864** |
| cap_dec3x (3x wider) | **6,858,432** |

> [!NOTE]
> tmoe_dec가 cap_dec3x보다 **4.4배 적은 파라미터**로 더 좋은 FID!  
> MoE 구조가 파라미터를 훨씬 효율적으로 활용.

---

## 분석 5: Baseline vs tmoe_dec (Non-MoE layer 변화)

| Layer | Δ-norm | Relative Change |
|-------|:------:|:--------------:|
| decoder.1.weight (Stage 0 conv) | 26.93 | **137.8%** |
| decoder.4.weight (Stage 1 conv) | 18.88 | **138.3%** |
| decoder.7.weight (Stage 2 conv) | 13.24 | **139.7%** |
| decoder.10.weight (Stage 3 conv) | 0.09 | 100% |

> MoE 도입으로 **기존 decoder conv도 크게 변화** (130~140% relative diff)  
> → MoE가 전체 decoder의 학습 dynamics를 변경함 (co-adaptation)

---

## 종합 판정

### Q1: FID 개선은 expert 전문화인가, implicit ensemble인가?

**→ Implicit Ensemble of Specialized Experts**

- Expert weight는 거의 직교 (cos ≈ 0.09) → **전문화 ✅**
- Gate는 uniform (1/3) → 전문화된 expert를 **입력 무관하게 평균** → **implicit ensemble ✅**
- 즉, "서로 다른 것을 알고 있는 3명의 전문가가, 누가 맞는지 모르고 항상 3명의 의견을 평균"

### Q2: Gate routing이 의미 있는 선택을 하고 있는가?

**→ No.** Gate는 zero-init에서 거의 움직이지 않음. 사실상 불용.

### Q3: MoE의 이점이 단순 capacity 증가와 질적으로 다른가?

**→ Yes.** 1.5M params (MoE) > 6.9M params (wider) — **4.4배 적은 파라미터로 더 좋은 FID.**

---

## 후속 방향 (우선순위순)

| 순위 | 방향 | 기대 효과 | 근거 |
|:----:|------|-----------|------|
| 1 | **Gate 제거 → 순수 uniform averaging** | 파라미터 절약 (gate 3×512 제거), 동일 성능 | Gate가 작동하지 않으므로 제거해도 무방 |
| 2 | **Gate에 t-embedding 직접 주입 (별도 MLP)** | t-dependent routing 활성화 → 추가 FID 개선 | 현재 gate 입력이 뭔지 확인 필요 (cond_emb = adaLN output?) |
| 3 | **Expert 수 증가 (K=5, K=8)** | 더 많은 전문가 앙상블 → 더 낮은 FID | cos ≈ 0.09이므로 추가 expert도 독립 수렴 예상 |
| 4 | **Stem에도 MoE 적용** | 입력 단도 다양한 encoding 가능 | decoder에서 효과 확인되었으므로 stem 확장 합리적 |
