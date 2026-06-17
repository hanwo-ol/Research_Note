# Deep Stem V2 — 문제 본질 대비 비판적 검토

> 검토 대상: [deep_stem_improvement_plan.md](file:///home/student1/.gemini/antigravity-ide/brain/921a79fc-46b9-442a-9092-725b75f38bef/deep_stem_improvement_plan.md)  
> 참조: [track_b_architecture_analysis.md](file:///home/student1/.gemini/antigravity-ide/brain/921a79fc-46b9-442a-9092-725b75f38bef/track_b_architecture_analysis.md)  
> 질문: **정말, 우리 문제 본질에 맞게 개선을 시도했는가?**

---

## 1. Track B의 진단된 핵심 병목 vs 현재 개선 대상

Track B 분석 보고서(§5.1)에서 4관점이 합의한 **핵심 병목 3가지**:

| # | 병목 | 관련 모듈 | 증거 | 현재 개선이 이것을 다루는가? |
|---|---|---|---|:---:|
| **L1** | 공간 해상도 부족 (P=32 → 256 tokens) | Stem + Backbone | P=16에서 direction_corr +91% | ❌ **이미 P=16 전환으로 해결** |
| **L2** | Motion 학습 경로 제한 (SPADE affine만, warp 없음) | Dynamic Decoder | σ_resid 병목 r=0.998 | ❌ **Stem과 무관** |
| **L3** | Skip connection 부재 (U-Net advantage 미활용) | Stem → Decoder | Deep Skip 실패(t-gate 미적용) | ❌ **Skip이 아닌 Stage 내부 수정** |

그리고 개선 우선순위 테이블(§5.3)에서 **Stem Residual connection**의 위치:

```
P0  — patch=16 전환         ← 이미 완료
P0  — SwiGLU MLP            ← Backbone
P1  — 2D flow + warp loss   ← Dynamic Decoder
P1  — Skip + t-gate 재시도  ← Stem↔Decoder 연결
P1  — Residual-weighted dyn ← Dynamic Decoder
P2  — InstanceNorm ablation ← SPADE
P2  — Spatial Expert K sweep← Backbone
P2  — Stem Residual         ← ★ 우리가 하고 있는 것
```

> [!WARNING]
> **현재 Stem V2 개선(Residual + start_ch=128)은 P2 수준**으로 분류되어 있었다.
> 핵심 병목 L1/L2/L3 어디에도 직접 대응하지 않는다.

---

## 2. 4관점 비판

### 설계 분석 관점

**Stem은 전체 파이프라인의 ~4% (1.55M / 39.5M)**에 불과하다.

현재 개선의 가설: "4단 직렬 구조에서 gradient vanishing → fine-grained feature 학습 부족"

이 가설의 문제:
1. **4단은 깊지 않다.** ResNet의 skip connection이 필수가 된 것은 50~152 layer 규모. 4단 Conv+GELU에서 gradient vanishing이 실제 병목인지 **gradient magnitude를 측정한 적이 없다.**
2. **Stem 출력은 Transformer가 8회(D=4, Self+Cross) 재가공한다.** Stem의 미세한 품질 차이가 최종 output까지 전파되려면 Transformer의 residual stream을 통과해야 한다. Transformer 자체의 표현력이 Stem의 부족을 보상할 가능성이 높다.
3. **start_ch=128 (params +175%)는 과투자.** 1.55M → 4.26M으로 Stem만 2.7× 확대하지만, 같은 파라미터 예산을 Backbone (D=4→D=6)이나 Expert (K=8→K=16)에 투자하면 ROI가 더 높을 수 있다.

**핵심 질문**: Stem의 token 품질이 정말 병목인가, 아니면 Transformer가 이미 보상하고 있는가?

### 구현 검수 관점 — 실증 관점

**가설 검증 절차가 누락되어 있다.**

개선 전에 확인했어야 할 진단:
1. ❌ **Stem Stage별 gradient magnitude** — Stage 0의 gradient가 Stage 3 대비 얼마나 작은가?
2. ❌ **Stem 출력 token의 품질 비교** — Stem 출력과 GT token (GT를 같은 방식으로 patchify)의 cosine similarity
3. ❌ **Stem feature의 PCA 시각화** — Stage 0~3에서 어떤 패턴이 학습되었는가?
4. ❌ **Ablation baseline**: Linear Stem vs Deep Stem의 Track B 직접 비교

이 진단 없이 "gradient flow 강화"를 목표로 Residual을 추가한 것은 **가설 기반 개선이 아닌 직관 기반 개선**.

### 도메인 적합성 관점

**Solar irradiance forecasting의 핵심 challenge는 temporal이지 spatial이 아니다.**

- **Persistence baseline과의 격차**: MAE 기준 ~3.7%뿐. 모델이 "어제와 비슷하다"보다 약간 나은 수준
- **Error의 주된 원인**: 구름의 이동과 발달/소멸 예측 실패 (temporal)
- **Stem은 순수 spatial encoder** — 개별 프레임을 독립적으로 토큰화. 시간 정보를 전혀 다루지 않음
- **Stem을 개선해서 얻는 것**: 더 좋은 공간 특징 → 하지만 temporal reasoning은 Backbone의 Cross-Attention에 의존

결론: **더 정밀한 공간 인코딩이 temporal prediction 개선으로 이어진다는 인과 관계가 불분명.** Cloud boundary를 더 잘 인코딩하더라도, 그 boundary가 30분 후 어디로 이동하는지를 Backbone이 더 잘 예측하게 되는 것은 아니다.

### 방법론 및 실험 검토 관점

1. **Ablation 설계 오류**: 변수 2개 동시 변경 (Stem 구조 + A3 loss)으로 시작했다가 뒤늦게 통제 실험으로 재설계. 초기 실험 시간 낭비.
2. **성공 기준이 모호**: "MAE -2% 또는 FID -20%"는 P=16 3-seed mean 기준으로도 noise 범위 내일 수 있다. Single seed에서 -2%는 paradigm σ의 2배이므로 **유의미할 수도 있으나 확실하지 않다.**
3. **GroupNorm 기각의 해석이 피상적**: "noise schedule과 미스매치"라 했지만, DDPM U-Net은 GroupNorm을 사용한다. 차이점은 DDPM U-Net은 **timestep-conditioned normalization**을 사용한다는 것. 단순 GroupNorm 실패가 아니라, **Stem이 timestep을 받지 않으므로** noise-level에 무관하게 normalize되는 것이 문제. 이 분석이 빠져 있다.

---

## 3. 그럼에도 불구하고 — 현재 개선이 유효한 이유

위 비판에도 불구하고, **완전히 무의미하지는 않다**:

1. **P=16에서 Stem의 역할이 커졌다.** P=32(5단, 256 tokens)에서 P=16(4단, 1024 tokens)으로 전환하면서 Stem이 처리하는 해상도당 정보량이 4배 증가. Stem 품질의 중요도가 P=32보다 높아졌다.

2. **Residual의 FID 개선은 실제 신호.** 9.54 (FID)는 이전 P=16 결과 (45.4~57.8) 대비 극적 개선. 비록 baseline과의 차이인지 Residual 효과인지 아직 분리되지 않았지만, 분포 품질 자체는 좋다.

3. **저비용 실험.** 100 epoch ≈ 30분. Stem 개선의 ROI가 낮더라도 실험 비용도 낮으므로, 확인 후 넘어가는 것이 합리적이다.

4. **Pareto 개선 가능성.** FID↓ + LPIPS↓ (분포/지각 품질) 이면서 MAE가 동등하면, 논문에서 "Stem Residual이 distribution quality를 개선"이라는 소규모 contribution claim이 가능하다.

---

## 4. 종합 판단

| 관점 | 판단 |
|---|---|
| **본질 적합성** | ❌ Track B의 핵심 병목(L1 해상도, L2 motion, L3 skip)과 직접 연결되지 않음 |
| **가설 검증** | ❌ "gradient vanishing" 가설이 사전 진단 없이 시작됨 |
| **실험 타당성** | ⚠️ 통제 실험 재설계 후에는 양호. 단, 초기 설계 오류로 시간 소비 |
| **비용 대비 가치** | ✅ 저비용 실험이므로 확인 후 넘어가는 전략은 합리적 |
| **다음 단계 시사점** | Stem 결과 무관하게, **진짜 ROI가 높은 P1 작업**(Skip+t-gate, 2D flow, dynamic loss weighting)으로 이동해야 함 |

> [!IMPORTANT]
> **결론**: 현재 Stem V2 실험은 **"해볼 만했지만 핵심은 아닌"** 작업이다.
> 진행중인 (4) Residual + A3 실험 결과를 확인하고, 결과에 무관하게
> **P1 우선순위 작업** (특히 motion 학습 경로 강화)으로 전환하는 것이 바람직하다.
>
> Stem 개선에서 배운 교훈:
> - GroupNorm은 timestep-agnostic stem에서 유해 → **normalization은 t-conditioned 경로에서만 사용**
> - Residual이 FID를 개선할 가능성 → **최종 config에 포함하되 주력이 아님**

---

## 5. 제안: 다음에 해야 할 "본질적" 개선

Track B의 핵심 병목에 직접 대응하는 작업:

| 우선순위 | 작업 | 병목 | 기대 효과 |
|:---:|---|---|---|
| **1** | Dynamic loss weighting (motion 큰 pixel에 집중) | L2 | MAE 직접 개선 (trivial pixel 95% gradient 제거) |
| **2** | Deep Skip + t-gate (alpha_bar schedule) 재시도 | L3 | Multi-scale feature reuse → fine detail 복원 |
| **3** | SwiGLU MLP 교체 (이미 구현됨) | Backbone | 파라미터 동등, 표현력 향상 |
