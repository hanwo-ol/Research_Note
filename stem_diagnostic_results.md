# Deep Stem 진단 결과 분석

**모델**: `stem_baseline_p16_seed42` (Sequential path, P=16, D=4)  
**진단 스크립트**: [diagnose_stem.py](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/scripts/diagnose_stem.py)  
**실행 일시**: 2026-05-20

---

## [A] Gradient Flow — ✅ 양호

| Stage | Channels | Resolution | Grad Norm |
|-------|----------|-----------|-----------|
| 0 | 1→64 | 512→256 | 0.1076 |
| 1 | 64→128 | 256→128 | 0.0488 |
| 2 | 128→256 | 128→64 | 0.0228 |
| 3 | 256→512 | 64→32 | 0.0258 |

- **Stage 0 / Stage 3 ratio = 4.18** → 건강한 gradient flow (vanishing 임계치 < 0.1)
- Stage 2→3에서 gradient가 약간 증가 — 학습 가중치가 뒷단 Conv에 집중되는 자연스러운 패턴

> [!TIP]
> **결론**: Deep Stem의 4-stage Sequential path에서 gradient vanishing 문제는 **없음**. Residual shortcut이 gradient 관점에서는 불필요.

---

## [B] Token Cosine Similarity — 🔑 핵심 인사이트

![Token Cosine Similarity vs Noise Level](/home/student1/.gemini/antigravity-ide/brain/921a79fc-46b9-442a-9092-725b75f38bef/token_cosine_similarity.png)

| Timestep t | Cosine Sim | 해석 |
|-----------|-----------|------|
| 0 | 1.0000 | Clean → 완벽 일치 |
| 50 | 0.9976 | 거의 동일 |
| 100 | 0.9929 | 미세 차이 |
| 200 | 0.9761 | 약한 noise |
| 500 | 0.8027 | 중간 noise — 구조 유지 |
| 999 | **0.1036** | 순수 noise → 거의 직교 |

> [!IMPORTANT]
> **핵심 관찰: Stem이 t-agnostic하여 noise level에 따라 선형적으로 token 품질이 저하됨.**
> 
> - t=500에서 cosine sim = 0.80 → Stem 출력이 clean GT와 20% 다른 방향을 가리킴
> - t=999에서 cosine sim = 0.10 → 거의 랜덤 (정보 손실)
> - **Stem이 timestep을 모르기 때문에** 동일한 Conv filter가 clean/noisy 입력에 동일하게 적용됨
> - 이는 backbone transformer가 t-embedding을 받아 보정하지만, **token 자체의 품질 열화는 이미 발생한 후**

### t-conditioned Stem의 이론적 근거

현재 Stem은 `f(x_t)` — timestep 정보 없이 x_t만 처리. 만약 `f(x_t, t)`로 바꾸면:

1. **Low-t (clean)**: 세밀한 texture feature 추출에 집중
2. **High-t (noisy)**: noise-robust한 coarse structure 추출로 전환
3. **Backbone 부담 감소**: transformer가 t-aware token 보정에 쓰는 capacity를 다른 곳에 활용 가능

---

## [C] PCA 시각화 — Spatial 구조 학습 확인

![Stem Feature PCA](/home/student1/.gemini/antigravity-ide/brain/921a79fc-46b9-442a-9092-725b75f38bef/stem_pca.png)

| Stage | Channels | Resolution | PCA Var (PC1) | 관찰 |
|-------|----------|-----------|--------------|------|
| 0 | 64ch | 256×256 | 96.1% | 밝기 위주 (cloud vs clear sky) |
| 1 | 128ch | 128×128 | 98.5% | 구름 경계 선명화 |
| 2 | 256ch | 64×64 | 96.5% | 대규모 구조 유지 |
| 3 | 512ch | 32×32 | 90.2% | **blocky artifact 출현** |

> [!WARNING]
> **Stage 3 (512ch, 32×32)에서 PCA 이미지에 block 격자 패턴이 관찰됨.**
> 
> - PC1의 설명 분산이 96.1% → 90.2%로 감소 = feature diversity 증가
> - 하지만 32×32 해상도에서 patch 경계와 정렬된 blocky 패턴 존재
> - 이는 최종 token으로 변환되는 단계이므로, token 품질에 직접 영향

---

## 종합 진단 및 다음 단계

### 현재 상태

| 진단 항목 | 결과 | 문제 여부 |
|----------|------|----------|
| Gradient flow | ratio=4.18 | ✅ 없음 |
| Token quality (low-t) | sim≥0.97 | ✅ 양호 |
| Token quality (mid-t) | sim=0.80 | ⚠ 경미 |
| Token quality (high-t) | sim=0.10 | ❌ 심각 |
| PCA spatial structure | Stage 0-2 양호 | ✅ |
| PCA Stage 3 blocky | 패턴 존재 | ⚠ 주의 |

### 결론

1. **Residual shortcut 불필요** — gradient vanishing이 없으므로 Stem V2의 residual은 성능이 아닌 안정성 보험
2. **t-conditioned Stem이 유의미** — token cosine sim 데이터가 이를 명확히 뒷받침
3. **PCA Stage 3의 blocky pattern** — Deep Stem이 패치 아티팩트를 줄였다는 기존 관찰과 일치하나, 32×32 해상도에서 여전히 잔여 격자 존재

### 추천 다음 단계

| 우선순위 | 작업 | 근거 |
|---------|------|------|
| **P1** | t-gated GroupNorm PoC 구현 | cosine sim 열화가 가장 큰 bottleneck |
| P2 | Stage 3 blocky pattern 원인 분석 | 최종 token 품질에 영향 |
| P3 | Res+A3 모델과 동일 진단 비교 | V2 개선의 정량적 효과 측정 |
