# MoE v2: E1 vs E2 FID 차이 분석 종합 리포트

## 실험 결과

| 실험 | MSE | PSNR | SSIM | FID | LPIPS |
|------|:---:|:---:|:---:|:---:|:---:|
| baseline (DC-6 3yr) | 0.0315 | 22.31 | 0.5645 | 99.59 | 0.4346 |
| **E1: Temp only** | 0.0335 | 22.01 | 0.5584 | **35.29** | **0.4112** |
| **E2: Temp+FiLM** | 0.0316 | 22.25 | 0.5599 | 86.09 | 0.4308 |

---

## A4: 조건별 분해 ✅ (전수 데이터, 4,079 샘플)

### Month별 LPIPS 차이 (E1−baseline / E2−baseline / E2−E1)

| 월 | E1−BL | E2−BL | E2−E1 |
|:--:|:-----:|:-----:|:-----:|
| 1 | −0.018 | −0.003 | **+0.015** |
| 4 | −0.020 | −0.004 | **+0.016** |
| 7 | −0.030 | −0.004 | **+0.027** |
| 9 | −0.035 | −0.003 | **+0.032** |
| 10 | −0.048 | −0.006 | **+0.042** |

> [!IMPORTANT]
> **E2−E1 LPIPS 차이가 모든 12개월에서 양수(+0.009 ~ +0.042)**.
> 특정 계절/조건 문제가 아닌 **구조적·전역적 차이**.
> E1은 baseline 대비 모든 월에서 LPIPS 개선, E2는 미미한 개선에 그침.

### 추가 발견: MSE는 E2가 더 좋음

- E1은 baseline 대비 MSE **악화** (+0.001~0.004 전 월)
- E2는 baseline과 MSE **동등** (±0.0003)

→ **E1은 pixel 정확도를 희생하면서 perceptual quality를 획득**

---

## A1: FiLM γ, β 분석 ✅ (체크포인트 기반, synthetic cond_emb 200개)

E2의 FiLM 파라미터 분석:

| Stage | C_in | γ std | (1+γ) mean | γ<0 ratio | FiLM inter-expert cos_sim |
|:-----:|:----:|:-----:|:----------:|:---------:|:------------------------:|
| 0 | 512 | **0.171** | 1.007 | 48.3% | **0.652** |
| 1 | 256 | **0.229** | 1.012 | 48.9% | **0.746** |
| 2 | 128 | 0.183 | 1.018 | 47.6% | 0.475 |
| 3 | 64 | **0.357** | 0.999 | 50.2% | 0.054 |

### 해석

1. **γ std 0.17~0.36**: FiLM은 **유의미한 크기의 변조**를 수행 (zero가 아님)
2. **(1+γ) ≈ 1.0, γ<0 ≈ 50%**: smoothing 편향 없음. 채널의 절반은 증폭, 절반은 억제 → **방향성 없는 변조**
3. **🔴 FiLM inter-expert cos_sim Stage 0/1 = 0.65~0.75**: 5개 expert의 FiLM 변조가 **매우 유사**

> [!WARNING]
> **핵심 발견**: Stage 0~1에서 expert별 FiLM이 cos_sim 0.65~0.75로 **거의 같은 변조**를 수행.
> expert 간 차별화된 입력 변조라는 FiLM의 설계 의도와 달리, 실제로는 **5개 expert 모두에 동일한 affine transform**을 가하는 것에 가까움.

---

## A5: Expert 출력 다양성 ✅ (체크포인트 기반, synthetic input)

| Stage | E1 cos_sim (no FiLM) | E2 cos_sim (no FiLM) | E2 cos_sim (+FiLM) | Δ(FiLM 효과) |
|:-----:|:---:|:---:|:---:|:---:|
| 0 | 0.239 | 0.226 | 0.245 | **+0.019** |
| 1 | 0.199 | 0.161 | 0.177 | **+0.016** |
| 2 | 0.225 | 0.164 | 0.166 | +0.002 |
| 3 | 0.032 | 0.043 | 0.056 | +0.013 |

### 해석

> [!IMPORTANT]
> **FiLM 적용 후 expert 출력 cos_sim이 상승** (Stage 0~1에서 +0.015~0.019).
> FiLM이 expert 입력을 변조하면서 **출력이 오히려 더 유사해짐**.
> FiLM이 각 expert에 같은 변조를 가하므로(A1에서 확인), Conv의 고유한 출력 차이가 FiLM 변조에 의해 **일부 상쇄됨**.

---

## A2/A3: 주파수/다양성 분석 ⚠️ (80장 시각화 PNG — 참고용)

> [!CAUTION]
> 80장은 GT vs Pred 비교용 시각화 PNG이므로, raw prediction이 아닙니다.
> 전수 데이터 분석을 위해서는 **모델 재추론이 필요**합니다.

참고 결과 (80장 기준):
- PSD: E1/E2 ratio ≈ 1.00~1.04 — 주파수 에너지 차이 미미
- Sharpness: 3개 모델 동일 (Laplacian var ≈ 32.0)
- 이 결과만으로는 원인 특정 불가

---

## 종합 메커니즘: 왜 FiLM이 FID를 악화시키나?

```
FiLM 설계 의도:
  expert_k(x * (1+γ_k) + β_k)  ← 각 expert에 다른 변조 → 전문화 촉진

실제 학습 결과:
  γ_0 ≈ γ_1 ≈ γ_2 ≈ γ_3 ≈ γ_4  (cos_sim 0.65~0.75)
  → 모든 expert에 거의 동일한 변조
  → expert 출력 cos_sim 상승 (A5)
  → 가중합(Σ w_k * out_k)의 다양성 감소
  → FID 악화
```

**E1이 더 좋은 이유**:
- Temp Annealing이 sharp routing을 유도 → 특정 expert에 집중
- FiLM 없이 Conv weight 차이만으로 expert 출력 다양성 유지
- pixel accuracy는 약간 희생하지만, perceptual quality(LPIPS, FID)는 크게 개선

**FiLM이 해로운 이유**:
- 5개 expert에 동일한 변조 적용 → expert 출력 동질화
- pixel MSE는 유지(FiLM이 mean 보정 역할) but perceptual diversity 감소
- 결과적으로 baseline과 동등한 성능으로 회귀

---

## 남은 작업

- [ ] **A2/A3 전수 분석**: E1/E2 모델 재추론하여 전체 4,079 샘플의 PSD/histogram 비교
  - test_only.py에 raw prediction 저장 옵션 추가 필요
- [ ] **E3 (Temp+FiLM+LB) / E4 (Temp+LB) 완료 대기**: LB Loss 효과 확인
  - E4(Temp+LB, FiLM 없음)이 E1과 비슷하면 → FiLM 제거 확정
- [ ] **FiLM 분리 실험 고려**: FiLM 입력을 cond_emb가 아닌 gate_extra로 변경하여 재실험
