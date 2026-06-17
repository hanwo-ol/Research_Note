# 나의 검토자 관점 실험 전략 진단

## 현재 상태 진단

### 확보된 것

| 항목 | 상태 |
|:---|:---|
| 3yr 모델 baseline (9G) | MSE 0.0374, SSIM 0.540, FID 121 |
| DDIM 40 steps ablation | 모든 지표 악화 확정 -- step 수는 원인이 아님 |
| Depth ablation (4/8/12) | Depth 8이 최적 균형점 |
| Stem ablation (flat/grouped) | FreqMoE가 핵심 기여 요인 |
| Conv LR 5x (12C) | FID -15.5%, SSIM +1.7% -- perceptual quality 개선 |
| BayesShrink sweep | noise_var=0.01 확정, soft threshold 본질적 한계 확인 |
| 13B Adaptive Schedule (D4) | **실험 진행 중** |

### 논문 관점에서 부족한 것

#### 1. "원인" 분리가 안 됨

현재까지 확인된 사실:
- ST-DiT vs checkthis 성능 격차: MSE -30.5%, FID -34.5%
- DDIM step 수 증가는 원인이 아님

그러나 **남은 후보 원인들이 분리되지 않았다**:

| 후보 원인 | 검증 상태 |
|:---|:---|
| Architecture (U-Net vs DiT) | 미검증 (분리 불가) |
| Input space (pixel vs DWT latent) | 미검증 (6-A/6-B 미실행) |
| Noise schedule (global vs region-adaptive) | **검증 중 (13B)** |
| Training data volume/quality | 동일 데이터 사용 -- 배제 |

> [!IMPORTANT]
> Reviewer가 "DWT latent가 pixel space보다 근본적으로 불리한 것 아닌가?"라고 물으면 답할 근거가 없다.
> 6-A (VAE Reconstruction Error 분석)가 이 질문에 대한 최소한의 방어선.

#### 2. 조합 실험 부족

Phase 9의 ConvLR5x(12C)가 FID를 크게 개선했고, 지금 Adaptive Schedule(13B)을 테스트 중이다. 그런데 **두 개를 조합한 실험이 없다**. Reviewer는 "이 두 개선이 직교적인가, 아니면 중복되는가?"를 반드시 물을 것이다.

#### 3. MSE vs Perceptual Trade-off 반복

| 실험 | MSE 변화 | FID 변화 | 패턴 |
|:---|:---:|:---:|:---|
| 12C (ConvLR5x) | +3.9% | **-15.5%** | MSE 악화, FID 개선 |
| BayesShrink 강화 | **개선** | 악화 | MSE 개선, FID 악화 |

이 trade-off가 구조적인 것인지, 아니면 해결 가능한 것인지에 대한 분석이 필요하다.

---

## 실험 우선순위 제안

### Tier 1: 지금 당장 (13B 결과 대기 + 후속)

**13B 결과에 따라 분기:**

```
13B (Adaptive Schedule D4) 결과
  ├── 효과 있음 (MSE/SSIM 개선) → 13C: Adaptive + ConvLR5x 조합
  │     → D8로 확장 (13D: Adaptive Schedule D8)
  │     → 3yr 학습으로 확장 (Final)
  └── 효과 없음/미미 → 원인 재분석 필요
        → 분산맵 시각화로 adaptive schedule이 의도대로 작동하는지 진단
        → clamp 범위 조정 등 하이퍼파라미터 탐색
```

**13C: Adaptive Schedule + ConvLR5x (D4)**
- 목적: 두 개선의 직교성 검증
- 12B(ConvLR5x D4) 대비 Adaptive Schedule의 추가 효과 측정
- 논문 기여: Ablation Table에서 "조합 효과" 입증

### Tier 2: Tier 1 결론 후 (3-5일 내)

**7-A: Hidden Size 1024**
- 목적: expansion ratio 병목 해소
- 현재 512dim = 2x expansion. 1024로 올리면 4x.
- **가장 큰 잠재 효과**이나, 학습 시간 2배 + VRAM 3배
- 논문 기여: Section 4.5 Scaling Analysis

**12-A: Dual Aux Deep Supervision (D4)**
- 목적: DWT 계층 구조 명시적 학습
- gradient flow 개선으로 HF 채널 학습 가속
- 논문 기여: Section 3.x Architecture

### Tier 3: 논문 방어용 (제출 전 필수)

**6-A: VAE Reconstruction Error 분석**
- 30분 소요. 스크립트 1개.
- DWT(lossless) vs VAE(lossy) 복원 오차 정량화
- Reviewer 질문 "DWT latent가 왜 pixel space보다 나은가?"에 대한 실증 근거
- 논문 기여: Section 4.4 또는 Appendix

**App. E: Computational Cost Table**
- 5분 소요.
- Params(M), Peak Memory(GB), Train Time(h)
- 논문 Table 필수 항목

---

## 논문 Ablation Table 구조 (제안)

모든 실험이 하나의 Table로 수렴해야 한다:

| Exp | Adaptive Sched | ConvLR5x | Depth | MSE | SSIM | FID |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 11B (baseline) | - | - | 4 | 0.0401 | 0.508 | 212 |
| 12B | - | 5x | 4 | 0.0412 | 0.507 | 191 |
| 13B | ON | - | 4 | ? | ? | ? |
| 13C | ON | 5x | 4 | ? | ? | ? |
| 9G | - | - | 8 | 0.0384 | 0.515 | 203 |
| 12C | - | 5x | 8 | 0.0399 | 0.524 | 172 |
| 13D | ON | - | 8 | ? | ? | ? |
| 13E | ON | 5x | 8 | ? | ? | ? |

이 테이블이 완성되면:
- Adaptive Schedule의 독립 기여도
- ConvLR5x의 독립 기여도
- 두 기법의 직교성/시너지
- Depth에 따른 효과 변화

를 모두 검증할 수 있다.

---

## 변경이력

| 날짜 | 내용 |
|------|------|
| 2026-04-23 | 초안 작성. 나의 검토자 관점 진단 + 실험 우선순위 제안. |
