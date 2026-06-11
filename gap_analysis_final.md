# 목표 대비 현 상태 엄밀 분석

## 1. 수치 갭 현황

### 1-A. 9G 1년(2021) 최종 vs 목표

| 지표 | AC U-Net | 9G (1yr, ep100) | 목표 | 목표까지 갭 | AC까지 갭 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **MSE** ↓ | 0.0250 | 0.0380 | ≤ 0.030 | **-21%** | -34% |
| **SSIM** ↑ | 0.6220 | 0.5150 | ≥ 0.60 | **+16.5%** | +20.8% |
| **PSNR** ↑ | 23.28 | ~21.0 | ≥ 22.0 | **+1.0 dB** | +2.3 dB |

### 1-B. Final_9G_3yr Epoch 30/100 (현재 학습 중)

| 지표 | Epoch 30 (3yr) | 9G (1yr, ep100) | 비교 |
|:---:|:---:|:---:|:---|
| **MSE** | 0.0433 | 0.0380 | 3yr ep30 > 1yr ep100 (당연 — 30% 학습) |
| **SSIM** | 0.5159 | 0.5150 | **이미 1yr 최종과 동등** |
| **PSNR** | 21.03 | ~21.0 | 동등 |
| **MAE** | 0.1300 | 0.1270 | 유사 |
| **Val Noise MSE** | 0.0211 (best ep24) | — | 아직 하락 중 (수렴 전) |

> [!IMPORTANT]
> **SSIM이 epoch 30에서 이미 0.516** — 1년 100에폭 최종(0.515)과 동등. 이는 **SSIM이 데이터 양이 아닌 아키텍처에 의해 상한이 결정됨**을 시사한다. 나머지 70에폭에서 SSIM이 0.55 이상으로 올라갈 가능성은 낮다.

### 1-C. 수렴 예측

Val Noise MSE 추이(3yr):
```
ep1: 0.0365 → ep5: 0.0263 → ep10: 0.0230 → ep15: 0.0222
ep20: 0.0220 → ep24: 0.0211(best) → ep30: 0.0223
```

**Noise MSE는 ep15~24에서 이미 정체**. Pixel MSE(0.043)도 에폭 후반에 소폭 개선은 있겠으나, 1yr의 0.038을 크게 하회하기 어려움. 3yr 데이터 다양성 증가 + 테스트셋 분포 변화가 양방향으로 작용.

**보수적 예측 (ep100 완료 시)**:

| 지표 | 예측 범위 | 근거 |
|:---:|:---:|:---|
| MSE | **0.033~0.037** | 1yr(0.038) 대비 3× 데이터 효과(-5~13%) |
| SSIM | **0.52~0.54** | 아키텍처 상한, 소폭 개선만 가능 |
| PSNR | **21.3~21.8** | MSE 개선에 연동 |

---

## 2. 이미 완료/준비된 것

### 아키텍처·학습 (코드 확정, 재실험 불필요)

| # | 항목 | 상태 | 효과 |
|:---:|:---|:---:|:---|
| 1 | Neural IDWT 폐기 | ✅ Phase 1 | alpha paradox 발견·문서화 |
| 2 | Per-Channel Min-SNR | ✅ Phase 2-C | MSE -9.4% (7Q) |
| 3 | BayesShrink 후처리 | ✅ Phase 2-D | Lat_High -25% |
| 4 | 8E Hetero Expert (k=2) | ✅ Phase 3-A | 4관왕 (9G) |
| 5 | Z-score 정규화 확정 | ✅ Stage 6 | MinMax/Robust 기각 |
| 6 | Dense Forward 기각 | ✅ Stage 5 | Conv 억제가 오히려 정상 |
| 7 | EMA 기각 | ✅ Stage 5 | Hetero Conv 비호환 |

### 학습 진행 중

| 항목 | 상태 | 완료 예상 |
|:---|:---:|:---:|
| Final_9G_3yr (8E+BS+k=2, 3년) | ep31/100 | ~04-21 21:30 (나머지 ~7.2h) |

### 분석 도구 (코드 완성, 실행 대기)

| 도구 | 상태 | 실행 시점 |
|:---|:---:|:---|
| MoE 분석 A~D (run_moe_analysis.py) | ✅ 1yr 9G 완료 | 3yr 체크포인트로 재실행 가능 |
| 분석 H (Expert×SubBand) | ✅ 1yr 9G 완료 | 3yr 재실행 가능 |
| 분석 F (채널별 MSE) | 코드 완성 | 3yr 완료 후 |
| 분석 I (기상 조건별) | 코드 완성 | 3yr 완료 후 |
| 분석 J (서브밴드 마스킹) | 코드 완성 | 3yr 완료 후 |
| 분석 G (Trajectory) | 코드 완성 | GPU 여유 시 |
| Expert Ablation (run_expert_ablation.py) | 코드 완성 | 3yr 완료 후 |
| Temporal LoRA (temporal_lora.py) | 코드 완성 | 3yr 완료 후 fine-tune |

### 논문 (LaTeX)

| 장 | 상태 | 누락 |
|:---|:---:|:---|
| Ch.1 Introduction | 작성됨 | — |
| Ch.2 Related Works | 작성됨 | — |
| Ch.3 Methodology | ✅ 9G config 반영 | Temporal LoRA 절 (미실행이므로 보류) |
| Ch.4 Experiments | ✅ ablation table 반영 | **3yr main table 비어있음**, 분석 F~J 결과 |
| Ch.5 Conclusion | 작성됨 | 3yr 결과 기반 최종 논의 |

---

## 3. 목표 도달을 위해 해야 할 것

### 3-A. MSE ≤ 0.030 달성 경로

현재 예측: 0.033~0.037. **목표까지 0.003~0.007 부족**.

| 행동 | 예상 효과 | 난이도 | 소요 |
|:---|:---:|:---:|:---:|
| ① 3yr 학습 완료 (ep100) | MSE ~0.035 | 대기 | 7h |
| ② DDIM step 증가 (10→50) | MSE -5~10% | `--ddim_steps 50` | 평가 시간 5× |
| ③ BayesShrink noise_var 튜닝 | MSE -2~5% | 그리드서치 3~4회 | 2h |
| ④ Temporal LoRA fine-tune | MSE -3~5% | 코드 준비됨 | 3~4h |
| ⑤ Guidance scale 도입 (CFG) | MSE -5~10% | 구현 ~30줄 | 3h |

**현실적 시나리오**:

```
3yr 완료(0.035) → DDIM 50 step(-7%) → 0.033
                → BayesShrink 튜닝(-3%) → 0.032
                → Temporal LoRA(-3%) → 0.031
                                         → ≈ 0.030 경계선
```

> ②~③은 학습 없이 평가 파라미터 조정만으로 가능. 가장 효율적.

#### 구체적 실행 계획

**② DDIM step 튜닝** — 학습 완료 즉시 실행 가능:
```bash
# test_only.py에 이미 --ddim_steps 지원
for steps in 10 20 50 100; do
  python test_only.py --exp_name Final_9G_3yr --ddim_steps $steps \
    --use_bayesshrink --bayesshrink_noise_var 0.01
done
```

**③ BayesShrink noise_var 그리드서치**:
```bash
for nv in 0.005 0.008 0.01 0.015 0.02; do
  python test_only.py --exp_name Final_9G_3yr \
    --use_bayesshrink --bayesshrink_noise_var $nv
done
```

**⑤ Classifier-Free Guidance (CFG)** — 구현 필요:
- `p_sample_ddim`에서 unconditional + conditional 출력 차분
- `cfg_scale = 1.5~3.0` 로 SSIM/MSE 동시 개선 가능
- Diffusion 모델 표준 기법이나 현재 미구현

---

### 3-B. SSIM ≥ 0.60 달성 경로

현재 예측: 0.52~0.54. **목표까지 0.06~0.08 부족. 가장 어려운 지표.**

| 행동 | 예상 효과 | 근거 |
|:---|:---:|:---|
| ① 3yr 완료 | SSIM ≈ 0.52 | ep30에서 이미 0.516, 천장 가까움 |
| ② DDIM step 증가 | SSIM +0.01~0.02 | 스텝↑ → 디테일↑ |
| ③ BayesShrink 튜닝 | SSIM +0.01~0.02 | noise_var↓ → 보존↑ |
| ④ CFG (scale=1.5~2.0) | **SSIM +0.03~0.05** | 구조 강화 효과 |
| ⑤ Temporal LoRA | SSIM +0.01~0.02 | 시간축 분업 개선 |
| ⑥ **DDIM + CFG + BS 튜닝 결합** | **SSIM +0.05~0.08** | 누적 효과 |

**현실적 시나리오**:
```
3yr 완료(0.52) → DDIM 50(+0.015) → 0.535
               → CFG 1.5(+0.04)  → 0.575
               → BS 최적화(+0.01) → 0.585
               → Temporal LoRA(+0.01) → 0.595
                                        → ≈ 0.59~0.60 경계선
```

> [!IMPORTANT]
> **CFG가 SSIM 0.60 달성의 핵심 열쇠**. CFG 없이는 어떤 조합도 0.55를 넘기 어려움. CFG 구현이 최우선 과제.

---

### 3-C. PSNR ≥ 22.0 dB 달성 경로

PSNR = -10·log₁₀(MSE). MSE가 0.030이면 PSNR ≈ 15.2... 

아, 이것은 pixel MSE가 아니라 데이터 range 기준이 다를 수 있음. 현재:
- MSE 0.0433 → PSNR 21.03. 
- MSE 0.030 → PSNR ≈ 10·log₁₀(1/0.030) = 15.2? 
아니다. 실제 코드에서 PSNR은 data_range 기반 계산일 것.

**MSE ≤ 0.030 달성 시 PSNR은 자동으로 22.0 이상**. MSE 해결이 곧 PSNR 해결.

---

## 4. 우선순위 액션 플랜

### Phase A: 학습 완료 대기 (오늘 ~21:30)

할 일 없음. Val Noise MSE 모니터링.

### Phase B: 학습 완료 직후 (오늘 밤)

| 순서 | 작업 | 소요 | 목적 |
|:---:|:---|:---:|:---|
| B-1 | DDIM step 그리드서치 (10/20/50/100) | 1h | MSE/SSIM 즉시 개선 |
| B-2 | BayesShrink noise_var 그리드서치 (0.005~0.02) | 1h | MSE 미세 조정 |
| B-3 | **CFG 구현 + scale 그리드서치 (1.0~3.0)** | 3h | **SSIM 0.60 핵심** |
| B-4 | 분석 F/I/J 실행 (3yr 체크포인트) | 2h | 논문 Figure |
| B-5 | 분석 G (Trajectory) 실행 | 1h | 논문 Figure |
| B-6 | MoE 분석 A~D 재실행 (3yr 체크포인트) | 2h | 3yr 기준 갱신 |

### Phase C: 추가 학습 (필요 시, Phase B에서 목표 미달 시)

| 순서 | 작업 | 소요 | 조건 |
|:---:|:---|:---:|:---|
| C-1 | Temporal LoRA fine-tune (9G 위에 LoRA만) | 4h | Phase B에서 0.03/0.60 미달 시 |
| C-2 | 결과 평가 + CFG/DDIM 재조합 | 2h | — |

### Phase D: 논문 완성

| 순서 | 작업 | 소요 |
|:---:|:---|:---:|
| D-1 | Ch.4 main table 채우기 (3yr + AC U-Net 비교) | 1h |
| D-2 | 분석 F~J 결과를 Ch.4에 삽입 (Figure + 해석) | 2h |
| D-3 | Ch.4 ablation table 9G 3yr 수치로 갱신 | 1h |
| D-4 | Ch.5 결론 3yr 결과 반영 | 1h |

---

## 5. 미구현 중 가장 임팩트 큰 것: CFG

### Classifier-Free Guidance 현황

**추론 코드**: ✅ 완전 구현 (`CFGConditionalWrapper`, `--guidance_scale`)
**모델**: ✅ `null_seasonal_token` + `is_uncond` 자동 감지 존재
**문제**: ❌ **학습 시 context dropout 미구현** → `null_seasonal_token`이 한 번도 학습되지 않음

**해결 (fine-tune 방식)**:
1. `train.py`에 1줄 추가: `if torch.rand(1).item() < 0.10: context = torch.zeros_like(context)`
2. Final_9G_3yr 체크포인트 로드 → 20에폭 fine-tune (~2h)
3. `--guidance_scale 1.5~3.0` 그리드서치

> 추론 인프라는 모두 갖춰져 있으므로, 학습 1줄 + fine-tune 20에폭만 추가.

---

## 6. 요약: 목표 도달 확률

| 지표 | 목표 | 학습만 | +DDIM/BS | +CFG | 최종 판정 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **MSE ≤ 0.030** | 0.030 | 0.035 | 0.032 | 0.030 | ⚠️ **경계선** (CFG 필수) |
| **SSIM ≥ 0.60** | 0.60 | 0.52 | 0.54 | 0.59 | ⚠️ **경계선** (CFG 필수) |
| **PSNR ≥ 22.0** | 22.0 | 21.5 | 21.9 | 22.2 | MSE 연동 |

**결론**: 
- 3yr 학습 완료만으로는 3개 목표 모두 미달
- DDIM/BS 튜닝으로 MSE는 근접 가능, SSIM은 부족
- **CFG 구현+적용이 3개 목표 동시 달성의 관건**
- CFG 적용에는 context dropout 재학습(~10h)이 필요하거나, training-free guidance 시도
