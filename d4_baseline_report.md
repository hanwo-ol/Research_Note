# D4 Baseline 계보 및 후보군 종합 분석

> 모든 실험은 D4 (depth=4, 1yr 2021 test) 기준. 100 epoch 학습.
> 비교 대상: AC-UNet (SSIM 0.562, MSE 0.026, FID 79.27)

---

## 1. Baseline 계보 (확정 경로)

```
11B_Depth4 (SSIM 0.508) ── DWT 3-level, 64ch
    │
    ├─ [DWT 2-level 전환]
    │
15A_DWT2Level (SSIM 0.518, +1.9%) ── DWT 2-level, 16ch, LPIPS -11.2%
    │
    ├─ [Xavier Conv Init 추가]
    │
E1_XavierConv (SSIM 0.530, +2.3%) ── D4 1yr 역대 최고 ★
```

| 단계 | SSIM | MSE | LPIPS | FID | 핵심 변경 |
|:---|:---:|:---:|:---:|:---:|:---|
| 11B (원본) | 0.508 | 0.040 | 0.672 | 212 | 8E Hetero MoE + BayesShrink |
| 15A (+DWT 2-level) | 0.518 | 0.038 | 0.597 | 200 | 64ch -> 16ch, 중간주파수 보존 |
| **E1** (+Xavier) | **0.530** | **0.038** | **0.590** | **204** | Conv 전문가 활성화 정상화 |

---

## 2. 전체 실험 성능 순위 (SSIM 기준)

### Tier 1: 개선 달성 (SSIM > 0.520)

| 순위 | 실험 | MSE | SSIM | LPIPS | FID | 비고 |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | **E1_XavierConv** | 0.038 | **0.530** | 0.590 | 204 | ★ 현재 baseline |
| 2 | E3a_AuxL1 | **0.036** | 0.527 | 0.588 | 204 | MSE 최저 |
| 3 | E5_DWTBias | 0.037 | 0.526 | 0.596 | 218 | FID 악화 |
| 4 | E6c_dwt_SG | 0.039 | 0.526 | 0.590 | 210 | E1 기반 |
| 5 | E6a_PhysVSG | 0.040 | 0.523 | 0.594 | 204 | V+SG 조합 |
| 6 | E3b_AuxL0 | 0.038 | 0.523 | 0.590 | 216 | FID 악화 |
| 7 | E6k_PhysSD | 0.039 | 0.522 | 0.596 | 222 | FID 악화 |
| 8 | E3a_E6a_Combo | 0.040 | 0.520 | 0.589 | 211 | 시너지 부재 |

### Tier 2: 동등 수준 (0.500 <= SSIM <= 0.520)

| 실험 | MSE | SSIM | LPIPS | FID | 비고 |
|:---|:---:|:---:|:---:|:---:|:---|
| 15A_DWT2Level | 0.038 | 0.518 | 0.597 | 200 | E1 이전 baseline |
| E2_MLPRatio8 | 0.038 | 0.517 | 0.602 | 201 | MLP 확장 무효 |
| E12B_AuxPixSSIM | 0.039 | 0.517 | 0.600 | 217 | Pixel SSIM aux |
| 12A_TemporalLoRA | 0.042 | 0.509 | 0.665 | 203 | LoRA 미미 |
| 11B_Depth4 | 0.040 | 0.508 | 0.672 | 212 | 원본 baseline |
| 12B_ConvLR5x | 0.041 | 0.507 | 0.661 | 191 | FID 최선 |
| 14A_Hidden768 | 0.041 | 0.505 | 0.666 | 192 | 파라미터 2.25배 |
| E4_FreqRes | 0.039 | 0.505 | 0.590 | 195 | 주파수 잔차 |
| E6h_PhysCSI | 0.042 | 0.504 | 0.607 | 222 | CSI 미미 |
| E3a_E6a_dwt_Combo | 0.040 | 0.500 | 0.601 | 221 | DWT V+SG 악화 |

### Tier 3: 악화 (SSIM < 0.500)

| 실험 | SSIM | 원인 |
|:---|:---:|:---|
| E6f_PhysTV | 0.498 | 분산 정보 노이즈 |
| 11E_PCIR | 0.489 | PCIR 불필요 |
| E6b_PhysV | 0.476 | V 단독 불충분 |
| E6j_PhysPD | 0.477 | 잔차 기반 불안정 |
| E6i_PhysLSV | 0.465 | 고주파 노이즈 증폭 |
| 11C_BL_Flat (no MoE) | 0.446 | MoE 제거 시 -12.2% |
| E3a_AuxPixSSIM | 0.444 | 두 aux 조합 gradient 간섭 |
| 13B_v2_AdaptSched | 0.436 | Adaptive Schedule 실패 |
| 13C_BandSched | 0.431 | Band Schedule 이중보정 |
| E6c_PhysSG | 0.411 | SG 단독 심각 악화 |
| E6g_PhysLAP | 0.408 | Laplacian 2차 미분 극단 |
| 13B_AdaptSched | 0.405 | Schedule 과도 분산 |
| E6a_dwt_VSG | 0.392 | DWT 공간 피처 최악 |
| 11D_BL_Grouped | 0.343 | Grouped Stem 단독 실패 |

---

## 3. Phase별 개선 기여도 요약

| Phase | 핵심 기법 | SSIM 기여 | 비고 |
|:---|:---|:---:|:---|
| Phase 2-A | 8E Hetero MoE | +12.2% | 아키텍처 핵심 (11C -> 11B) |
| Phase 2-B | DWT 2-level | +1.9% | 채널 수 축소 + 중간주파수 보존 |
| Phase 2-C | Per-Channel Min-SNR | 내재 | 채널별 gradient 균형 |
| Phase 2-D | BayesShrink | 내재 | 추론 HF 후처리 |
| Phase 2-E | Xavier Conv Init | +2.3% | Conv 전문가 활성화 정상화 |
| Phase 2-F | AuxDWT L1 | +1.7% (15A 기준) | E1 조합 미검증 |

---

## 4. 현재 진단: LL MSE Bottleneck

MSE Tracker 분석 결과, **ch0 (LL의 LL)이 전체 Pixel MSE의 78.3%를 기여**:

| 채널 | Z-score MSE | std (scale) | Raw MSE | 기여도 |
|:---|:---:|:---:|:---:|:---:|
| ch0 (LL) | 0.158 (최소) | 1.783 | 0.504 | **78.3%** |
| ch1-3 (LL) | ~1.1 | 0.10~0.18 | 0.01~0.04 | 11.4% |
| ch4-15 (HF) | ~1.0 | 0.07~0.10 | 0.004~0.013 | 10.3% |

**원인**: per_channel_snr (gamma=5)이 ch0의 gradient를 과도하게 억제.

---

## 5. 진행 중 / 예정 실험

### Phase 2 잔여 (tmux 0 실행중)

| 실험 | 상태 | 목적 |
|:---|:---:|:---|
| E1_AuxL1_D4 | 대기 | Xavier + AuxL1 시너지 최종 확인 |
| 17A_MaskedContext_D4 | 대기 | Context 토큰 마스킹 |
| 17B_LongSkip_D4 | 대기 | U-DiT 잔차 연결 |
| 17C_CtxEncoder_D4 | 대기 | Context 전용 인코더 |
| 17D_Hourglass_D4 | 대기 | 다해상도 Hourglass |

### Phase 3 (이후 실행)

| 실험 | 목적 |
|:---|:---|
| M3a/b Heun | 2차 Sampler NFE 비교 |
| M6 DWTCtxStats | 서브밴드 통계 조건 주입 |
| M1 EDM Precond | 입력 정규화 |
| M2a/b/c CtxNoise | Context 섭동 |
| M4a/c CrossTrack | LF-HF 교차 어텐션 |
| **L1 GammaLL20** | LL gradient +29.8% 강화 |
| **L2 Gamma20** | Global gamma 대조군 |
| **L3 GammaLL50** | gamma 상한 탐색 |
| **L4 LLx0Aux** | LL x_0 직접 감독 |
| **L5 GammaLL20+x0Aux** | A+C 조합 |
