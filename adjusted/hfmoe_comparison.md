# HFMoE 심층 분석 종합 보고서

> 분석일: 2026-04-29
> 데이터: 504개 test 샘플 pred_pixel npy (P0 vs A1_HFMoE_D4)

---

## 핵심 발견: 기존 가설의 전복

> [!IMPORTANT]
> **HFMoE의 문제는 smoothing(평탄화)이 아니라, 과도한 HF 노이즈 주입 + LF 구조 오염이다.**

---

## 1. PSD 분석 (D-4)

![PSD comparison](/home/oem/.gemini/antigravity/brain/19396cf4-6322-45b8-9603-521137006f13/d4_psd_comparison.png)

| 주파수 대역 | P0 에너지 | HFMoE 에너지 | 변화 |
|:---|---:|---:|:---:|
| DC+VLF (0-5%) | 7.66B | 7.70B | +0.49% |
| LF (5-20%) | 3.06M | 3.65M | **+19.3%** |
| MF (20-50%) | 203K | 254K | **+24.6%** |
| HF (50-100%) | 60.7K | 79.7K | **+31.3%** |

**HFMoE가 P0보다 모든 주파수 대역에서 에너지가 높다.** 특히 HF 대역에서 +31.3%. 이는 HFMoE가 예측에 **불필요한 고주파 아티팩트를 추가**하고 있음을 의미한다.

기존 가설("expert 가중합이 smoothing을 유발")은 **기각**. 실제로는 반대: 5개 expert의 다양한 convolution이 서로 다른 고주파 패턴을 생성하여, 가중합 시 **구조적 노이즈**가 된다.

---

## 2. 주파수 대역별 오차 분해 (D-6)

![Frequency band MSE](/home/oem/.gemini/antigravity/brain/19396cf4-6322-45b8-9603-521137006f13/d6_freq_band_mse.png)

| 대역 | \|FFT(HF-P0)\|^2 | 비율 |
|:---|---:|:---:|
| **VLF (0-10%)** | **622,841** | **93.0%** |
| LF (10-25%) | 38,335 | 5.7% |
| MF (25-50%) | 5,164 | 0.8% |
| HF (50-75%) | 2,168 | 0.3% |
| VHF (75-100%) | 1,240 | 0.2% |

**P0과 HFMoE 간 차이의 93%는 VLF(저주파) 대역에서 발생한다.** HFMoE가 HF 에너지를 과도하게 추가하면서, 동시에 **전체 이미지의 저주파 구조(밝기, 대규모 구름 분포)가 틀어졌다.**

이는 **HF 트랙 -> LF 트랙 역전파 경로(BiDirCrossScaleInteraction)를 통한 LF 오염**을 강하게 시사한다.

---

## 3. 공간 오차 맵 (D-5)

![Mean error map](/home/oem/.gemini/antigravity/brain/19396cf4-6322-45b8-9603-521137006f13/d5_mean_error_map.png)

| 영역 | Mean |Diff| |
|:---|:---:|
| Edge (경계) | 0.10405 |
| Interior (내부) | 0.09980 |
| Edge/Interior 비율 | **1.04x** |

오차가 edge와 interior에서 **거의 균일**하다 (1.04x). 이는 HFMoE의 문제가 특정 영역(edge 등)에 국한되지 않고, **이미지 전체에 걸친 systematic error**임을 확인한다.

---

## 4. 시간대별 패턴 (D-7)

![Hourly pattern](/home/oem/.gemini/antigravity/brain/19396cf4-6322-45b8-9603-521137006f13/d7_hourly_pattern.png)

| 시간(UTC) | N | dMSE | dSSIM | 비고 |
|:---:|:---:|:---:|:---:|:---|
| 0 | 17 | **+0.00474** | -0.01298 | MSE 최악 |
| 6 | 54 | +0.00313 | -0.01576 | 양쪽 악화 |
| 9 | 27 | +0.00051 | **-0.02499** | SSIM 최악 |
| 2 | 60 | +0.00073 | -0.00978 | 비교적 양호 |

- **야간(0시 UTC)**: MSE 최악 (+0.00474). 저신호 환경에서 expert들이 noise를 증폭
- **오전(9시 UTC)**: SSIM 최악 (-0.02499). 일출 후 복잡한 구름 패턴에서 구조적 유사도 크게 손상
- **새벽(2시 UTC)**: 가장 양호. 안정적인 야간 패턴에서 HFMoE 피해 최소

---

## 5. 종합 진단

```
                    HF MoE (5 Expert)
                         |
                    [과도한 HF 에너지 +31%]
                         |
              BiDirCrossScaleInteraction
                    (alpha 파라미터)
                         |
                    LF 트랙 오염
                         |
              [VLF 오차 93% = 전체 밝기/구조 왜곡]
                         |
                   MSE +9.5%, SSIM -3.7%
```

**근본 원인**: 5개 HF expert가 각각 다른 고주파 패턴을 생성 -> 가중합 시 **구조적 노이즈** 형성 -> BiDirCrossScaleInteraction의 alpha를 통해 LF 트랙으로 역전파 -> **LF 예측의 저주파 구조가 오염됨**

LF MoE(ST-Aware Router)가 성공한 이유: LF 트랙은 Transformer 기반으로 self-attention이 전체 구조를 조율. HF 트랙은 Conv2d 기반으로, 5개 expert의 독립적 conv 출력이 **전체 구조적 일관성 없이** 합산됨.

---

## 6. 개선 방향 (수정)

기존 가설(smoothing)이 기각되었으므로 개선 방향을 수정한다.

### 6.1. Cross-Scale Interaction alpha 분석 결과 + 감쇠 전략

체크포인트에서 추출한 alpha 학습값:

| Block | P0 alpha | HFMoE alpha | 차이 |
|:---:|:---:|:---:|:---:|
| 0 | -0.000040 | -0.000081 | -0.000041 |
| 1 | -0.000084 | -0.000036 | +0.000048 |
| 2 | -0.000074 | -0.000064 | +0.000010 |
| 3 | -0.000070 | -0.000051 | +0.000019 |

**두 모델 모두 alpha가 거의 0에 수렴**했다. 이는 BiDirCrossScaleInteraction이 학습 과정에서 cross-scale 교류를 스스로 억제하는 방향으로 수렴했음을 의미한다.

> [!IMPORTANT]
> alpha가 ~0이므로 "HF -> LF 역전파를 통한 LF 오염" 가설은 **약화**된다. alpha가 거의 비활성이면, HFMoE의 성능 악화는 cross-scale interaction이 아니라 **HF 트랙 자체의 출력 품질 악화**가 IWT(역웨이블릿) 복원 단계에서 직접 pixel 품질에 영향을 준 것으로 재해석해야 한다.

수정된 인과 경로:
```
HF MoE (5E 가중합) -> HF 채널에 구조적 노이즈 +31%
                   -> IWT 복원 시 HF 채널이 pixel에 직접 반영
                   -> pixel SSIM 악화 (-3.7%)
```

alpha 감쇠 전략은 alpha가 이미 ~0이므로 **추가 효과 없음**. 이 방향은 폐기.

### 6.2. Subband-Aware 3E (routing 제거, 물리 정렬)

5개 범용 expert -> 3개 DWT subband 전용. **routing 자체를 제거**하여 expert 간 간섭 원천 차단.
각 subband가 독립 처리되므로 가중합에 의한 구조적 노이즈 불가.

### 6.3. Residual Gating (P0 보존)

P0의 단일 7x7 DWConv expert를 **기본 경로로 유지**하면서, 보조 expert(예: EdgeExpert)를 학습 가능한 gate를 통해 조건부로 혼합하는 방식.

**동작 원리**:
1. 입력 `x_high`가 두 경로로 분기
2. 기본 경로: 기존 P0의 `HighFreqLocalExpert(7x7 DWConv + FFN + AdaLN)` -- 검증된 성능
3. 보조 경로: `EdgeExpert(5x5 DWConv)` 등 1개 추가 expert
4. Sigmoid gate `g(x) in [0, 1]`가 입력 특성에 따라 보조 expert의 기여도를 결정
5. 최종 출력: `main_out + g * (aux_out - main_out)` = `(1-g)*main + g*aux`

**안전성 보장**: gate가 0으로 수렴하면 출력은 P0과 완전히 동일. 따라서 P0 대비 성능 악화가 **구조적으로 불가능**.

**기대 효과**: gate가 Q1(어두운 장면)처럼 보조 expert가 유리한 조건에서만 열리고, Q3/Q4(밝은 장면)에서는 닫혀 P0 성능을 유지.

**한계**: 개선 폭이 gate의 학습에 의존. 데이터가 적으면(1yr 분기) gate가 열리지 않아 P0과 동일한 결과가 될 수 있음.

### 6.4. (폐기) Sparse Forward

기존 A2a(Sparse Forward)는 "선택된 expert만 실행"으로 효율 개선을 목표했으나, 근본 문제가 **expert 가중합의 구조적 노이즈**이므로 sparse execution만으로는 해결 불가. expert 수가 2개로 줄어도 서로 다른 HF 패턴의 가중합 문제는 잔존.

---

## 7. 추가 분석 가능 항목

| 분석 | 필요 자원 | 기대 인사이트 | 우선순위 |
|:---|:---:|:---|:---:|
| **A-1 Router distribution** | GPU (ZLoss 완료 후) | Router collapse 여부 -> z-loss 효과 판단 | 높음 |
| **A-2 Expert output correlation** | GPU (ZLoss 완료 후) | 5 expert 간 redundancy 정도 -> expert 수 결정 근거 | 높음 |
| ~~alpha 값 추적~~ | ~~checkpoint 로드~~ | ✅ 완료. alpha ~0 수렴, cross-scale 비활성 확인 | ~~높음~~ |
| **P0 vs HFMoE latent MSE** | 재추론 (--save latent) | latent 공간에서 LF/HF 채널별 오차 분리 | 중간 |

alpha 값 추적은 checkpoint에서 즉시 확인 가능합니다.

---

## 8. 분석 파일

| 파일 | 설명 |
|:---|:---|
| `analysis/p0_vs_hfmoe/d4_psd_comparison.png` | Radial PSD + PSD ratio + band energy |
| `analysis/p0_vs_hfmoe/d5_mean_error_map.png` | 504개 평균 오차 맵 + edge vs interior |
| `analysis/p0_vs_hfmoe/d6_freq_band_mse.png` | 주파수 대역별 HF-P0 오차 분해 |
| `analysis/p0_vs_hfmoe/d7_hourly_pattern.png` | 시간대별 dMSE/dSSIM 패턴 |
