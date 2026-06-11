# BayesShrink noise_var 채널별 영향 분석

> thesis_notation.md 규약 준수

---

## 1. 7-C Grid Search 결과 (Final_9G_3yr, 2021 test set)

| $\hat{\sigma}^2_{\text{noise}}$ | MSE | SSIM | LPIPS | FID |
|:---:|:---:|:---:|:---:|:---:|
| 0.001 | 0.0418 | 0.517 | **0.615** | **146** |
| 0.003 | 0.0414 | 0.522 | 0.623 | 148 |
| 0.005 | 0.0411 | 0.524 | 0.629 | 150 |
| 0.008 | 0.0412 | 0.525 | 0.636 | 151 |
| 0.01  | 0.0408 | 0.526 | 0.639 | 154 |
| 0.02  | 0.0405 | 0.526 | 0.646 | 160 |
| 0.05  | **0.0398** | **0.527** | 0.653 | 175 |

MSE/SSIM은 단조 개선, LPIPS/FID는 단조 열화 -> **trade-off 존재**.

---

## 2. 현재 BayesShrink 수식 (thesis Eq. 3.9)

BayesShrink는 **raw wavelet space** (역스케일링 완료 후)에서 적용:

$$
\lambda_c = \frac{\hat{\sigma}^2_{\text{noise}}}{\hat{\sigma}_{\text{signal},c}}, \qquad
\hat{\mathbf{z}}^{(c)}_{\text{refined}} = \mathrm{sign}(\hat{\mathbf{z}}^{(c)}_0) \cdot \max\!\big(|\hat{\mathbf{z}}^{(c)}_0| - \lambda_c,\; 0\big)
$$

여기서 $\hat{\sigma}_{\text{signal},c} = \sigma_{\text{data},c}$ (채널 $c$의 raw space 표준편차).

**채널별 threshold 실측값:**

| 채널 | 대역 | $\sigma_{\text{data},c}$ | $\lambda_c$ (nv=0.01) | $\lambda_c$ (nv=0.05) |
|:---:|:---:|:---:|:---:|:---:|
| ch0 | LL3 | 3.636 | 0.003 | 0.014 |
| ch4 | L2-LL | 0.164 | 0.061 | 0.305 |
| ch16 | L1-LL | 0.083 | 0.120 | 0.601 |
| ch48 | L1-LL | 0.039 | 0.253 | 1.267 |
| ch63 | L1-HH | 0.058 | 0.174 | 0.869 |

> ch63의 threshold는 ch4의 2.8배. $\hat{\sigma}^2_{\text{noise}}$가 커질수록 이 비율은 유지되지만 **절대값이 모든 채널에서 비례 증가**.

---

## 3. Trade-off의 근원: 공간 불일치 문제

### 3-A. Diffusion 학습은 normalized space에서 수행

모델은 Z-score 정규화된 $\hat{\mathbf{z}}^{(c)} = (\mathbf{z}^{(c)} - \mu^{(c)}) / \sigma^{(c)}$에서 학습. 각 채널의 분산은 1로 균일화.

DDIM 10-step 스케줄의 마지막 timestep에서 잔류 노이즈 분산:

$$
\sigma^2_{\text{residual, norm}} \approx 1 - \bar{\alpha}_{\text{final}} \approx 0.01
$$

이 값은 **모든 채널에서 동일** (정규화 공간이므로).

### 3-B. BayesShrink는 raw space에서 적용

역정규화 후 raw wavelet 계수: $\mathbf{z}^{(c)} = \sigma_{\text{data},c} \cdot \hat{\mathbf{z}}^{(c)} + \mu^{(c)}$.

Raw space에서 실제 잔류 노이즈의 분산:

$$
\sigma^2_{\text{residual, raw}, c} = \sigma^2_{\text{data},c} \cdot \sigma^2_{\text{residual, norm}} = \sigma^2_{\text{data},c} \cdot (1 - \bar{\alpha}_{\text{final}})
$$

**채널마다 다르다.** 그러나 현재 구현은 단일 스칼라 $\hat{\sigma}^2_{\text{noise}}$를 모든 채널에 동일하게 사용.

### 3-C. 불일치의 구체적 영향

현재 threshold: $\lambda_c = \hat{\sigma}^2_{\text{noise}} / \sigma_{\text{data},c}$

"이상적" threshold (채널별 잔류 노이즈 반영):

$$
\lambda^*_c = \frac{\sigma^2_{\text{residual, raw}, c}}{\sigma_{\text{data},c}} = \frac{\sigma^2_{\text{data},c} \cdot (1 - \bar{\alpha}_{\text{final}})}{\sigma_{\text{data},c}} = \sigma_{\text{data},c} \cdot (1 - \bar{\alpha}_{\text{final}})
$$

두 threshold의 비교:

| | 현재: $\lambda_c$ | 이상적: $\lambda^*_c$ | 비율 $\lambda_c / \lambda^*_c$ |
|:---|:---:|:---:|:---:|
| **공식** | $\hat{\sigma}^2_{\text{noise}} / \sigma_{\text{data},c}$ | $\sigma_{\text{data},c} \cdot \epsilon^2$ | $\hat{\sigma}^2_{\text{noise}} / (\sigma^2_{\text{data},c} \cdot \epsilon^2)$ |
| **LL (ch0)**: $\sigma = 3.64$ | 0.003 | 0.036 | 0.08x (과소 shrink) |
| **HH (ch63)**: $\sigma = 0.058$ | 0.174 | 0.00058 | **300x (과대 shrink)** |

> [!IMPORTANT]
> **핵심 발견**: 현재 BayesShrink는 고주파 채널을 이상적 수준의 ~300배 과도하게 thresholding하고 있다.
> 이것이 noise_var 증가 시 LPIPS/FID가 열화되는 직접 원인이다.
> 반대로 LL 채널은 과소 thresholding되어 MSE가 충분히 내려가지 않는다.

---

## 4. Trade-off 해소 방안: Per-Channel Noise Variance

### 제안

단일 $\hat{\sigma}^2_{\text{noise}}$를 채널별 잔류 노이즈 분산 $\hat{\sigma}^2_{\text{noise},c}$로 대체:

$$
\hat{\sigma}^2_{\text{noise},c} = \sigma^2_{\text{data},c} \cdot \hat{\epsilon}^2
$$

여기서 $\hat{\epsilon}^2$는 정규화 공간에서의 잔류 노이즈 분산 추정치 (단일 스칼라).

이를 대입하면 threshold:

$$
\lambda^*_c = \frac{\hat{\sigma}^2_{\text{noise},c}}{\sigma_{\text{data},c}} = \frac{\sigma^2_{\text{data},c} \cdot \hat{\epsilon}^2}{\sigma_{\text{data},c}} = \sigma_{\text{data},c} \cdot \hat{\epsilon}^2
$$

### 기존 vs 제안 비교

| 속성 | 기존 ($\lambda_c \propto 1/\sigma_{\text{data},c}$) | 제안 ($\lambda^*_c \propto \sigma_{\text{data},c}$) |
|:---|:---|:---|
| LL 채널 (큰 $\sigma$) | 약한 shrink | 강한 shrink |
| HH 채널 (작은 $\sigma$) | **과도한 shrink** -> 텍스처 파괴 | 약한 shrink -> 텍스처 보존 |
| MSE 영향 | HH 노이즈 제거로 개선 | LL 노이즈 제거로 개선 (에너지 95%) |
| LPIPS/FID 영향 | HH 텍스처 파괴로 열화 | HH 보존으로 유지/개선 |

### 기대 효과

- **MSE 개선**: LL 채널(에너지 95%) shrink 강화 -> pixel error 감소
- **LPIPS/FID 유지**: HH 채널 과도 shrink 해소 -> perceptual quality 보존
- **Trade-off 해소 가능성**: MSE/SSIM 개선과 LPIPS/FID 유지가 동시 달성

### 구현

코드 변경 1줄: `bayesshrink.py` L104

```python
# 기존: thresholds[i] = noise_var / sigma_signal
# 제안: thresholds[i] = sigma_signal * epsilon_sq
thresholds[i] = sigma_signal * epsilon_sq  # epsilon_sq = CLI 파라미터
```

CLI 인자명을 `--bayesshrink_epsilon_sq`로 분리하여 기존 `--bayesshrink_noise_var`와 비교 가능하도록 설계.

---

## 5. 검증 계획

1. `epsilon_sq` 후보: [0.001, 0.005, 0.01, 0.02, 0.05]
2. Final_9G_3yr 체크포인트 + 2021 test set (기존 sweep과 동일)
3. 기존 sweep 결과와 Pareto front 비교 (MSE-FID, SSIM-LPIPS 2D plot)
4. Trade-off 해소 여부 판단: MSE/SSIM 개선하면서 FID가 열화되지 않는 점이 존재하는지

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-04-23 | 7-C Grid Search 결과 분석. 채널별 threshold 불일치 발견. Per-Channel Noise Variance 제안. |
