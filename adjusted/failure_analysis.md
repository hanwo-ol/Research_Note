# Phase 6 R2/R3 + Phase 9 실패 분석

> P0 Baseline: MSE 0.0359, MAE 0.1142, SSIM 0.538, FID 215

---

## 1. SAM Optimizer 실패 분석 (T1b_SAM: SSIM 0.040)

### 1.1 결과

| 지표 | P0 | T1b_SAM | 변화 |
|:---:|:---:|:---:|:---:|
| MSE | 0.0359 | **0.3508** | +877% |
| MAE | 0.1142 | **0.4845** | +324% |
| SSIM | 0.538 | **0.040** | -93% |
| FID | 215 | **485** | +126% |

### 1.2 실패 원인 (코드 레벨)

**원인 1: SAM 2nd forward에서 aux loss 경로 불일치**

SAM의 second forward ([train.py:286-294](file:///home/oem/ST-DiT2/ST-DiT/stdit/engine/train.py#L286-L294))에서 `diffusion.p_losses()`만 호출하고, 그 내부에서 aux_dwt_loss, load_balance, per_channel_snr weight가 모두 포함되지만, **train.py의 외부 pixel loss 경로(MS-SSIM+L1 등)는 SAM 2nd forward에 포함되지 않음**. 이는 1st forward와 2nd forward의 loss landscape가 다른 함수를 최적화하는 결과를 초래.

단, T1b_SAM은 pixel loss 미사용이므로 이 문제는 T1b에는 직접 해당하지 않음. 그러나 **p_losses 내부의 aux_dwt_loss는 model.forward()를 호출해야 aux_dwt_preds가 생성**되므로, 2nd forward에서도 정상적으로 aux loss가 계산되어야 함. 실제로 p_losses 내에서 처리되므로 이 경로는 정상.

**원인 2 (핵심): per_channel_snr + v-prediction + SAM의 gradient 불안정**

SAM의 ascent step에서 파라미터가 `w + rho * grad/||grad||` 방향으로 perturbation됨. per_channel_snr weighting은 **16채널(DWT 2-level) 간 SNR 비율이 850x** (ch0 var=3.54, ch15 var=0.004)에 달하는 극단적 비대칭 가중치를 적용함.

> [!NOTE]
> 로그의 "ch63_var" 표기는 pipeline.py의 하드코딩 문자열 오류. 실제 channel_scales shape은 (1, 16, 1, 1)이며, 16채널 중 마지막 채널(ch15)의 분산을 가리킴.

이 환경에서 SAM first_step의 perturbation은 **고에너지 LL 채널(ch0~3)의 gradient가 전체 grad_norm을 지배**하여, 저에너지 HF 채널(ch4~15)의 perturbation 방향이 사실상 noise가 됨. 결과적으로 perturbed loss landscape에서의 2nd forward gradient가 **HF 채널에 대해 무의미한 방향**을 제시하여, 전체 학습이 붕괴됨.

수학적으로: per_channel_snr의 가중치가 채널별로 [0.01, 5.0] 범위일 때, `||grad||_2`는 고에너지 LL 채널이 지배. SAM의 perturbation `e_w = rho * grad / ||grad||`에서 `||grad||`가 LL 채널에 의해 결정되므로, HF 채널에 대한 perturbation magnitude가 상대적으로 과소. perturbed position에서의 gradient는 HF에 대해 원래 위치와 거의 동일하여 SAM의 "sharp minima 회피" 효과가 발현되지 않음.

**원인 3: v-prediction의 target이 혼합 구조**

v-prediction target은 `v = sqrt(alpha) * noise - sqrt(1-alpha) * x_start`. noise와 x_start의 선형 결합이므로, **target 자체가 timestep t에 따라 급격히 변화**. SAM의 perturbation 후 동일한 t에서 재계산하지만, v-target의 noise/signal 비율이 perturbation에 민감하게 반응하여 **gradient 방향 진동**을 유발.

**원인 4: torch.compile과의 비호환**

`torch.compile`이 적용된 모델에서 SAM의 `p.data.clone()` / `p.data = self.state[p]["old_p"]` 연산이 compiled graph를 파괴할 가능성. compiled 모델은 in-place 데이터 교체에 대해 graph recompilation을 유발하며, 매 step마다 recompilation overhead가 발생하여 실질적인 학습 불안정으로 이어질 수 있음.

### 1.3 결론

SAM은 CNN(AC U-Net)에서 효과적이지만, DWT 64ch DiT 환경에서는 다음 이유로 부적합:
1. per_channel_snr의 850x 비대칭이 SAM의 isotropic perturbation 가정을 파괴
2. v-prediction target의 t-dependent 혼합 구조가 perturbation 민감성을 증폭
3. torch.compile 환경에서 in-place weight 교체의 graph 파괴

### 1.4 T1c_GA4_SAM 확인 실험 (2026-04-29 완료)

T1c는 "GA4(effective batch=16)가 SAM의 gradient noise를 안정화시킬 수 있는가?"를 검증하는 최대 공정 비교 조건 실험.

| 지표 | P0 | T1a_GA4 | T1b_SAM | T1c_GA4+SAM |
|:---:|:---:|:---:|:---:|:---:|
| MSE | 0.0359 | 0.0408 | **0.3508** | **0.3525** |
| MAE | 0.1142 | 0.1352 | **0.4845** | **0.4887** |
| SSIM | 0.538 | 0.429 | **0.040** | **0.043** |
| FID | 215 | 267 | **485** | **491** |

**GA4 = Gradient Accumulation 4 steps** (batch_size 4 x accum 4 = effective batch 16).
AC U-Net의 batch_size 16과 동일한 조건을 DiT에서 재현하기 위한 설정.

통찰:
- GA4 단독(T1a)은 P0 대비 MSE +13.6%, SSIM -20.3%로 **유의미한 악화**. 프로토타이핑 규모(16.5K steps)에서 effective batch 증가는 총 gradient update 횟수를 1/4로 감소시키므로 under-training 발생.
- T1c(GA4+SAM)는 T1b(SAM only)와 **거의 동일한 수준의 완전 붕괴**. GA4가 gradient variance를 감소시켜도, per_channel_snr의 850x 비대칭에서 SAM perturbation이 HF 채널을 무작위화하는 문제는 해결 불가.
- **SAM 붕괴는 batch size 문제가 아니라 per_channel_snr + SAM의 구조적 비호환**임을 확정.
- Phase 9 SAM 관련 실험 3건(T1a/T1b/T1c) 전원 비채택 확정.

---

## 2. EMA 실패 분석 (EMA_D4: SSIM 0.160)

### 2.1 결과

| 지표 | P0 | EMA_D4 | 변화 |
|:---:|:---:|:---:|:---:|
| MSE | 0.0359 | **0.2042** | +469% |
| MAE | 0.1142 | **0.3832** | +236% |
| SSIM | 0.538 | **0.160** | -70% |
| FID | 215 | **455** | +112% |

### 2.2 구현 타당성 검증

EMA 구현 자체는 표준적이며 정상:
- `EMAHelper` 클래스: `shadow = decay * shadow + (1-decay) * param` 공식 정확
- 매 optimizer step 후 `ema_helper.update(model)` 호출 위치 정상
- 추론 시 EMA weight 로딩 로직 정상 (torch.compile 접두사 처리 포함)
- `get_shadow_state_dict()`의 `_orig_mod.` 접두사 제거도 정상

### 2.3 실패 원인 (다각적 분석)

**원인 1 (핵심): decay=0.9999 + 100 epoch + 165 step/epoch = 16,500 total steps에서의 EMA 지연**

EMA의 effective half-life는 `1 / (1 - decay)` = 10,000 steps. 전체 학습이 16,500 steps이므로, **EMA shadow가 학습의 전반부(~60%) 가중치에 고착**되는 구조적 문제가 있음.

구체적으로:
- Epoch 1~60 (step 0~9,900): 모델이 빠르게 수렴하는 구간. EMA shadow는 이 구간의 평균
- Epoch 60~100 (step 9,900~16,500): 모델이 fine-tuning하는 구간. EMA가 이 구간의 개선을 반영하려면 `(1-0.9999)^6600` 비율로 반영해야 하므로, **후반 학습의 기여가 지수적으로 억제**

DiT (Peebles & Xie, 2023)에서 decay=0.9999를 사용하지만, 그 논문의 학습은 **400K~7M steps**이므로 half-life 10K가 적정. 우리는 16.5K steps로 그 1/24~1/424 수준이므로, **decay=0.9999는 과도하게 보수적**.

> [!IMPORTANT]
> **Full data (3yr 전체)에서의 전망**: Train 약 9,800개, batch=4 기준 steps/epoch=2,460. 100 epoch 시 total=246K steps. EMA half-life(10K)는 전체의 4.1%로 적정 범위. 따라서 **full data 학습에서는 decay=0.9999가 정상 작동할 가능성이 높다**. 프로토타이핑(16.5K steps)에서의 실패가 full data로 전환 시 자연 해소될 수 있으므로, **full data 실험군에 EMA를 포함하는 것을 권장**.

**원인 2: Best Model 선택이 학습 weight 기반이나 추론은 EMA weight 사용**

[pipeline.py:707-714](file:///home/oem/ST-DiT2/ST-DiT/stdit/pipeline.py#L707-L714)에서 `val_loss`는 **학습 weight**로 계산. [pipeline.py:737-744](file:///home/oem/ST-DiT2/ST-DiT/stdit/pipeline.py#L737-L744)에서 `best_val_loss` 갱신 시 학습 weight 기준으로 best epoch를 선택하고, 그 시점의 EMA shadow를 함께 저장.

문제: Epoch 30에서 학습 weight의 val_loss가 최소라서 best model로 선택되었다면, 그 시점의 EMA shadow는 **Epoch 1~30의 이동 평균**이므로 Epoch 30의 실제 학습 weight보다 상당히 열등. 추론 시 이 열등한 EMA weight를 사용하므로 성능이 급격히 악화.

> [!NOTE]
> **EMA 표준 방식 (DiT, DDPM, Stable Diffusion)**: DiT 논문은 모든 보고 결과를 **EMA model weight로 평가**. 즉, best checkpoint 선택 자체가 EMA weight 기준의 FID로 이루어짐. 우리 구현은 "학습 weight로 val 선택 + EMA weight로 추론"이라는 **비표준 방식**. 표준 방식은 EMA weight로 val_loss를 계산하여 best epoch를 결정하고, 그 EMA를 추론에 사용.

**원인 3: cosine LR schedule과의 간섭**

cosine LR은 warmup(10 epoch) 후 점진적으로 감쇠. EMA는 모든 step에서 동일한 decay를 적용하므로, **높은 LR 구간(warmup 직후)의 불안정한 weight가 낮은 LR 구간의 안정적 weight와 동일한 비율로 혼합**됨. 이는 EMA shadow의 품질을 저하시킴.

### 2.4 개선 방안

| 방안 | 변경 | 기대 효과 | 우선순위 |
|:---|:---|:---|:---:|
| **Full data 실험 추가** | 3yr 전체 데이터(246K steps)에서 재실험 | step 수 15x 증가로 half-life 비율 정상화 | 높음 |
| EMA val 평가 (표준화) | EMA weight로 val_loss 계산 후 best 선택 | 비표준 로직 해소. DiT 원논문 방식 | 높음 |
| decay 감소 | 0.999 또는 0.995 (프로토타이핑 전용) | half-life 1K~200 steps. 후반 학습 반영 가능 | 중간 |
| Warmup decay | 학습 초기 decay=0.99 -> 점진적 0.9999 | 초기 불안정 weight 배제 | 낮음 |
| 최종 EMA 저장 | Epoch 100 종료 시 EMA shadow 별도 저장 + 추론 | last EMA가 best보다 나을 가능성 | 낮음 |

### 2.5 결론

EMA 구현 자체는 정상이지만, **decay=0.9999 + 16.5K steps(프로토타이핑)의 구조적 부정합**이 핵심 원인. best model 선택 로직도 비표준(학습 weight 기준 선택 + EMA 추론). **Full data(246K steps)에서는 half-life 비율이 4.1%로 정상화되므로, full data 실험군에 EMA를 포함하는 것을 권장**. EMA val 평가의 표준화(DiT 원논문 방식)도 병행 필요.

---

## 3. AuxPixSSIM_D4 "선방" 분석 (MSE 0.0381, SSIM 0.523)

### 3.1 이전 pixel loss 실험들과의 비교

| 실험 | 시점 | 경로 | SSIM | P0 대비 | 비고 |
|:---|:---:|:---|:---:|:---:|:---|
| E3a+AuxPixSSIM | Phase 2 | AuxL1 + AuxPixSSIM 동시 | 0.444 | -17.5% | 두 aux 간 gradient 간섭 |
| E12B_AuxPixSSIM | Phase 2 | AuxPixSSIM 단독 (l1 없음) | 0.517 | -3.9% | P0에 l1 aux가 빠짐 |
| **AuxPixSSIM_D4** | **Phase 6 R2** | **P0(l1 aux 포함) + AuxPixSSIM 추가** | **0.523** | **-2.8%** | **P0 위에 추가** |
| MSSSIMPixel_D4 | Phase 6 R2 | MS-SSIM+L1 pixel loss 교체 | 0.109 | -79.7% | 완전 붕괴 |

### 3.2 AuxPixSSIM_D4가 선방한 이유

**이유 1: AuxPixelHead의 분리된 파라미터 구조**

[aux_pixel_head.py](file:///home/oem/ST-DiT2/ST-DiT/stdit/models/aux_pixel_head.py)의 `AuxPixelHead`는 **depth//4 블록에서 spatial tokens를 탈취하여 별도의 Linear -> PixelShuffle -> IWT 경로**로 pixel 예측을 생성. 이 head는 메인 denoising 경로와 **별도의 파라미터**(proj: Linear 512->256)를 가지므로, pixel loss의 gradient가 메인 weight에 미치는 영향이 **auxiliary head의 gradient를 통해 간접적으로만 전파**됨.

핵심: AuxPixelHead가 depth//4 블록의 **output에 대해서만 gradient를 전파**하므로, 블록 1개의 representation만 pixel-aware하게 조정되고, 나머지 블록들은 여전히 noise prediction에 전념. 이것이 "분리된 감독"의 효과.

**이유 2: 가중치 0.02의 보수적 설정**

`aux_pixel_loss_weight = 0.02`로 매우 낮은 가중치. 메인 noise loss 대비 약 1/5~1/10 수준의 gradient magnitude이므로, 학습 궤적을 크게 왜곡하지 않음.

**이유 3: P0의 l1 aux가 이미 DWT 구조를 안정화**

Phase 6 R2의 AuxPixSSIM_D4는 **P0 baseline(l1 aux + DWT bias) 위에 추가**된 실험. P0의 l1 aux가 이미 중간 representation의 DWT 구조를 감독하고 있으므로, 추가된 pixel SSIM loss는 기존의 안정적 학습 궤적 위에서 **미세 조정**만 수행. 이것이 E3a(l1+pixSSIM 동시, SSIM 0.444)와 결정적으로 다른 점 -- E3a에서는 l1 aux의 구조 감독이 pixel SSIM과 동시에 시작되어 gradient 간섭이 발생했지만, P0는 이미 l1이 안정화된 상태에서 pixel SSIM이 추가.

**이유 4: SSIM loss의 구조적 특성 (1-SSIM)**

DifferentiableSSIM은 `1 - SSIM`을 반환. SSIM 자체가 [-1, 1] 범위이므로 loss 값은 [0, 2]. pixel-space에서 SSIM은 **구조적 유사도**를 측정하므로, MSE처럼 pixel-wise error를 직접 최소화하는 것이 아니라 **구조적 패턴 보존**을 유도. 이것이 MSSSIMPixel_D4(MS-SSIM+L1의 0.84 혼합)보다 훨씬 안정적인 이유 -- MSSSIMPixel은 **다중 스케일 downsampling + L1 혼합**이 v-prediction gradient를 완전히 파괴한 반면, 단순 SSIM은 단일 스케일에서만 작동하여 gradient 충돌이 제한적.

### 3.3 AuxPixSSIM_D4 vs MSSSIMPixel_D4: 왜 하나는 선방하고 하나는 붕괴했는가

| 측면 | AuxPixSSIM_D4 | MSSSIMPixel_D4 |
|:---|:---|:---|
| **loss 적용 위치** | 중간 블록의 auxiliary head (분리 경로) | 메인 noise prediction output (직접 경로) |
| **gradient 경로** | aux head -> 1개 블록의 output만 | 전체 모델 weight에 직접 |
| **W^TW 통과** | 통과하지만, 별도 head로 격리 | 통과하며, 메인 loss에 직접 합산 |
| **가중치** | 0.02 (보수적) | 0.1 (공격적) |
| **loss 구조** | 단일 스케일 SSIM (1-SSIM) | 다중 스케일 MS-SSIM (2x, 4x downsample) + L1 |
| **결과** | SSIM 0.523 (-2.8%) | SSIM 0.109 (-79.7%) |

MSSSIMPixel_D4의 치명적 실패 원인:
1. MS-SSIM의 다중 스케일 downsampling이 **DWT의 multi-resolution 구조와 충돌** (DWT 자체가 이미 multi-resolution)
2. 0.84 * MS-SSIM + 0.16 * L1 혼합이 **pixel loss weight 0.1에 의해 메인 noise loss를 압도**
3. **메인 prediction output에 직접 pixel loss**를 적용하므로, v-prediction target과 pixel target 간의 gradient 충돌이 전 레이어에 전파

### 3.4 개선 가능성

AuxPixSSIM_D4의 -2.8% SSIM 악화는 **개선 여지가 있음**:

| 방안 | 구체적 변경 | 기대 효과 | 로드맵 ID |
|:---|:---|:---|:---:|
| **가중치 감소** | 0.02 -> 0.005~0.01 | gradient 간섭 추가 감소 | AuxPix-1 |
| **Warmup 가중치 스케줄** | Epoch 0: w=0, 마지막 epoch: w=target, 선형/cosine 증가 | loss 급등 방지 + best val epoch 갱신 보장 | AuxPix-2 |
| **IWT gradient detach** | 예측 경로의 IWT 전에 detach 적용 | W^T 역전파 차단 (아래 상세 분석) | AuxPix-3 |
| **depth//4 -> depth//2** | 더 깊은 블록에서 pixel 예측 | 더 풍부한 representation 활용 | AuxPix-4 |
| **Latent SSIM 대체** | AuxPixelHead 대신 latent-space SSIM | W^TW 완전 회피 (Phase 10 실험) | - |

#### Warmup 가중치 스케줄 (AuxPix-2) 상세

학습 후반에 갑자기 pixel SSIM을 활성화하면 loss가 급등하여 best_val_loss가 갱신되지 않을 수 있음. 대안:
- **Epoch 0부터 마지막 epoch까지 가중치를 0에서 target까지 점진적으로 증가**
- 스케줄: 선형 `w(e) = w_target * e/E` 또는 cosine `w(e) = w_target * (1 - cos(pi*e/E)) / 2`
- 초기 epoch에서는 noise prediction 학습이 지배적이므로 pixel SSIM이 학습 궤적을 왜곡하지 않음
- 후반 epoch에서는 noise prediction이 안정화된 상태에서 pixel 구조 보존 신호가 점진적으로 주입
- best_val_loss 갱신이 자연스럽게 이루어짐 (loss 급등 없음)

#### IWT gradient detach (AuxPix-3) 효과 분석

현재 AuxPixelHead의 gradient 경로:

```
SSIM loss → pixel_pred(1ch, 512x512)
  → IWT (고정, gradient 통과) → PixelShuffle 출력(4ch, 256x256)
    → proj Linear(512→256) → spatial_tokens
      → Transformer Block (depth//4)
```

**detach를 적용하면**: `x = self.iwt(x.contiguous().detach())`

```
SSIM loss → pixel_pred(1ch, 512x512)
  → IWT (gradient 차단됨) ← detach 지점
    → PixelShuffle 출력(4ch, 256x256)  [gradient 도달 불가]
      → proj Linear  [gradient 도달 불가]
        → Transformer Block  [gradient 도달 불가]
```

이 경우 pixel SSIM loss의 gradient가 IWT에서 차단되어 **어떤 학습 가능 파라미터에도 gradient가 도달하지 않음**. 즉, 사실상 loss가 존재하지만 학습에 아무 기여를 하지 않는 dead loss가 됨. 따라서 IWT 출력에서의 detach는 무의미.

**대안적 detach 위치**: PixelShuffle 출력(latent 4ch)에서 detach 후 IWT 적용

```
SSIM loss → pixel_pred(1ch, 512x512)
  → IWT (gradient 통과, 단 W^T만 적용됨)
    → PixelShuffle 출력(4ch, 256x256).detach()  ← detach 지점
```

이 경우:
- SSIM gradient가 IWT의 W^T를 통과하여 **latent 공간에서의 gradient**로 변환됨
- 하지만 detach 때문에 이 gradient는 proj Linear나 Transformer에 전파되지 않음
- 결과: 역시 dead loss

**결론**: IWT gradient detach는 위치와 무관하게 실질적으로 **loss를 비활성화**하는 효과. pixel SSIM의 gradient를 활용하면서 W^TW 편향을 줄이려면, detach보다는 **Latent SSIM(Phase 10)**이 근본적 해결책. AuxPixSSIM 자체의 개선은 **가중치 감소(AuxPix-1)**와 **warmup 스케줄(AuxPix-2)**이 실질적으로 유효한 방안.

다만, **Phase 10의 Latent SSIM이 근본적으로 더 우수한 접근**:
- AuxPixSSIM: IWT 경유 -> W^TW 편향 존재 (비록 aux head로 격리되었지만)
- Latent SSIM: IWT 미경유 -> W^TW 완전 부재, 직접 latent 구조 감독

AuxPixSSIM의 선방은 **"W^TW 편향이 격리된 aux 경로에서는 치명적이지 않다"**는 것을 보여주며, 이는 역으로 **"메인 경로에서의 W^TW 편향은 치명적"**이라는 RQ3 Prop.6.1의 진단을 재확인하는 증거.

---

## 4. 종합 인사이트

### 4.1 DWT 16ch DiT에서의 학습 안정성 원칙

1. **per_channel_snr의 850x 비대칭(16ch, LL 4ch vs Detail 12ch)은 외부 개입에 극도로 민감**: SAM, GradAccum 모두 이 환경에서 역효과. 이 비대칭 가중치가 모델의 내재적 정규화 역할을 하고 있으며, 외부에서 gradient 통계를 교란하면 즉시 붕괴.

2. **Pixel-space loss는 "어디서, 얼마나" 적용하느냐가 핵심**: 
   - 메인 경로 직접 적용 -> 붕괴 (MSSSIMPixel: -80%)
   - 분리된 aux head + 저가중치 -> 선방 (AuxPixSSIM: -2.8%)
   - Latent-space 직접 적용 -> 최선의 기대 (Phase 10 검증 대기)

3. **EMA는 학습 길이에 맞는 decay 설정이 필수**: 프로토타이핑(16.5K steps)에서 decay=0.9999는 과도하나, **full data(246K steps)에서는 정상 작동 기대**. 프로토타이핑 한정으로 decay=0.999 권장, 또는 full data 전환 시 decay=0.9999 유지.

### 4.2 RQ3 Prop.6.1 실증 자료로서의 가치

AuxPixSSIM_D4의 "상대적 성공"과 MSSSIMPixel_D4의 "완전 실패"는 논문에서 다음 논증을 강화:

> "Pixel-space 구조 감독의 효과는 **W^TW gradient 편향의 격리 정도**에 비례한다. 
> 메인 denoising 경로에서의 직접 pixel loss는 치명적이지만 (MSSSIMPixel: SSIM 0.109),
> 분리된 auxiliary head를 통한 간접 pixel loss는 제한적 성능 저하에 그친다 (AuxPixSSIM: SSIM 0.523).
> 이것이 latent-space 직접 감독(Latent SSIM)의 이론적 우월성을 동기화한다."

---

## 5. 참고: 실험 시계열

| 시각 | 실험 | 상태 |
|:---|:---|:---|
| 04-28 17:40 | AuxPixSSIM_D4 완료 | ❌ SSIM 0.523 |
| 04-28 20:04 | MSSSIMPixel_D4 완료 | ❌ SSIM 0.109 |
| 04-29 00:46 | EMA_D4 + MinSNR_D4 완료 | ❌ SSIM 0.160/0.494 |
| 04-29 03:02 | T1a_GA4 완료 | ❌ SSIM 0.429 |
| 04-29 06:51 | T1b_SAM 완료 | ❌ SSIM 0.040 |
| 04-29 ~09:00 | T1c_GA4_SAM 학습 중 | Epoch 41/100 |
