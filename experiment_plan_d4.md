# Depth-4 비교 실험 계획

> **기준선 (Baseline)**: 11B (Depth 4, 1yr, 9G config)
> **기준 성능**: MSE 0.0401 | SSIM 0.508 | FID 212

---

## 실험 설계 원칙

1. **Single-variable 원칙**: 각 실험은 11B baseline에서 **한 가지 기능만** 추가하여 해당 기능의 독립적 효과를 측정한다.
2. **누적 실험**: 개별 효과가 확인된 기능들을 순차적으로 누적하여, 기능 간 상호작용을 검증한다.
3. **모든 명령어는 baseline 명령어 기반**: 변경점만 플래그로 추가한다.
4. **test_only.py에는 반드시 `--prediction_type v` 명시**.

---

## Phase 1: 개별 기능 검증 (Single-variable)

각 기능을 11B baseline에 **단독으로** 추가하여 기여도를 측정한다.

| 실험 ID | 기능 | 추가 플래그 | 기대 | 비고 |
|:---:|:---|:---|:---|:---|
| E1 | 9-C: Xavier Conv Init | `--conv_init_type xavier --conv_init_alpha 0.01` | Conv expert 초기 학습 가속 | 기존 Zero-init 대비 |
| E2 | 9-D: MLP Ratio 8 | `--mlp_ratio 8` | FFN 용량 2배 확장 | VRAM ~16GB 예상 |
| E3 | 12-A: Dual Aux DWT | `--use_aux_dwt_loss` | gradient flow 개선 | 학습 전용 (추론 무영향) |
| E4 | 13-A: Freq Residual | `--use_freq_residual` | 고주파 보존 강화 | 최저 비용 (~5줄) |
| E5 | 13-B: DWT Channel Bias | `--use_dwt_channel_bias` | DWT 계층 구조 주입 | 64 params |
| E6 | 16A: PhysInput (V, SG) | `--use_phys_input --phys_features V SG` | 물리 맥락 주입 | 연산 +20% |

---

## Phase 2: 누적 조합 실험 (Stacking)

Phase 1에서 성능이 향상된 기능들만 선별하여 순차 누적한다. 추가 순서는 기대 효과와 비용을 고려하여 결정한다.

> [!IMPORTANT]
> Phase 1의 6개 실험 결과가 나온 후, 성능 향상이 확인된 기능만 Phase 2에 진입한다. 성능이 악화되거나 무의미한 기능은 제외한다.

**예시 누적 순서** (Phase 1 결과에 따라 변경 가능):

| 실험 ID | 누적 구성 | 목적 |
|:---:|:---|:---|
| S1 | 13-A + 12-A | Freq Residual + Aux DWT (상보적 설계) |
| S2 | S1 + 16A | S1 + PhysInput |
| S3 | S2 + 9-C | S2 + Xavier Conv Init |
| S4 | S3 + 13-B | S3 + DWT Channel Bias |
| S5 | S4 + 9-D | 전체 조합 (MLP Ratio 8) |

---

## Phase 3: PhysInput Ablation (Phase 1의 E6이 효과 있는 경우)

| 실험 ID | 피처 | Gate | N | 목적 |
|:---:|:---|:---:|:---:|:---|
| E6 | V, SG | X | 2 | 기본 (Phase 1에서 수행) |
| E7 | V, A, SG, CSI | X | 4 | 피처 수 확장 효과 |
| E8 | V | X | 1 | 속도 맵 단독 효과 |
| E9 | V, SG | O | 2 | Gate 발전안 검증 |

---

## 명령어 템플릿

모든 실험은 아래 baseline에서 추가 플래그만 append한다:

```bash
cd /home/oem/ST-DiT2/ST-DiT && python main.py \
  --exp_name "<실험명>" \
  --depth 4 \
  --use_freq_moe --asymmetric_low_stem --use_dwt \
  --target_years 2021 \
  --inference_timesteps 10 \
  --use_channel_scale --prediction_type v \
  --loss_weighting per_channel_snr \
  --num_moe_experts 8 --use_shared_expert --moe_top_k 2 \
  --moe_capacity_factor 1.5 --use_st_aware_router \
  --use_hetero_experts \
  --use_bayesshrink --bayesshrink_noise_var 0.01 \
  --use_cosine_lr --warmup_epochs 10 --min_lr 1e-6 \
  --batch_size 4 \
  --use_periodic_pixel_eval --pixel_eval_interval 10 --pixel_eval_steps 5 \
  <추가 플래그>
```

### 각 실험 명령어 (추가 플래그만)

| 실험 | exp_name | 추가 플래그 |
|:---:|:---|:---|
| E1 | `E1_XavierConv_D4` | `--conv_init_type xavier --conv_init_alpha 0.01` |
| E2 | `E2_MLPRatio8_D4` | `--mlp_ratio 8` |
| E3 | `E3_DualAux_D4` | `--use_aux_dwt_loss` |
| E4 | `E4_FreqRes_D4` | `--use_freq_residual` |
| E5 | `E5_DWTBias_D4` | `--use_dwt_channel_bias` |
| E6 | `E6_PhysInput_D4` | `--use_phys_input --phys_features V SG` |

---

## 결과 기록 형식

| 실험 | MSE | MAE | PSNR | SSIM | LPIPS | FID | vs 11B |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 11B (baseline) | 0.0401 | 0.1286 | 20.99 | 0.508 | 0.672 | 212 | - |
| E1 | | | | | | | |
| E2 | | | | | | | |
| E3 | | | | | | | |
| E4 | | | | | | | |
| E5 | | | | | | | |
| E6 | | | | | | | |

---

## 실행 순서 권장

```
E4 (Freq Residual)  -- 최저 비용, 빠른 확인
  |
E3 (Dual Aux DWT)   -- 독립적, gradient flow 개선
  |
E5 (DWT Channel Bias) -- 독립적, 64 params
  |
E1 (Xavier Conv)     -- Conv expert 관련
  |
E6 (PhysInput)       -- 신규 기능, 연산 +20%
  |
E2 (MLP Ratio 8)     -- VRAM 최대, 마지막 실행
```

이유: 비용이 낮고 독립적인 기능부터 실행하여, 초기에 성능 트렌드를 파악한다. PhysInput은 신규 코드이므로 단독 검증이 선행되어야 한다. MLP Ratio 8은 VRAM이 가장 높아 마지막에 배치한다.
