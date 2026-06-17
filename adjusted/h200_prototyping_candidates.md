# H200 프로토타이핑 후보군

> **현황**: A6000에서 B1_BG_3yr_Full_D4 학습 중 (~50-70시간 소요). H200 자원 활용 가능.
> **비교 기준**: B1_BG 1yr (MSE 0.0358 / SSIM 0.5337 / FID 202.69)

---

## 후보 총괄 (우선순위순)

| 순위 | ID | 후보 | 출처 | 구현 상태 | 예상 소요 | 기대 효과 |
|:---:|:---:|:---|:---|:---:|:---:|:---|
| **1** | F1 | **Spatial Loss Masking** | 0040_F1 | 미구현 (~30줄) | ~3h | 합의 1순위였으나 미실험. 공간적 HF 가중치로 SSIM 직접 공략 |
| **2** | Phase 8 | **Huber Loss** | 0009_0001 | 구현 완료 | ~3h | v-prediction MSE를 Huber로 교체. outlier robust |
| **3** | Phase 8 | **Loss Truncation** | 0009_0005 | 구현 완료 (미실험) | ~3h | 상위 25% 어려운 sample 제외. Huber와 결합 가능 |
| **4** | B2 | **Timestep Curriculum** | 0040_B2 | 미구현 (~20줄) | ~3h | 학습 초반 high-noise, 후반 low-noise. 수렴 속도 +30% 보고 |
| **5** | F2 | **Timestep-dep HF Reweighting** | 0040_F2 | 미구현 (~30줄) | ~3h | low-noise timestep에서 HF 가중치 추가 |
| **6** | AuxL1 pos | **Aux L1 Block Position** | road_map 후순위 | 구현 완료 | ~3h x N | block 0 외 위치 탐색 (C, D, E 우선 5건) |
| **7** | AuxPix | **AuxPixSSIM 개선** (w 감소/warmup) | road_map 후순위 | 구현 완료 | ~3h | w=0.005~0.01 또는 warmup schedule |
| **8** | 0012_0006 | **Hash Routing Ablation** | Phase 11 | 미구현 | ~3h | 논문 RQ2 ablation 가치. ST-Aware Router 효과 정량화 |

---

## 상세 분석

### 1순위: Spatial Loss Masking (F1)

**출처**: [0040_jump_ideas.md](file:///home/oem/ST-DiT2/docs/0040_jump_ideas.md) F1, [0004_adaptive_schedule_meeting.md](file:///home/oem/ST-DiT2/docs/0004_adaptive_schedule_meeting.md)

**배경**: Adaptive Schedule 회의(2026-04-24)에서 Loss-side 공간 가중치가 1순위 합의안이었으나, 실제로는 Schedule-side(13C/13D)만 실험 후 전체 비채택 처리됨. **합의된 1순위가 아직 미실험.**

**구현 내용**:
- DWT latent에서 HF 채널의 에너지 맵 계산 (이미 latent에 내재)
- 에너지가 높은 공간 위치(구름 경계 등)에 더 높은 loss 가중치 부여
- per_channel_snr 위에 공간 차원 가중치 추가 (~30줄)

**기대**: 구름 경계 등 detail-rich 영역의 복원 정밀도 향상으로 SSIM 직접 공략. BrightnessGate(시간 조건부)와 직교적 개선축(공간 가중치)이므로 결합 시너지 기대.

**명령어** (P0 baseline 기반):
```
python main.py --exp_name H_SpatialMask_D4 \
  [P0 baseline 옵션] \
  --use_spatial_loss_mask
```

---

### 2순위: Huber Loss (Phase 8, 0009_0001)

**출처**: [0028_ideas_0009_review.md](file:///home/oem/ST-DiT2/docs/0028_ideas_0009_review.md)

**배경**: v-prediction의 기본 MSE loss를 Huber(또는 Charbonnier)로 교체. outlier gradient 폭발 방지. 구현 완료(`--loss_type huber --huber_delta 0.1`).

**기대**: Phase 2-B에서 pixel-space Charbonnier(MSE 0.098)가 L1(0.110) 대비 개선을 보인 전례. v-space에서도 유사 효과 기대.

**명령어**:
```
python main.py --exp_name H_Huber_D4 \
  [P0 baseline 옵션] \
  --loss_type huber --huber_delta 0.1
```

---

### 3순위: Loss Truncation (Phase 8, 0009_0005)

**출처**: [0028_ideas_0009_review.md](file:///home/oem/ST-DiT2/docs/0028_ideas_0009_review.md)

**배경**: batch 내 상위 25% 어려운 sample의 loss를 제외하여 noisy gradient 방지. B=4에서 1개 sample 제외.

**명령어**:
```
python main.py --exp_name H_Truncation_D4 \
  [P0 baseline 옵션] \
  --use_loss_truncation --truncation_ratio 0.25
```

**결합**: Huber + Truncation 이중 적용 실험도 가능 (`H_HuberTrunc_D4`).

---

### 4순위: Timestep Curriculum (0040_B2)

**출처**: [0040_jump_ideas.md](file:///home/oem/ST-DiT2/docs/0040_jump_ideas.md) B2

**배경**: arXiv:2403.10348에서 easy(high-noise) -> hard(low-noise) curriculum으로 수렴 속도 +30% 보고. 100 epoch 프로토타이핑에서는 효과 제한적일 수 있으나, 저비용(~20줄)으로 시도 가치 있음.

---

### 5순위: Timestep-dependent HF Loss Reweighting (0040_F2)

**배경**: 현재 per_channel_snr은 채널별 균일 schedule. low-noise timestep(디테일 복원 단계)에서 HF 채널 가중치를 추가로 높이면 SSIM 개선 가능. F1(공간 가중치)과 직교적 축(시간축 가중치).

---

### 6순위: Aux L1 Block Position Ablation

**배경**: 현재 aux_l1은 block 0(첫 블록)에만 부착. 최적 위치 미검증. road_map.md에 16개 조합 나열되어 있으나, 우선 A~F 5~6건만 실행.

**우선 실행 대상**:
- A: No Aux (baseline 재확인)
- C: block 1 단독
- D: block 2 단독
- E: block 0+1+2 (3개)
- F: block 3 단독

---

### 7순위: AuxPixSSIM 개선 (가중치 감소/warmup)

**배경**: 기존 w=0.02에서 SSIM -2.8%로 선방. w=0.005~0.01 감소 또는 epoch warmup schedule(초반 w=0 -> 선형 증가)로 gradient 간섭 완화 시 동등~개선 가능.

---

### 8순위: Hash Routing Ablation (0012_0006)

**배경**: ST-Aware Router를 랜덤 라우팅으로 교체. 논문 RQ2에서 "router가 의미 있는 전문화를 유도하는가?"에 대한 ablation 근거. 성능은 열화 예상이나 논문 가치 높음.

---

## 권장 실행 전략

A6000이 B1_BG_3yr_Full_D4 장기 학습에 점유된 동안, H200에서 1yr 프로토타이핑을 병렬로 실행:

```
H200 실행 순서 (각 ~3h, 1yr 분기 데이터):

1. H_SpatialMask_D4          (F1, 미구현 -> 구현 후 실행)
2. H_Huber_D4                (Phase 8, 구현 완료)
3. H_LossTrunc_D4            (Phase 8, 구현 완료)
4. H_HuberTrunc_D4           (2+3 결합)
5. H_HashRouting_D4           (0012_0006, 논문 ablation)
6. H_AuxL1_Block1_D4         (Aux position C)
7. H_AuxL1_Block2_D4         (Aux position D)
```

> [!IMPORTANT]
> 1순위(Spatial Loss Masking)와 BrightnessGate는 직교적 개선축(공간 vs 시간)이므로, 각각 독립 검증 후 결합 가능. 두 기법 모두 P0 초과 시 **B1_BG + SpatialMask 결합 3yr 학습**이 최종 SSIM 0.60 목표 달성 후보.

