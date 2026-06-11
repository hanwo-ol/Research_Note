# Deep 계열 — 실험 현황 + 성능 분석

> **최종 업데이트**: 2026-05-14 11:22 KST

---

## 1. 전체 실험 결과 비교표

### Core Experiments (baseline → 챔피언 경로)

| # | 실험 | MSE ↓ | MAE ↓ | SSIM ↑ | LPIPS ↓ | FID ↓ | 비고 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| 0 | r32n2 (old baseline) | 0.0566 | 0.1438 | 0.522 | 0.443 | 51.0 | spatial expert rank32×2 |
| 1 | **r16n8 baseline** | 0.04586 | 0.12677 | 0.5311 | 0.4194 | 35.96 | rank16×8 = 현재 base |
| 4 | SG step10 (0087) | 0.04485 | 0.12509 | 0.5328 | 0.4117 | 29.58 | ✅ 전 지표 개선, 학습 0 |

### IP γ Sweep (0088)

| # | γ | MSE ↓ | MAE ↓ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | **1.0** | 0.04438 | 0.12222 | 0.5344 | 0.4223 | 30.07 |
| 9 | **1.5** | 0.04266 | 0.12014 | 0.5368 | 0.4135 | 22.69 |
| 10 | **2.0** | 0.04255 | 0.12040 | 0.5370 | 0.4117 | 22.37 |
| 12 | **2.5** | **0.04240** | **0.12020** | **0.5373** | 0.4102 | 21.96 |
| 13 | 3.0 | 0.04409 | 0.12440 | 0.5368 | 0.4120 | 21.65 |

> Sweet spot: γ=2.5 (MSE ALL-TIME BEST), γ=2.0 (FID와의 균형)

### DDD + IP 결합 + Idea 2 (t-gate)

| # | 실험 | MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | 비고 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| D-60 | DDD λ=0.1 단독 | 0.0478 | 20.22 | — | 0.424 | 27.79 | |
| D-66 | IP γ=2.0 단독 | 0.0426 | 20.80 | — | 0.412 | 22.37 | |
| D-71 | DDD + IP γ=2.0 | 0.0441 | 20.67 | 0.537 | **0.399** | 12.27 | 이전 챔피언 |
| **NEW** | D-71 + **dyn_t_schedule=alpha_bar** | 0.0443 | 20.64 | 0.537 | 0.399 | **12.05** | 🏆 **현 챔피언** |

### Steps=40 안정화 (Idea 2 핵심 발견)

| ckpt | steps=10 FID | steps=40 FID | 변화 |
|---|---:|---:|---|
| D-71 (t-gate 없음) | 12.27 | **70.26** | +472% ❌ 폭발 |
| tab (alpha_bar) | **12.05** | **15.00** | +25% ✅ 안정 |
| tlin (linear) | 12.52 | 14.68 | +17% ✅ 안정 |

---

## 2. 핵심 발견

### ① IP γ Sweet Spot: γ=2.5

논문 기본값 γ=0.1은 실패. Residual space에서 **25배 큰 γ=2.5가 최적** (MSE −7.5%).

### ② DDD+IP 시너지: FID 12.27

두 인터벤션 단독 (27.79, 22.37)의 단순 합 이론치보다 훨씬 좋음 → **직교 메커니즘** 확인.

### ③ dyn_t_schedule=alpha_bar = 새 챔피언 FID 12.05

high-t에서 SPADE를 silence → step 누적 robustness 확보. **§5.2 가설 완전 입증**.

### ④ Idea 1 (dyn_input_noise) 기각

DynamicHead 입력에 noise 추가는 **모든 γ에서 D-71 갱신 실패**. §4.2 가설 기각.

### ⑤ steps=40 실용 가능

t-gate 없는 D-71은 steps=40에서 FID 70 (사용 불가). t-gate로 FID 15 (실용 가능).

---

## 3. 검증된 가설/기각된 가설

| 가설 | 상태 | 증거 |
|---|---|---|
| §4.2: DynamicHead drift가 FID 병목 | **기각** | Idea 1 sweep — γ_dyn 모든 값에서 갱신 실패 |
| §4.3: 공간+단계 sharpness 시너지 | **부분 입증** | D-71 FID 12.27 < 단독 합 |
| **§5.2: high-t SPADE silence 필요** | **완전 입증** | REF FID 70.26 + tab/tlin 안정성 |

---

## 4. 현재 최적 설정

### Pixel Accuracy 최적 (MSE/MAE 우선)

```
학습: r16n8 + IP γ=2.5, fine-tune 30ep
결과: MSE=0.04240, FID=21.96
```

### Visual Quality 최적 (FID/LPIPS 우선) 🏆

```
학습: DDD λ=0.1 + IP γ=2.0 + dyn_t_schedule=alpha_bar
ckpt 사슬: scratch100 → ft30(DDD) → ft30(IP) → ft30(tab)
결과: MSE=0.0443, FID=12.05, LPIPS=0.399
```

---

## 5. 다음 실험: E2E Scratch Combo (§10)

**핵심 질문**: 점진적 ft 190ep의 효과를 scratch 100ep에서 재현 가능한가?

```bash
$PY main.py \
  [기존 deep 옵션] \
  --use_dynamic_decoder --dynamic_lambda 10 \
  --input_perturbation 2.0 \
  --dyn_t_schedule alpha_bar \
  --epochs 100 \
  --exp_name ..._E2E
```

**기대**: FID < 12.27이면 점진적 ft 불필요 → 운영 단순화.

---

## 6. 구현 상태

| 기능 | 문서 | 코드 | 상태 |
|---|---|---|---|
| Self-Guidance (0087) | ✅ | ✅ | 완료 |
| Input Perturbation (0088) | ✅ | ✅ | 완료, γ=2.5 최적 |
| Dynamic Decoder (0085) | ✅ | ✅ | 완료 |
| dyn_input_noise (0090 Idea 1) | ✅ | ✅ | 완료, **기각** |
| dyn_t_schedule (0090 Idea 2) | ✅ | ✅ | 완료, **입증** |
| Residual-weighted dyn loss (0090 Idea 3) | 설계만 | ⬜ | 대기 |
| 2D flow + warp (0090 Idea 4) | 설계만 | ⬜ | 대기 |
| Canonical SPADE (0090 Idea 5) | 설계만 | ⬜ | 대기 |
