# MoE v2 실험 종합 결과 (run_moe_v2.sh)

## 실험 설계

| ID | 구성 | 설명 |
|----|------|------|
| baseline | DC-6 (3yr) | MoE 없음, 기존 최적 설정 |
| **E1** | Temp only | τ=5→1 annealing만 |
| **E2** | Temp + FiLM | τ annealing + expert별 affine 변조 |
| **E3** | Temp + FiLM + LB | τ annealing + FiLM + Load Balancing |
| **E4** | Temp + LB | τ annealing + LB (FiLM 없음) |

공통 설정: K=5, xavier gate init, domain-conditioned gate, diversity λ=0.20, 3yr 데이터

---

## Overall 지표 비교

| 실험 | MSE ↓ | MAE ↓ | PSNR ↑ | SSIM ↑ | **FID ↓** | **LPIPS ↓** |
|------|:-----:|:-----:|:------:|:------:|:---------:|:-----------:|
| baseline | 0.0315 | 0.1019 | 22.31 | 0.5645 | 99.59 | 0.4346 |
| **E1** Temp | 0.0335 | 0.1055 | 22.01 | 0.5584 | **35.29** | **0.4112** |
| E2 Temp+FiLM | **0.0310** | **0.0986** | **22.38** | **0.5627** | 86.09 | 0.4308 |
| E3 Full | 0.0310 | 0.0986 | 22.38 | 0.5627 | 102.09 | 0.4382 |
| E4 Temp+LB | 0.0333 | 0.1052 | 22.07 | 0.5573 | 72.62 | 0.4224 |

---

## Δ (vs baseline) 분석

| 실험 | ΔMSE | ΔPSNR | ΔFID | ΔLPIPS |
|------|:----:|:-----:|:----:|:------:|
| **E1** Temp | +0.0020 | −0.30 | **−64.3** | **−0.023** |
| E2 Temp+FiLM | −0.0005 | +0.07 | −13.5 | −0.004 |
| E3 Full | −0.0005 | +0.07 | +2.5 | +0.004 |
| E4 Temp+LB | +0.0018 | −0.24 | −27.0 | −0.012 |

---

## 핵심 발견

### 1. E1 (Temp only) — 압도적 FID 최강

> [!IMPORTANT]
> FID **35.29** (baseline 대비 −64.3), LPIPS **0.4112** (−0.023)
> **MSE/PSNR는 약간 악화**되지만, perceptual quality에서 압도적 개선.

- τ=5→1 annealing만으로 expert routing 분화 성공
- 추가 모듈(FiLM, LB) 없이 최고 성능 → **단순성이 최선**

### 2. FiLM (E2, E3) — perceptual quality 파괴

| | FID | LPIPS |
|---|:---:|:---:|
| E1 (FiLM ✗) | **35.29** | **0.4112** |
| E2 (FiLM ✓) | 86.09 | 0.4308 |
| E3 (FiLM+LB) | 102.09 | 0.4382 |

- A1 분석에서 확인: inter-expert FiLM cos_sim **0.65~0.75** → 5개 expert에 **동일 변조**
- A5 분석: FiLM 적용 후 expert 출력 cos_sim **+0.01~0.02 상승** → 출력 동질화
- pixel MSE는 개선되지만(FiLM이 mean 보정), FID/LPIPS는 악화

### 3. LB Loss (E3, E4) — routing 분화 방해

| | FID | LPIPS |
|---|:---:|:---:|
| E1 (LB ✗) | **35.29** | **0.4112** |
| E4 (LB ✓) | 72.62 | 0.4224 |
| E2 (FiLM, LB ✗) | 86.09 | 0.4308 |
| E3 (FiLM+LB) | 102.09 | 0.4382 |

- LB Loss가 expert 사용 빈도를 **강제 균등화** → τ annealing의 sharp routing과 **충돌**
- E4(Temp+LB) FID 72.62 vs E1(Temp only) FID 35.29 → **LB가 FID를 2배 악화**
- E3(Full) FID 102.09 → FiLM + LB 동시 적용은 baseline과 동등 (개선 없음)

### 4. MSE vs FID Trade-off

```
              MSE 좋음 ←─────────────→ FID 좋음
                                         
E2/E3 ●─────────●                        
                  pixel 정확            perceptual 약함
                                         
             E4 ●──────────●             
                            중간         
                                         
                     E1 ●────────────────● 
                         pixel 약간 희생   perceptual 최강
```

> [!TIP]
> **E1의 전략**: pixel-level 정확도를 소폭 희생(MSE +6%)하면서 
> perceptual quality를 극적으로 개선(FID −65%). 
> 기상 예측에서 FID/LPIPS가 더 중요하다면 E1이 최적.

---

## 결론 및 다음 단계

### 확정 사항
- ✅ **Temp Annealing (τ=5→1)**: 유효. MoE v2의 핵심 기여.
- ❌ **FiLM**: 제거. inter-expert 동질적 변조로 expert 다양성 파괴.
- ❌ **LB Loss**: 제거. τ annealing의 sharp routing과 충돌.

### 진행 중 실험
- 🔄 **E5 (Temp + Channel Gate)**: `run_moe_v2_chgate.sh` 실행 중
  - FiLM 대신 Conv 출력 후 sigmoid channel gating
  - orthogonal init으로 expert별 채널 분화 유도
  - E1보다 FID 개선 여부 확인

### 향후 검토
- E5 결과에 따라 Channel Gate 채택/기각 결정
- E1 기반으로 추가 개선 방향 탐색 (diversity loss 조정, K 수 변경 등)
