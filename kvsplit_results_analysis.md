# K/V Split Experiment Results

## 1. 결과 요약

| Variant | MAE | PSNR | SSIM | FID | ΔMAE vs baseline |
|---|---|---|---|---|---|
| **baseline** (SLD shared) | 0.1103 | 21.71 | 0.5408 | **48.67** | — |
| **v1** (D+L K-only) | 0.1086 | 21.82 | 0.5428 | 58.15 | -1.5% |
| **v2** (L K-only) ⭐ | **0.1072** | **21.90** | **0.5438** | 63.69 | **-2.8%** |
| **v1+P** (D+L K-only +P) | 0.1076 | 21.90 | 0.5436 | 64.27 | -2.4% |

## 2. 핵심 인사이트

### 2.1 MAE vs FID 트레이드오프

```
MAE    ← better     worse →
0.1072 ████████░░░░░ v2 (best MAE)
0.1076 █████████░░░░ v1+P
0.1086 ██████████░░░ v1
0.1103 ████████████░ baseline

FID    ← better     worse →
48.67  ████████░░░░░ baseline (best FID)
58.15  ██████████░░░ v1
63.69  ████████████░ v2
64.27  ████████████░ v1+P
```

> [!IMPORTANT]
> **K/V split은 MAE를 개선하지만 FID를 악화시킨다.** 이는 K-only 토큰이 pixel 정확도(attention key = "where to look")를 강화하지만, V 채널 부재로 생성 다양성이 줄어듦을 의미한다.

### 2.2 Lead 토큰의 K-bias 효과

- v1 (D+L K-only) vs v2 (L K-only): Lead만 K-only로 해도 **MAE 더 좋음** (0.1072 vs 0.1086)
- Dynamics(D)를 K-only로 보내면 오히려 **해로움** — D는 V 채널에서도 유용한 정보를 전달
- → **Lead = K-bias (위치 정보), Dynamics = KV 모두 필요 (변화 정보)**

### 2.3 Persistence 토큰 추가 (v1+P)

- v1+P vs v1: MAE 개선 (0.1086→0.1076)이지만 FID 악화 (58→64)
- v1+P vs v2: MAE 거의 동일 (0.1076 vs 0.1072)
- → Persistence 토큰이 새로운 정보를 거의 제공하지 못함 (이전 persistence 분석과 일치: 정확 일치율 0.04%)

### 2.4 baseline MAE 차이 (kvsplit vs 64-experiment)

- 64개 실험 `ctx_S_L_D`: MAE = **0.1062**
- kvsplit `baseline`: MAE = **0.1103**
- 차이 = +0.0041 (3.9%)

> [!WARNING]
> **같은 SLD 조건인데 kvsplit baseline이 왜 더 나쁜가?** 
> 가능성: 난수 시드 차이, 학습 조건 차이 (kvsplit 스크립트에만 있는 인자), 또는 다른 데이터 스플릿.
> 이 차이가 있으므로, v2의 MAE=0.1072는 원래 ctx_S_L_D의 0.1062보다 나쁠 수 있다.

## 3. 추가 개선 방향

### 방향 A: Task Arithmetic (기존 설계)
- `ctx_S_L_D` backbone + kvsplit v2의 K-bias 학습 효과를 task arithmetic로 합성
- `θ_merged = θ_base + α × (θ_v2 - θ_baseline)` 

### 방향 B: K-only + FID 보존
- v2에서 FID가 악화되는 원인: V 채널 다양성 손실
- **개선안**: Lead만 K-only하되, V 채널에 learnable residual을 추가
  - `V_lead = V_shared + δ_V` (작은 learnable offset)
  - K-only의 MAE 이점 + V 채널 보존

### 방향 C: TRL과 결합
- K/V split (MAE↓, FID↑) + TRL (persistence 방지)
- TRL이 FID 악화를 보상할 수 있는지 확인
- 스크립트 준비 완료: `scripts/run_trl_experiments.sh`

## 4. 모델 가중치 분석 (진행 중)

baseline vs v2 간 가중치 차이 분석으로:
- 어떤 레이어가 K/V split에 의해 가장 많이 변했는지
- backbone drift가 있는지 (task arithmetic 적용 가능성)
