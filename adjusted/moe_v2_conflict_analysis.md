# MoE v2: Temperature Annealing × FiLM 충돌 분석

## 실험 결과 재해석

| 실험 | MSE | MAE | PSNR | SSIM | FID | LPIPS |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline (DC-6 3yr) | 0.0315 | 0.0997 | 22.31 | 0.5645 | 99.59 | 0.4346 |
| E1: Temp only | 0.0335 | 0.1043 | 22.01 | 0.5584 | **35.29** | **0.4112** |
| E2: Temp+FiLM | 0.0316 | 0.1015 | 22.25 | 0.5599 | 86.09 | 0.4308 |

유저 지적 핵심: **E2가 baseline과 거의 동일** → FiLM이 Temp Annealing의 FID 개선(-64pt)을 거의 완전히 상쇄.

---

## 충돌 메커니즘: 3중 구조적 간섭

### 1. FiLM이 Gate 학습을 "단락(short-circuit)"시킴

[decoder.py L244-252](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/decoder.py#L244-L252) forward 순서:

```
1. gate logits = gate(cond_emb)        ← cond_emb 사용
2. weights = softmax(logits / τ)
3. for k in K:
     x_k = x * (1 + γ_k) + β_k        ← FiLM: cond_emb 사용 (같은 입력!)
     expert_out[k] = Conv(x_k)
4. out = Σ weights[k] * expert_out[k]
```

**핵심 문제**: Gate와 FiLM이 **같은 `cond_emb`**을 입력으로 사용.

- **Gate**: `cond_emb` → "어떤 expert를 선택할지" 결정
- **FiLM**: `cond_emb` → "각 expert 입력을 어떻게 변조할지" 결정

Gate가 routing을 학습하려면, **expert 간 출력 차이가 입력의 inherent 차이(Conv weight 차이)에서 발생**해야 합니다.
그런데 FiLM이 `cond_emb` 기반으로 **입력 자체를 expert마다 다르게** 만들면:

> Gate 입장: "expert 출력 차이가 내 routing 때문인지, FiLM 변조 때문인지 구분 불가"
> → Gate gradient가 불안정해지고 routing 수렴 실패

### 2. Diversity Loss가 FiLM 효과를 "적대적으로" 억제

[train.py L539-542](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/engine/train.py) diversity loss:
```python
div_loss = _base_model.final_layer.get_moe_diversity_loss()
loss = loss + moe_div_lambda * div_loss  # λ=0.20
```

[decoder.py L567-575](file:///data/student1/docker_Solar/ST-DiT2/ST-DiT/stdit/models/decoder.py#L567-L575) diversity loss 계산:
```python
# expert 출력 간 cosine similarity → minimize
cos_sim = F.cosine_similarity(outs[i].flatten(1), outs[j].flatten(1))
```

**FiLM이 expert 입력을 다르게 만들면** → expert 출력도 자연히 달라짐 → cosine_sim 감소 → diversity loss 이미 낮음.

하지만 이것은 **Conv weight의 specialization 없이 달성된 "가짜 다양성"**.
FiLM γ,β가 충분히 발달하면, Conv weight는 동질적으로 유지되면서도 diversity loss가 만족됨.

> **결과**: FiLM이 diversity loss의 gradient를 흡수 → Conv weight specialization 동기 소멸
> → expert들이 사실상 "동일 Conv + 다른 FiLM 변조" = **FiLM이 없는 것과 동일한 앙상블 효과**

### 3. Temperature Annealing과 FiLM zero-init의 시간적 충돌

| Phase | τ (Temperature) | FiLM 상태 | 문제 |
|:-----:|:----:|:----:|------|
| Early (τ=5) | uniform routing | γ≈0, β≈0 (zero-init) | Gate/FiLM 모두 비활성 → 학습 신호 부족 |
| Mid (τ≈3) | 약간의 routing | FiLM 학습 중 | **FiLM gradient가 K개 expert 전체에 분산** (uniform weight이므로) |
| Late (τ=1) | sharp routing | FiLM 부분 학습 | Gate가 이미 수렴한 후에야 FiLM이 유의미 → **너무 늦음** |

Temp Annealing은 **early uniform → late sharp** 순서인데:
- **Early phase에서 FiLM gradient는 모든 expert에 동등하게 분배** (τ=5 → weight ≈ 1/K)
- 이는 FiLM 파라미터의 effective learning rate를 K배 희석
- 100 epoch 내에 FiLM이 충분히 학습되지 못함

> **E1(Temp only)**에서 FID가 좋은 이유: τ 감소 → sharp routing → expert 자체의 Conv 차별화 촉진
> **E2(Temp+FiLM)** 추가 시: FiLM이 "가짜 다양성"을 생성 → Conv 차별화 동기 약화 → baseline 수준으로 회귀

---

## 증거: E2 ≈ baseline 이유

E2의 모든 pixel 지표가 baseline과 거의 동일:
- MSE: 0.0316 vs 0.0315 (차이 0.3%)
- PSNR: 22.25 vs 22.31 (차이 0.3%)
- SSIM: 0.5599 vs 0.5645 (차이 0.8%)

이는 **FiLM이 Temp Annealing의 routing 학습을 완전히 상쇄**하여, 결국 baseline과 동일한 "uniform routing + 동질 expert" 상태로 수렴했음을 의미합니다.

---

## 수정 방안

### Option A: FiLM을 Gate 외부 조건으로 분리 (권장)

```python
# 현재: gate(cond_emb) + film_k(cond_emb) ← 같은 입력!
# 수정: gate(cond_emb) + film_k(gate_extra) ← 다른 조건!
# gate_extra = ctx_delta, ctx_temporal_std 등 이미 존재하는 보조 신호
```

Gate와 FiLM이 다른 정보를 기반으로 동작하면 gradient 간섭 해소.

### Option B: FiLM 적용 위치를 expert Conv 이후로 변경

```python
# 현재: x_k = FiLM_k(x) → Conv_k(x_k) ← 입력 변조
# 수정: y_k = Conv_k(x) → FiLM_k(y_k) ← 출력 변조
```

Conv weight가 먼저 specialization하고, FiLM은 후처리로 조건별 미세 조정.
Diversity loss가 Conv 출력 차이를 직접 보게 되어, Conv specialization 동기 유지.

### Option C: FiLM 제거 + E3/E4 결과 대기 (최소 개입)

현 데이터에서 **E1(Temp only)이 최선**이므로, FiLM을 제거하고 E3(Temp+LB), E4(Temp+LB, no FiLM) 결과로 Load Balancing 효과만 검증.

---

## 결론

> FiLM이 "routing을 방해"한 것이 아니라, **FiLM + Diversity Loss + Temp Annealing 3자가 구조적으로 충돌**하여 서로의 학습 신호를 상쇄한 것입니다.
>
> 핵심은 **gate와 FiLM이 같은 `cond_emb`를 공유**하는 설계 — 이것이 gate gradient를 오염시키고, diversity loss가 FiLM의 "가짜 다양성"을 학습하게 만드는 근본 원인입니다.
