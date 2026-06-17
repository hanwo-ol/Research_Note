# t-gated Stem PoC 설계서 (v2 — 리뷰 반영)

## 변경 이력

| 버전 | 변경 | 근거 |
|------|------|------|
| v1 | AdaGN (GroupNorm + t-modulation) | 초안 |
| **v2** | **GroupNorm 제거, 순수 t-modulated affine** | GNorm 실패(FID 62.23) 확정. 실패한 것을 재시도하지 않음. 분리 설계 원칙 적용 |

---

## 문제 정의

진단 결과, 현재 Stem의 `_StemStage`는 `f(x_t)`로 timestep 정보 없이 동작.
t=999에서 token cosine similarity가 0.10으로, clean token과 거의 직교함.

**가설**: Stem이 t를 인지하면(`f(x_t, t)`), noise level에 맞게 feature 추출을 적응시켜 token 품질이 개선된다.

---

## 설계 원칙

### 코딩 규칙 준수 (AGENTS.md)

- **기존 코드 삭제 금지**: `_StemStage(V1)`, `_StemStageV2` 전부 유지
- **config로 변경점 비교**: `STEM_USE_T_GATE` 단독 플래그, 기존 플래그와 직교
- **모듈별 분리**: 변경은 stem.py 내부에 한정

### 직교 설계 (Orthogonal Design)

| config | 역할 | 독립성 |
|--------|------|--------|
| `STEM_USE_NORM` | GroupNorm 정규화 (실패 확정) | 독립 |
| `STEM_USE_RESIDUAL` | Residual shortcut | 독립 |
| `STEM_USE_T_GATE` | **t-modulated affine (신규)** | 독립 |

모든 조합이 유효하며 독립적으로 on/off 가능.

---

## Proposed Changes

### 1. stem.py — `_StemStageV3` 추가

```
흐름: x → Conv3x3(stride=2) → t_affine(scale, shift) → GELU
                                      ↑
                                t_emb → Linear(hidden_size → 2*c_out)
                                      zero-init
```

- GroupNorm **없음**
- zero-init 시: `h * (1+0) + 0 = h` → `_StemStage`(V1)의 `GELU(Conv(x))`와 수학적 동등
- Residual은 `STEM_USE_RESIDUAL` 플래그에 따라 독립 적용 (V3 자체는 미포함)

### V2와의 조합

`STEM_USE_T_GATE=True` + `STEM_USE_NORM=True` 동시 설정 시:
- Conv → GroupNorm(affine=False) → t_affine → GELU (DiT AdaLN 스타일)
- 이 조합은 2순위 실험으로 예약. PoC에서는 테스트하지 않음

### 2. stem.py — `HierarchicalDeepStem.__init__` 수정

`_use_v3 = use_t_gate` 조건 추가. stages path에서 `_StemStageV3` 사용.

`forward(self, x, t_emb=None)` 시그니처 확장:
- `t_emb=None` 기본값 → 기존 모든 호출(context_embedder 포함) 영향 없음
- V3 path에서만 `stage(x, t_emb)` 호출
- expose_features, dual_scale path도 V3일 때 `stage(x, t_emb)` 호출하도록 `_call_stage` helper 사용

### 3. st_dit.py — x_embedder 호출 시 t_emb 전달

현재 (line 1431):
```python
x = self.x_embedder(x_target)
```

변경:
```python
x = self.x_embedder(x_target, t_emb=t_emb)
```

- `HierarchicalDeepStem.forward`가 `t_emb=None` 기본값이므로, t_gate 비활성 시 무시됨
- **context_embedder에는 전달하지 않음** (clean 프레임, t와 무관)
- `t_emb`는 line 1270에서 이미 계산 완료. 순서 충돌 없음

### 4. config.py — 플래그 추가

DEFAULT_CONFIG:
```python
"STEM_USE_T_GATE": False,
```

CLI_TO_CONFIG:
```python
"stem_use_t_gate": "STEM_USE_T_GATE",
```

---

## 비교 대상 Baseline

### 실험 명령어

```bash
# Baseline (이미 학습/테스트 완료)
$PY main.py \
  --use_raw_pixel --hidden_size 512 --num_heads 8 \
  --prediction_type x0 --loss_weighting none \
  --loss_type charbonnier --charbonnier_eps 1e-3 \
  --target_years 2021 \
  --use_periodic_pixel_eval --pixel_eval_interval 10 \
  --cache_in_memory \
  --depth 4 --stem_type deep --decoder_type deep \
  --use_persistence_residual --residual_auto_stats 1024 \
  --use_freq_aware_learning --use_cosine_lr --warmup_epochs 10 --min_lr 1e-6 \
  --use_spatial_expert --spatial_expert_rank 16 --spatial_expert_num 8 --spatial_expert_blocks all \
  --use_dynamic_decoder --dynamic_lambda 10 \
  --input_perturbation 2.0 --dyn_t_schedule alpha_bar \
  --use_temporal_gradient_square \
  --epochs 100 --guidance_scale 1.0 \
  --patch_size 16 --batch_size 4 \
  --use_dwt_hf_loss --dwt_hf_loss_weight 0.1 \
  --exp_name patch16_dwthf_w01
```

```bash
# PoC (단일 변수 추가: --stem_use_t_gate)
$PY main.py \
  --use_raw_pixel --hidden_size 512 --num_heads 8 \
  --prediction_type x0 --loss_weighting none \
  --loss_type charbonnier --charbonnier_eps 1e-3 \
  --target_years 2021 \
  --use_periodic_pixel_eval --pixel_eval_interval 10 \
  --cache_in_memory \
  --depth 4 --stem_type deep --decoder_type deep \
  --use_persistence_residual --residual_auto_stats 1024 \
  --use_freq_aware_learning --use_cosine_lr --warmup_epochs 10 --min_lr 1e-6 \
  --use_spatial_expert --spatial_expert_rank 16 --spatial_expert_num 8 --spatial_expert_blocks all \
  --use_dynamic_decoder --dynamic_lambda 10 \
  --input_perturbation 2.0 --dyn_t_schedule alpha_bar \
  --use_temporal_gradient_square \
  --epochs 100 --guidance_scale 1.0 \
  --patch_size 16 --batch_size 4 \
  --use_dwt_hf_loss --dwt_hf_loss_weight 0.1 \
  --stem_use_t_gate \
  --exp_name patch16_dwthf_w01_tgate
```

### 단일 변수 차이 확인

| 항목 | Baseline | PoC |
|------|----------|-----|
| `--stem_use_t_gate` | 없음 (False) | **있음 (True)** |
| 나머지 모든 플래그 | 동일 | 동일 |
| batch_size | 4 | 4 |
| seed | default | default |
| epochs | 100 | 100 |

### Baseline 결과 (비교 기준)

| Metric | Baseline (patch16_dwthf_w01) |
|--------|------------------------------|
| MSE | 0.0361 |
| MAE | 0.1084 |
| PSNR | 21.51 |
| SSIM | 0.5460 |
| LPIPS | 0.4188 |
| FID | 53.43 |

---

## Verification Plan

### 1차: 학습 수렴 확인
- loss curve가 baseline과 유사한 속도로 감소하는지 확인
- `pixel_eval_interval=10`으로 학습 중 pixel-space 평가 추적

### 2차: 진단 스크립트 재실행
- `scripts/diagnose_stem.py`로 t-gated stem의 token cosine similarity 측정
- 기대: t=999에서 cosine sim > 0.20 (baseline 0.10)

### 3차: 테스트 metrics 비교
- 동일 조건으로 테스트 실행 후 MSE/PSNR/SSIM/FID 비교

### 성공 기준

| Metric | Baseline | 최소 목표 |
|--------|----------|----------|
| PSNR | 21.51 | >= 21.51 (악화 없음) |
| FID | 53.43 | <= 53.43 (악화 없음) |
| Token sim (t=999) | 0.10 | > 0.20 |

### 실패 시 대응
- PSNR 악화 -> t_proj의 learning rate 감쇠 (0.1x)
- 학습 불안정 -> t_proj zero-init 검증 (weight가 실제 0에서 시작하는지)
- 두 metric 모두 악화 -> t_gate의 적용 stage를 줄임 (마지막 stage만)
