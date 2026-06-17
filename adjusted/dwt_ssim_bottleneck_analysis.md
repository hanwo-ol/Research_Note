# DWT vs VAE: 왜 MSE는 좋은데 SSIM은 낮은가?

## 현상

| 모델 | in_ch | hidden | depth | MSE ↓ | **SSIM** ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **9G (DWT, 8E+BS+k2)** | 64 | 512 | 8 | **0.038** ✅ | 0.515 |
| VAE-DiT (solar-diffuse) | 4 | 768 | 12 | 0.049 | **0.535** ✅ |

> MSE에서 **22% 우세**한 DWT 모델이 SSIM에서 **4% 열세**. 왜?

---

## 원인 1: Patch Embedding의 정보 밀도 비대칭 ()

### DWT의 구조적 핸디캡

```
DWT:  64ch × (2×2)px = 256 values → Linear(256, 512) → 512dim
VAE:   4ch × (2×2)px =  16 values → Linear(16,  768) → 768dim
```

| 지표 | DWT | VAE |
|:---|:---:|:---:|
| Patch 입력 차원 | 256 | 16 |
| Hidden 차원 | 512 | 768 |
| **Expansion ratio** | **2.0×** (압축!) | **48.0×** (확장!) |
| 입력 값 당 hidden capacity | 2 dim/val | 48 dim/val |

> [!IMPORTANT]
> **DWT는 Patch Embedding에서 정보를 "압축"하고, VAE는 "확장"한다.**
> 
> DWT의 256개 입력값을 512개 hidden dim에 매핑하면 실질적으로 **2:1 압축**.
> VAE의 16개 입력값을 768개 hidden dim에 매핑하면 **48:1 확장** — 값 하나당 48배의 표현 용량.

### 왜 이것이 SSIM에 영향하나?

SSIM은 **국소 구조적 패턴의 일치도**를 측정. 구조를 정확히 복원하려면 64개 채널 간의 **위상 관계(phase coherence)**가 정밀해야 함.

```
DWT 토큰 1개 = [LL, HL, LH, HH, ...] × 4px = 256개의 서로 다른 주파수 계수
                ↓ Linear(256→512)
                🔀 64개 sub-band 정보가 512dim에 혼합
                ↓ 8블록 Transformer
                ↓ Linear(512→256)
                🔀 256개 값으로 복원 → 64개 sub-band의 위상 관계 재구성 필요
```

문제: 64개 sub-band 계수가 단일 토큰의 512dim에 **섞인 채로 처리**됨.
Transformer가 이 혼합된 표현에서 64개 채널 간의 정밀한 위상 관계를 학습해야 하지만,
**expansion ratio 2×로는 이 관계를 인코딩할 용량이 부족**.

반면 VAE: 4개 채널만 분리 → 위상 관계가 훨씬 단순 → 구조 보존 용이.

---

## 원인 2: Inter-Channel Phase Coherence의 비선형 복잡도 ()

IWT(역변환)는 64채널을 **동시에 조합**하여 1채널 512×512를 복원:

```
64ch → IWT Level3 → 16ch → IWT Level2 → 4ch → IWT Level1 → 1ch
```

각 IWT 레벨에서 4개 채널(LL,LH,HL,HH)이 **가감산으로 간섭**:
```
pixel[i,j] = LL + LH + HL + HH  (해당 위치)
pixel[i,j+1] = LL - LH + HL - HH
...
```

**MSE가 낮다 = 각 채널의 평균 오차가 작다**
하지만 SSIM이 낮다 = **채널 간 상대적 위상이 틀림** → 간섭 패턴이 왜곡

비유:
- 64명의 악기 연주자가 각자 음정을 잘 맞추지만(MSE↓), **앙상블 타이밍이 맞지 않으면**(phase misalignment) 합주 결과(SSIM)는 어색함.
- 4명의 연주자(VAE)는 앙상블 조율이 훨씬 쉬움.

### 채널 간 phase error의 SSIM 영향 시뮬레이션

```
64채널에서 각 채널 MSE = 0.0006 (총 MSE = 0.038)
만약 ch0(LL)과 ch1(HL)의 위상이 1px 어긋나면:
  IWT 후 pixel = (LL + HL_shifted) → 경계에서 ringing artifact
  → MSE는 거의 변하지 않지만, local contrast 왜곡 → SSIM 급감
```

---

## 원인 3: BayesShrink의 구조 정보 손실 ()

BayesShrink가 SSIM을 보호하는 역할을 하지만, soft-thresholding은 **작은 detail 계수를 0으로 축소** → 미세 구조 소실.

```
BS 없음 (9D): SSIM = 0.481   
BS 있음 (9G): SSIM = 0.515  (+0.034)
               하지만 여전히 VAE의 0.535에 미달
```

> BayesShrink는 노이즈 제거에는 탁월하나, **유효한 미세 구조까지 같이 제거**하는 보수적 측면이 있음.

---

## 개선 방안 (actionable)

### 방안 1: Hidden Size 확대 (, 가장 직접적)

```diff
- HIDDEN_SIZE: 512   (expansion ratio 2.0×)
+ HIDDEN_SIZE: 1024  (expansion ratio 4.0×)
```

| Config | Hidden | Exp.Ratio | Params | 기대 효과 |
|:---|:---:|:---:|:---:|:---|
| 현재 9G | 512 | 2.0× | ~30M | SSIM=0.515 |
| 제안 | **1024** | **4.0×** | ~100M | SSIM 향상 기대 |
| solar-diffuse | 768/16=48× | 48× | — | SSIM=0.535 |

> **근거**: 256→512는 sub-band 간 위상 관계를 인코딩하기에 용량 부족.
> 1024로 확장하면 expansion ratio 4× → 채널 간 관계 학습 여유 증가.
> **비용**: 파라미터 ~3배, 학습 시간 ~2배. 하지만 SSIM 개선의 가장 직접적 경로.

### 방안 2: 2-Level DWT (, 구조적 해법)

```diff
- DWT3Level: 64ch × 64×64,  patch=2 → 토큰 1024, dim 256
+ DWT2Level: 16ch × 128×128, patch=4 → 토큰 1024, dim 256  (동일!)
```

| 항목 | 3-Level (현재) | 2-Level (제안) |
|:---|:---:|:---:|
| 채널 수 | 64 | **16** |
| Spatial | 64×64 | 128×128 |
| Patch Size | 2 | **4** |
| Patch dim | 64×4=256 | 16×16=**256** (동일!) |
| **Token 수** | **1024** | **1024** (동일!) |
| 위상 관계 복잡도 | 64채널 | **16채널 (4×단순)** |
| Lossless | ✅ | ✅ |

> [!TIP]
> **2-Level DWT + Patch 4는 토큰 수와 차원이 현재와 완전 동일하면서, 채널 간 위상 복잡도를 4배 줄인다.**
> 이것은 단순히 채널을 버리는 것이 아니라, **구조적으로 더 적은 DWT 분해 레벨을 사용하는 것** → lossless 유지.

### 방안 3: Channel-Grouped Patchification (, 중간안)

64채널을 4그룹(16ch씩)으로 나누어 각각 patchify → 별도 토큰 스트림:

```python
# 현재: flatten all 64ch into one token
token = Linear(64 * 2 * 2, 512)  # 256 → 512

# 제안: group by DWT level
group_LL  = Linear(4  * 2 * 2, 512)  # ch 0-3   (LL chain)
group_Mid = Linear(12 * 2 * 2, 512)  # ch 4-15  (Level 2 detail)  
group_High= Linear(48 * 2 * 2, 512)  # ch 16-63 (Level 1 detail)
# 같은 공간 위치의 3토큰을 cross-attend
```

> 각 그룹 내에서는 동질적 주파수 → 위상 관계 단순.
> 그룹 간은 cross-attention으로 조율 → 병렬 처리 가능.

### 방안 4: BayesShrink 임계값 최적화 (, 즉시 가능)

현재 `noise_var=0.01` 고정 → grid search로 SSIM 최적점 탐색.

```python
for nv in [0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.05]:
    test_only --bayesshrink_noise_var {nv}
```

> 임계값이 낮을수록 structure 보존↑ but noise 잔류↑.
> MSE-SSIM Pareto frontier 탐색.

### 방안 5: Adaptive BayesShrink ()

고정 threshold 대신, **SSIM을 feedback으로** noise_var을 자동 결정:

```
for each test sample:
  binary search noise_var to maximize SSIM(IWT(shrink(pred, nv)), gt)
```

Train set에서 optimal nv의 분포 통계 → test에 적용.

---

## 추천 실행 순서 (논문 기한 고려)

| 순위 | 방안 | 소요 | 기대 Δ SSIM | 코드 변경 |
|:---:|:---|:---:|:---:|:---:|
| **1** | BS noise_var grid search | **30분** | +0.01~0.03 | **0줄** |
| **2** | 2-Level DWT (P=4) | **4시간** | +0.02~0.05 | **~20줄** |
| **3** | Hidden 1024 | **8시간** | +0.01~0.03 | **1줄** |
| 4 | Channel-Grouped Patch | 1일 | 미지수 | ~100줄 |
| 5 | Adaptive BS | 2시간 | +0.01 | ~30줄 |

> [!IMPORTANT]
> **방안 2 (2-Level DWT)가 가장 근본적인 해법.**
> 동일 토큰/차원에서 채널 복잡도만 4배 감소 → SSIM 향상의 구조적 보장.
> Lossless 유지 + 코드 변경 ~20줄(DWT2Level 클래스 + config).
