# Swin-style Shifted Window Attention 설계

> ST-DiT Raw Pixel 파이프라인에 Swin Transformer의 Shifted Window Attention을 적용하여,
> patch_size를 16에서 8로 줄이면서도 연산 비용 증가를 2.1x 이내로 억제하는 설계안.

---

## 1. 현행 구조 vs 제안 구조

### 현행 (Global Attention, P=16)

| 항목 | 값 |
|:---|:---|
| 입력 해상도 | 512 x 512, C=1 |
| patch_size | 16 |
| 토큰 그리드 | 32 x 32 = 1,024 tokens |
| Attention 범위 | **전역** (1024 x 1024) |
| Attention GFLOPs / 4블록 | 8.6 |
| 공간 해상도 (토큰 1개) | 16 x 16 pixel |

### 제안 (Shifted Window Attention, P=8)

| 항목 | 값 |
|:---|:---|
| 입력 해상도 | 512 x 512, C=1 |
| patch_size | 8 |
| 토큰 그리드 | 64 x 64 = 4,096 tokens |
| 윈도우 크기 (w) | 8 x 8 = 64 tokens |
| 윈도우 수 | (64/8)^2 = 64개 |
| Attention 범위 | **윈도우 내** (64 x 64) |
| Attention GFLOPs / 4블록 | 18.3 (**2.1x**) |
| 공간 해상도 (토큰 1개) | 8 x 8 pixel |
| 유효 수용 영역 (4블록 후) | ~256 x 256 pixel |

---

## 2. 핵심 메커니즘

### 2.1 Window Partition

토큰 그리드(64x64)를 겹치지 않는 8x8 윈도우로 분할한다.

```
64x64 토큰 그리드
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ W0 │ W1 │ W2 │ W3 │ W4 │ W5 │ W6 │ W7 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ W8 │ W9 │W10 │W11 │W12 │W13 │W14 │W15 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│    │    │    │    │    │    │    │    │
│   ...  (총 64개 윈도우, 각 8x8=64 tokens)  │
└────┴────┴────┴────┴────┴────┴────┴────┘
```

- 각 윈도우 내에서 독립적으로 Self-Attention 수행
- 윈도우 간에는 정보가 직접 교환되지 않음

### 2.2 Shifted Window (교대 적용)

Swin의 핵심: **홀수 블록에서 윈도우를 (w//2, w//2) = (4, 4) 토큰만큼 이동**시켜 파티션을 재구성한다.

```
Block 0 (정규 윈도우):         Block 1 (Shifted 윈도우, shift=4):

┌────┬────┬────┐              ┌──┬─────┬─────┬──┐
│ W0 │ W1 │ W2 │              │  │     │     │  │
├────┼────┼────┤   torch.roll  ├──┼─────┼─────┼──┤
│ W3 │ W4 │ W5 │  ──────────> │  │ W0' │ W1' │  │
├────┼────┼────┤              ├──┼─────┼─────┼──┤
│ W6 │ W7 │ W8 │              │  │     │     │  │
└────┴────┴────┘              └──┴─────┴─────┴──┘
                               ↑ 경계 토큰이 인접 윈도우와 그룹됨
```

이 교대 적용으로:
- **Block 0**: 각 윈도우 내부의 국소 관계 학습
- **Block 1**: 인접 윈도우 경계의 토큰들이 같은 윈도우에 속하게 되어 **윈도우 간 정보 전파**
- **Block 2-3**: 반복 → 수용 영역이 점진적으로 확장

### 2.3 Attention Mask (Shifted Window 경계 처리)

Shift 후 이미지 경계에서 wrap-around가 발생한다. 서로 멀리 떨어진 토큰이 같은 윈도우에 들어오는 것을 방지하기 위해 **attention mask**를 적용한다.

```
Shifted 후 윈도우 내부:
┌─────────┐
│ A │  B  │   A: 원래 좌상 영역 토큰
│───┼─────│   B: 원래 우상 영역 토큰 (wrap-around)
│ C │  D  │   C: 원래 좌하 영역 토큰 (wrap-around)
└─────────┘   D: 원래 우하 영역 토큰

Attention mask:
  A→A: 허용    A→B: 차단    A→C: 차단    A→D: 차단
  B→A: 차단    B→B: 허용    B→C: 차단    B→D: 차단
  ...
```

- Mask는 **사전 계산** 가능 (입력 크기 고정 512x512)
- `register_buffer`로 저장하여 학습 중 재계산 불필요
- Flash Attention과 호환 시 별도 처리 필요 (아래 구현 섹션 참조)

---

## 3. DiT Block 통합 설계

현행 DiT Block 구조에 Window Attention을 삽입한다. Cross-Attention과 MLP는 변경 없음.

```
입력 x: (B, 4096, 512)   ← 토큰 시퀀스
           │
           ▼
    ┌─────────────────────────┐
    │   AdaLN Modulation      │   ← timestep/conditioning 주입 (변경 없음)
    └─────────────────────────┘
           │
           ▼
    ┌─────────────────────────┐
    │  [변경] Window Self-Attn │   ← 전역 → 윈도우 내 attention
    │  - reshape (B,64x64,D)  │
    │    → (B*64, 8x8, D)    │      64개 윈도우로 분할
    │  - shifted일 경우       │
    │    torch.roll + mask    │
    │  - attention 수행       │
    │  - reverse partition    │
    └─────────────────────────┘
           │
           ▼
    ┌─────────────────────────┐
    │   Cross-Attention       │   ← context 참조 (변경 없음)
    └─────────────────────────┘
           │
           ▼
    ┌─────────────────────────┐
    │   MLP / FFN             │   ← 변경 없음
    └─────────────────────────┘
           │
           ▼
    출력 x: (B, 4096, 512)
```

### Block별 Shift 패턴 (depth=4)

| Block | 윈도우 모드 | Shift | 수용 영역 (누적) |
|:---:|:---|:---:|:---:|
| 0 | Regular | 0 | 64x64 px |
| 1 | **Shifted** | (4,4) | ~128x128 px |
| 2 | Regular | 0 | ~192x192 px |
| 3 | **Shifted** | (4,4) | ~256x256 px |

---

## 4. Cross-Attention 설계 선택지

Window Self-Attention은 target 토큰 내부의 국소 관계를 처리한다. Context 프레임과의 Cross-Attention에는 두 가지 선택지가 있다.

### 선택지 A: Cross-Attention도 윈도우 내 (비용 억제 우선)

- Context 토큰도 동일한 윈도우 파티션 적용
- 각 target 윈도우는 대응하는 context 윈도우만 참조
- 비용: Self-Attention과 동일 (2.1x)
- 단점: context의 원거리 정보 참조 불가

### 선택지 B: Cross-Attention은 전역 유지 (정확도 우선)

- Self-Attention만 윈도우화, Cross-Attention은 전역 유지
- 비용: Self-Attn 2.1x + Cross-Attn ~10x → 종합 ~6x
- 장점: context 프레임 전체를 참조하여 구름 이동 추적 유지

### 선택지 C: Cross-Attention은 다운샘플된 context 사용 (절충)

- Context를 2x2 average pooling → 16x16 = 256 토큰으로 축소
- Cross-Attention: 4096 query x 256 key → 비용 적정
- 비용: Self-Attn 2.1x + Cross-Attn ~1.5x → 종합 ~2x
- 장점: 전역 context 참조 + 비용 억제

> [!IMPORTANT]
> **선택지 C를 권장**. 기상 예측에서 context의 전역 구조(구름 이동 방향)가 중요하며, 다운샘플된 context로도 충분한 공간 정보를 제공한다. 비용 증가는 최소.

---

## 5. 연산 비용 상세 비교

Self-Attention FLOPs = `4ND^2 + 2N^2D` (D=512)

### Attention만

| Config | Self-Attn | Cross-Attn (선택지 C) | 합계 | 현행 대비 |
|:---|:---:|:---:|:---:|:---:|
| 현행 (P=16, 전역) | 8.6 GF | 8.6 GF | 17.2 GF | 1.0x |
| 제안 (P=8, Swin, CA=C) | 18.3 GF | ~13 GF | ~31 GF | **1.8x** |

### 전체 (Attention + MLP + Embedding)

| 항목 | 현행 (P=16) | 제안 (P=8) | 비율 |
|:---|:---:|:---:|:---:|
| Embedding (Stem) | 0.5 GF | 0.5 GF | 1.0x |
| Self-Attention (4블록) | 8.6 GF | 18.3 GF | 2.1x |
| Cross-Attention (4블록) | 8.6 GF | ~13 GF | 1.5x |
| MLP (4블록) | 8.4 GF | 33.6 GF | 4.0x |
| Decoder (Head) | 0.5 GF | 0.5 GF | 1.0x |
| **합계** | **~26.6 GF** | **~65.9 GF** | **~2.5x** |

> [!WARNING]
> MLP도 토큰 수에 비례하여 4x 증가한다 (4096 vs 1024). Attention만 윈도우화해도 **MLP 비용이 지배적**이 되어 전체 비용은 ~2.5x. 순수 "Attention만 비교"하면 1.8x이지만 실질 학습 시간은 2.5배 정도로 예상.

---

## 6. 메모리 사용량 추정

| 항목 | 현행 (P=16) | 제안 (P=8) |
|:---|:---:|:---:|
| 토큰 텐서 (B=4) | 4 x 1024 x 512 = 2M | 4 x 4096 x 512 = 8M |
| Attention 행렬 | 4 x 8 x 1024^2 = 33.6M | 4 x 8 x 64 x 64^2 = 8.4M |
| MLP 중간 | 4 x 1024 x 2048 = 8.4M | 4 x 4096 x 2048 = 33.6M |
| **Peak VRAM 추정** | **~8 GB** | **~14 GB** |

> Attention 행렬 자체는 윈도우화로 오히려 축소되지만, 토큰 텐서와 MLP 중간값이 4배 증가하여 VRAM이 약 1.7배 증가. A6000 48GB에서 batch_size=4 유지 가능.

---

## 7. 구현 계획

### 파일 구조

```
stdit/models/
├── swin_attention.py          [NEW] Window partition, shift, attention
├── swin_dit_block.py          [NEW] Swin 기반 DiT Block
└── st_dit.py                  [MODIFY] config 분기 추가
```

### swin_attention.py 핵심 함수

1. `window_partition(x, window_size)`: (B, H, W, D) → (B*nW, w, w, D)
2. `window_reverse(windows, window_size, H, W)`: 역변환
3. `create_attention_mask(H, W, window_size, shift_size)`: shifted window용 mask 사전 계산
4. `SwinSelfAttention(nn.Module)`: 윈도우 분할 → attention → 역변환

### swin_dit_block.py

- 기존 `STDiTBlock`을 상속하여 Self-Attention만 교체
- AdaLN, Cross-Attention, MLP는 기존 코드 재사용
- `shift_size` 인자로 regular/shifted 전환

### st_dit.py 변경

- `--use_swin_attention --swin_window_size 8` CLI 플래그
- `self.blocks` 생성 시 block_idx에 따라 shift_size를 0 또는 w//2로 교대 설정

### config.py / main.py

- `USE_SWIN_ATTENTION` (bool)
- `SWIN_WINDOW_SIZE` (int, default=8)
- `--patch_size 8`과 조합하여 사용

---

## 8. 예상 효과와 위험

### 기대 효과

| 항목 | 기대 |
|:---|:---|
| 공간 해상도 | 16x16 px → 8x8 px (4배 세밀) |
| 패치 아티팩트 | 패치 크기 축소로 경계 간격이 절반 → 시각적 아티팩트 감소 |
| 세부 구조 복원 | 구름 가장자리, 작은 구름 등 세밀한 구조 포착 개선 |
| SSIM | 공간 구조 정보 증가로 개선 기대 |

### 위험 요소

| 항목 | 위험 |
|:---|:---|
| 학습 시간 | ~2.5배 증가 (epoch당 85초 → ~210초) |
| 전역 맥락 | depth=4에서 유효 수용 영역 256x256 (이미지 절반) → 대규모 구름 시스템 포착 제한 |
| torch.compile 호환 | dynamic shape(shifted window masking)이 compile과 충돌 가능 |
| 기존 모듈 호환 | PhysInput, special tokens(register) 등과의 통합 검증 필요 |

---

## 9. 대안 비교 (의사결정 참고)

| 방안 | 비용 증가 | 아티팩트 해소 | 세밀도 향상 | 구현 난이도 |
|:---|:---:|:---:|:---:|:---:|
| **Swin P=8** (본 설계) | 2.5x | 중 | 높음 | 중 |
| ConvDecoderHead (Tier3_07) | 1.1x | 높음 | 없음 | 낮음 (완료) |
| ms_ssim_l1 loss (Tier3_13) | 1.0x | 중 | 없음 | 낮음 (완료) |
| Persistence Noise (Tier3_10) | 1.0x | 낮음 | 없음 | 낮음 (완료) |
| FAL (Tier3_11) | 1.0x | 중 | 중 | 중 (진행중) |

> [!NOTE]
> Swin은 근본적 해상도 향상이지만 비용 2.5x + 구현 복잡도가 높다. 현재 FAL/PersistNoise/ms_ssim_l1 실험이 비용 무증가로 진행 중이므로, 그 결과를 확인한 후 Swin 도입 여부를 결정하는 것이 합리적이다.
