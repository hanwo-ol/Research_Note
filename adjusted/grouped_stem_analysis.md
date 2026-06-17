# Grouped Stem 채널 정보 보존 엄밀 분석

## 결론: 채널 그룹 구조는 Transformer 내에서 소멸

Grouped Stem은 입력 임베딩 품질을 개선하지만, **Transformer 블록을 1개만 통과해도 그룹 구조가 완전히 혼합**됨.

---

## 엄밀 분석: 512dim을 혼합하는 연산 3가지

### 1. Self-Attention — QKV Linear가 그룹을 파괴

```python
# blocks.py L46-47
self.qkv = nn.Linear(hidden_size, hidden_size * 3)  # Linear(512, 1536)
self.proj = nn.Linear(hidden_size, hidden_size)      # Linear(512, 512)
```

`Linear(512, 1536)`는 **512dim 전체**에 대한 행렬곱:

```
Q[i] = sum_{j=0}^{511} W_Q[i,j] * x[j]
       ← dim 0~127(ch0)과 128~511(나머지)가 혼합
```

> 1번째 attention 이후 그룹 경계 소멸.

### 2. LayerNorm — 그룹 간 통계 공유

```python
# blocks.py L33
self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
```

LayerNorm은 **512dim 전체의 mean/std로 정규화** → 그룹 독립성 훼손.

### 3. FFN / MoE — 전체 dim 혼합

```python
nn.Linear(hidden_size, mlp_hidden_dim)  # Linear(512, 2048)
```

512dim 전체를 혼합. MoE 전문가도 동일.

### 출력 Linear — 그룹 무관

```python
nn.Linear(hidden_size, patch_size**2 * out_channels)  # Linear(512, 256)
```

단일 행렬곱으로 64ch을 한번에 출력 → 입력 그룹 구조가 출력에 반영 안 됨.

---

## 그룹 구조 생존도

```
입력 Grouped Stem    → 그룹 구조 존재 (ch0→dim[0:128], ...)
     ↓ LN (512 전체)  → 그룹 경계 약화
     ↓ QKV(512→1536)  → 그룹 구조 완전 소멸
     ↓ Attention       → 혼합 상태 유지
     ↓ FFN(512→2048)   → 재혼합
     ↓ x8 블록          → 완전히 혼합
출력 Linear(512→256) → 그룹 무관
     ↓ unpatchify      → 64ch 재구성 (그룹 정보 없이)
     ↓ IWT             → 1ch (정상 동작)
```

> **Grouped Stem의 실질적 효과**: 초기 임베딩 품질 개선만.
> ViT CNN Stem이 도움되는 것과 동일한 메커니즘.
> **채널 구조의 명시적 보존은 보장하지 않음**.

---

## 대안: 채널 구조를 실제로 보존하는 설계

### 대안 A: Grouped I/O + GroupNorm

- 입력: Grouped Stem
- 블록 내부: LayerNorm → GroupNorm(4, 512). 그룹별 독립 정규화
- 출력: Grouped Output Stem (dim[0:128]→ch0, 등)
- 문제: QKV/FFN의 전체 혼합은 여전히 존재 → 부분적 보존만

### 대안 B: Multi-Stream Transformer (가장 확실)

```
ch0    → Mini-Transformer(128dim, 2블록)  ─┐
ch1-3  → Mini-Transformer(128dim, 2블록)  ─┤→ Cross-Attn → output
ch4-15 → Mini-Transformer(128dim, 2블록)  ─┤
ch16-63→ Mini-Transformer(128dim, 2블록)  ─┘
```

- 그룹 보존 완전 보장. 스트림 간 Cross-attention으로 위상 조율.
- 연산 ~3-4배 증가.

### 대안 C: Grouped Residual Bypass (추천)

```python
class GroupAwareBlock(STDiTBlock):
    def __init__(self, ...):
        super().__init__(...)
 # 그룹별 독립 처리 경로
        self.group_conv = nn.Conv1d(512, 512, 1, groups=4)
        self.group_gate = nn.Parameter(torch.zeros(1))  # zero-init
    
    def forward(self, x, ...):
        x_main = super().forward(x, ...)       # 기존 (전체 혼합)
        x_group = self.group_conv(x.T).T       # 그룹 보존 경로
        return x_main + self.group_gate * x_group
```

- 기존 전체 혼합 유지 (cross-group 정보 흐름)
- 추가 잔차 경로가 그룹 내 구조 보조
- zero-init → 기존 성능 보장
- 구현 ~15줄/블록

---

## 추천

| 대안 | 그룹 보존 | 코드 변경 | VRAM | 난이도 |
|:---|:---:|:---:|:---:|:---:|
| Grouped Stem (기존) | 안됨 | ~50줄 | 동일 | 하 |
| **C. Grouped Residual** | **부분** | **~30줄** | **+5%** | **하** |
| A. Grouped I/O + GN | 부분 | ~80줄 | 동일 | 중 |
| B. Multi-Stream | 완전 | ~200줄 | x3-4 | 상 |
