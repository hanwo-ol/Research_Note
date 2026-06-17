# Deep Decoder 시각적 분석 및 Sharpness 복원 전략

## 1. 이미지 시각적 분석 (Visual Inspection)

세 가지 실험 모델의 예측 결과를 동일한 타임스텝(`2021-01-24_0300`) 기준으로 비교 분석했습니다.

### 분석 결과
1. **Tier 3 17 (기존 Linear Unpatchify)**
   - **장점**: 구름의 미세한 경계와 텍스처(고주파 성분)가 꽤 날카롭게(sharp) 살아있습니다.
   - **단점**: 패치(Patch) 단위로 분할하여 처리한 흔적인 모자이크/체커보드 아티팩트(Grid blockiness)가 시각적으로 확연히 나타납니다.
2. **Tier 3 20 (Deep Decoder 도입 계열)**
   - **장점**: 모자이크 아티팩트가 완벽하게 사라졌습니다. 구조적 연속성이 크게 개선되었으며 이는 SSIM 상승(0.566)의 주요 원인입니다.
   - **단점 (문제점)**: 전체적으로 강한 가우시안 블러(Gaussian Blur)를 먹인 것처럼 텍스처가 뭉개져 있습니다. **Sharpness가 완전히 상실**되었으며, 이로 인해 실제 기상 이미지의 통계적 분포와 멀어지면서 FID 수치가 급등(155 → 300)했습니다.

---

## 2. 왜 Sharpness를 잃었는가? (원인 분석)

현재 `HierarchicalDeepDecoder`는 공간 해상도를 키우기 위해 **Bilinear Upsampling + $3\times3$ Convolution**을 연속적으로 사용하고 있습니다.
Bilinear 보간법은 인접 픽셀 간의 가중 평균을 통해 픽셀을 생성하므로 태생적으로 **Low-pass Filter(저주파 통과 필터)** 역할을 하게 됩니다. 네트워크가 $3\times3$ Conv로 고주파를 다시 만들어내려 해도, 이미 Bilinear로 인해 정보가 뭉개진 상태라 복원에 한계가 있습니다.

---

## 3. Sharpness 복원 전략 (Architectural Solutions)

### 전략 A: U-Net 스타일의 Deep Connection (Skip-Connection)
- **개념**: `HierarchicalDeepStem`에서 정보를 압축할 때(Downsample), 각 해상도 단계별로 미세한 공간 정보(High-frequency Edge)가 담긴 피처 맵을 저장합니다. 이를 `HierarchicalDeepDecoder`의 동일한 해상도 단계에 더해주거나 연결(Concatenation)합니다.
- **효과**: DiT 병목(Bottleneck) 구간을 거치며 잃어버린 날카로운 원본 텍스처를 Stem에서 직접 빌려와 복원하므로 블러 현상을 근본적으로 해결할 수 있습니다.
- **참고**: `Deep Connection: U-Net Skip Bet.md` 제안과 정확히 일치합니다.

### 전략 B: Progressive PixelShuffle (Sub-pixel Convolution)
- **개념**: Bilinear Upsampling을 제거하고, 채널 차원을 공간 차원으로 변환하여 해상도를 키우는 `PixelShuffle` 기법을 사용합니다.
- **효과**: 보간(Interpolation) 연산이 없으므로 고주파 텍스처를 뭉개지 않고 그대로 해상도 팽창(Upscale)이 가능합니다. 초해상도(Super Resolution) 분야의 표준 방식입니다.
- **참고**: `Multi-Scale Bottleneck DiT.md`의 **Option 1** 제안과 일치합니다.

### 전략 C: 비대칭 주파수 손실 함수 강화 (Asymmetric Frequency Loss)
- **개념**: 디코더의 구조는 그대로 두되, Loss 함수 단계에서 고주파(High-frequency) 대역의 오차에 대해 페널티를 훨씬 강하게 주어 네트워크가 억지로 엣지를 살리도록 강제합니다.
- **효과**: 시각적 선명도가 일부 회복되지만, 구조적(Architectural) 한계를 Loss만으로 극복하려면 네트워크에 과도한 부담을 줄 수 있습니다. 전략 A/B와 병행하는 것이 이상적입니다.
