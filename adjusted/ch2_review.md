# Ch.2 Related Works 검토 보고서 (검토자 관점)

> 검토일: 2026-04-24
> 검토 기준: (a) 선행연구 커버리지 (b) 비판적 분석 vs 단순 요약 (c) 본 연구와의 gap 연결

---

## 전체 구조 평가

Ch.2는 4개 섹션으로 구성:

| 섹션 | 제목 | 분량 | 역할 |
|:---|:---|:---:|:---|
| 2.1 | Deep Learning for Solar Irradiance Prediction | 3단락 | 도메인 배경 |
| 2.2 | Diffusion Models and the Spectral Fidelity Bottleneck | 3단락 | 기술 배경 + Gap 1 |
| 2.3 | Mixture of Experts in Atmospheric Dynamics | 3단락 | 기술 배경 + Gap 2 |
| 2.4 | Evolution of Evaluation Paradigms | 3단락 | Gap 3 |

> ✅ 구조 자체는 RQ1/RQ2/RQ3에 1:1 대응하며 논리적.
> ⚠️ 그러나 전체적으로 **요약 > 비판적 분석** 경향이 강함.

---

## 섹션별 상세 검토

### 2.1 Solar Irradiance Prediction

**커버된 선행연구**: ConvLSTM, PredRNN, ViT(일반론), AC U-Net(자체 선행)

**누락된 주요 문헌**:
- **GenCast** (Price et al., 2024, Nature): 확산 모델 기반 기상 예보의 대표 성과. 반드시 인용 필요
- **DGMR** (Ravuri et al., 2021, Nature): bib에는 있으나 본문에서 활용이 약함. "generative model의 기상 적용" 문맥에서 핵심 선행
- **Pangu-Weather / FourCastNet**: 대규모 기상 AI의 맥락. 직접 비교 대상은 아니지만 "왜 diffusion이 필요한가"의 맥락에서 언급 가치

**비판적 분석 수준**: 
- "regression to the mean" 논변으로 deterministic -> generative 전환 필요성을 잘 유도 ✅
- 그러나 **solar irradiance 예측 분야의 다른 generative 접근(GAN 기반 등)이 전혀 없음** ❌

> **수정 제안**: DGMR을 "기상 분야의 generative 선행 성공 사례"로 본문에서 명시적으로 논의. GenCast를 "대규모 기상 diffusion"의 맥락으로 추가.

---

### 2.2 Diffusion Models and Spectral Fidelity

**커버된 선행연구**: DDPM, LDM(Rombach), WaveDiff(Phung), Wavelet Score Model(Guth), Mallat DWT

**누락된 주요 문헌**:
- **DiT** (Peebles & Xie, 2023, ICCV): Diffusion Transformer의 원 논문. 본 모델의 직접적 기반인데 **인용이 없음** -- 이것은 심각한 누락
- **U-ViT** (Bao et al., 2023): DiT와 함께 Transformer-based diffusion의 양대 축
- **WaveDM** (Huang et al., 2024): Wavelet Diffusion의 후속 연구

**비판적 분석 수준**:
- VAE의 spectral bottleneck 논변은 잘 구성됨 ✅
- WaveDiff/Guth 인용은 있으나, **이 연구들의 한계를 분석하지 않음** ❌
  - WaveDiff는 자연 이미지 대상이며, 기상 데이터의 multi-channel 특성을 고려하지 않음
  - Guth의 wavelet score model은 이론적 분석에 치중하며 실용적 적용이 제한적

> **수정 제안**: 
> 1. DiT 인용 반드시 추가 (본 모델의 backbone)
> 2. WaveDiff/Guth의 한계를 명시하여 "기상 데이터에 wavelet diffusion이 미개척"이라는 gap을 더 강하게 유도

---

### 2.3 Mixture of Experts

**커버된 선행연구**: Shazeer(원론), V-MoE(Riquelme), VideoMoE(Xue), Tutel(Hwang)

**누락된 주요 문헌**:
- **DeepSeek-MoE** (Dai et al., 2024): Shared Expert + Fine-grained Expert의 설계가 본 연구의 shared expert와 직접 관련
- **ST-MoE** (Zoph et al., 2022): Stable training 기법. capacity factor 설계의 근거
- **Switch Transformer** (Fedus et al., 2022): top-1 routing의 대표 연구

**비판적 분석 수준**:
- "static routing -> spatiotemporal-aware" 전환의 필요성이 잘 논증됨 ✅
- 그러나 **heterogeneous expert 설계의 선행 근거가 전혀 없음** ❌
  - 기존 MoE는 모두 homogeneous expert를 사용하는데, 왜 heterogeneous가 필요한가?
  - ConvNeXt의 depthwise convolution이 Transformer와 결합되는 선행(MetaFormer 등)을 언급해야 함

**bib 오류**:
- `xue2022videomoe`: 출처가 "(학회 확인 필요)"로 되어 있음 -- 정확한 출처 수정 필요
- `hwang2023tutel`: 저자 목록에 "others"가 포함되어 불완전

> **수정 제안**: DeepSeek-MoE의 shared expert 설계를 선행으로 인용하여 본 연구의 shared expert 선택에 근거 부여.

---

### 2.4 Evaluation Paradigms

**커버된 선행연구**: SSIM(암묵적), LPIPS(Zhang), FID(Heusel), double penalty(Ebert)

**비판적 분석 수준**:
- double penalty 문제를 Ebert(2008)과 연결한 것은 좋음 ✅
- 그러나 **LPIPS/FID가 기상 이미지에 적합한가에 대한 논의가 없음** ❌
  - LPIPS는 ImageNet pre-trained AlexNet 기반. 기상 이미지의 특성(단일 채널, 비자연 이미지)에서의 유효성 논의 필요
  - FID도 Inception-v3 기반이며 동일 문제 존재

> **수정 제안**: "pre-trained feature extractor가 기상 도메인에 최적화되지 않았으므로, 이 metric들의 절대값보다는 모델 간 상대 비교에 의미가 있다"는 한정(qualification)을 추가.

---

## 종합: 우선순위 수정 항목

| 우선순위 | 항목 | 영향도 |
|:---:|:---|:---:|
| 1 | **DiT(Peebles 2023) 인용 추가** -- backbone의 원 논문 미인용은 심각 | 높음 |
| 2 | **GenCast/DGMR 본문 논의 보강** -- 기상 생성 모델의 주요 선행 | 높음 |
| 3 | **WaveDiff/Guth 한계 분석 추가** -- 단순 인용 -> 비판적 분석으로 격상 | 중간 |
| 4 | **DeepSeek-MoE shared expert 선행 추가** | 중간 |
| 5 | **LPIPS/FID 도메인 적합성 한정** | 중간 |
| 6 | **bib 오류 수정** (xue2022videomoe, hwang2023tutel) | 낮음 |
