# Jump Ideas: P0 -> 목표 성능 도약을 위한 아이디어 수집

> 현재: P0 (MSE 0.0359, SSIM 0.538, FID 215)
> 목표: MSE ≤ 0.030, SSIM ≥ 0.60
> 격차: MSE -16.5%, SSIM +11.5%

---

## 왜 Jump이 필요한가

Phase 4~9에서 30회 이상의 incremental 실험이 **전원 P0 대비 악화 또는 동등**. 현 프레임워크 내 미세 조정이 **국소 최적에 도달**했음을 시사.

> [!IMPORTANT]
> **핵심 진단**: LF 트랙(MoE)은 최적. **HF 트랙(단일 ConvNeXt)과 Cross-Scale Interaction(FiLM)의 표현력 부족**이 SSIM bottleneck의 근본 원인 가능성. road_map.md에 "Conv 단독 HF 트랙이 SSIM bottleneck 원인 가능성" 기록.

---

## 카테고리 A: HF 트랙 강화

### A1. HF Hetero MoE (0008_0002) — 구현 완료, 대기 중

**근거 문헌**:
- Riquelme et al., "Scaling Vision with Sparse Mixture of Experts (V-MoE)," NeurIPS 2021 — Vision MoE의 표준
- Hwang et al., "Tutel: Adaptive MoE at Scale," MLSys 2023 — Hetero Expert 구현 효율화
- 본 thesis Phase 3-A: FreqMoE(LF MoE) 도입 시 **SSIM +12.2%** (11B vs 11C). 동일 원리의 HF 적용

**핵심**: 단일 ConvNeXt를 5종 Hetero Expert MoE(1x1, 3x3, 5x5, DWConv7x7, Dilated3x3)로 교체. 구현 완료(`--use_hf_moe`).

**기대 효과**: LF MoE +12% SSIM 전례 기반. 이것만으로 0.538 -> 0.60 도달 가능성.
**위험**: 낮음 (구현 완료). **소요**: ~2.5h.

### A2. Cross-Scale Attention (FiLM 대체)

**근거 문헌**:
- Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration," CVPR 2022 — Multi-scale cross-attention으로 HF detail 복원
- Chen et al., "HAT: Hybrid Attention Transformer for Image Restoration," arXiv:2309.05239, 2023 — Cross-window attention으로 주파수 간 정보 교환
- Wang et al., "Uformer: A General U-Shaped Transformer for Image Restoration," CVPR 2022 — Multi-scale feature interaction

**현재**: BiDirCrossScaleInteraction이 FiLM(gamma, beta)으로 LF->HF 전달. **단방향 affine transform**으로 표현력 제한.
**제안**: Cross-Attention 기반 교체. HF 토큰이 LF 토큰에 attend -> 공간적으로 선택적 정보 전달.

**과거 시도**: Phase 5에서 CrossTrack(additive/replace) 비채택. 당시 HF MoE 미적용 + additive 방식. **FiLM 자체를 교체하는 것은 미시도**. A1과 결합 시 효과 다를 수 있음.
**위험**: 중간. **소요**: ~3h (코드 존재, 재학습만).

### A3. HF 전용 Noise Schedule (Subband-specific) — ❌ 과거 실패 확인

> [!WARNING]
> **과거 4회 실패 확인**. 이 아이디어는 이미 철저히 시도되었으며, **구조적으로 부적합**이 확정됨.

**과거 실험 기록** ([0004_adaptive_schedule_meeting.md](file:///home/oem/ST-DiT2/docs/0004_adaptive_schedule_meeting.md)):

| 실험 | 방식 | MSE 변화 | SSIM 변화 | 판정 |
|:---|:---|:---:|:---:|:---:|
| 13B v1 | pixel-wise (공간+채널) | +95% | -20.4% | ❌ |
| 13B v2 | pixel-wise (공간만) | +16.7% | -14.1% | ❌ |
| 13C | Band (채널그룹별) + SNR | +24.7% | -15.1% | ❌ |
| 13D | Band (채널그룹별), no SNR | +23.9% | -9.0% | ❌ |

**실패 원인 (확정)**:
1. DiT의 self-attention은 **전역 예측**을 하는데, 역변환은 **국소 계수**를 적용하는 불일치가 SSIM 파괴
2. per_channel_snr(loss 가중치)과 Band Schedule(noise 주입량)이 같은 채널 축에서 **이중보정** 발생
3. 회의 결론: "DWT latent에서 noise schedule 변경은 구조적으로 부적합"

추가로 [0001_bayesshrink_analysis.md](file:///home/oem/ST-DiT2/docs/0001_bayesshrink_analysis.md) Section 7-D에서도 "Subband-adaptive schedule은 구현 복잡도가 높고, DWT latent space에서의 적용 사례가 아직 제한적. 향후 연구 과제"로 기록.

**문헌**:
- Guth et al., "Wavelet Score-Based Generative Modeling (WSGM)," NeurIPS 2022 — 서브밴드별 score 분해 (본 모델과 다른 접근)
- Phung et al., "WaveDiff: Wavelet Diffusion Models Are Fast and Scalable," CVPR 2023 — GAN-diffusion 하이브리드 (순수 DDIM 미해당)

**결론**: ❌ **Jump 후보에서 제외**. 4회 실패 + 구조적 부적합 확정.

---

## 카테고리 B: 학습 패러다임 전환

### B1. Flow Matching / Rectified Flow 전환

**근거 문헌**:
- Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023 (arXiv:2210.02747) — Flow Matching 정식 도입
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow," ICLR 2023 (arXiv:2209.03003) — Rectified Flow, Reflow 알고리즘
- Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)," arXiv:2403.03206, 2024 — Stable Diffusion 3가 Rectified Flow 채택
- Price et al., "GenCast: Diffusion-based Ensemble Forecasting for Medium-range Weather," Nature, 2024 — 기상 예보에서의 확산 모델 활용
- FlowCast (arXiv, 2024) — 강수 nowcasting에 Conditional Flow Matching 적용

**현재**: DDPM v-prediction + DDIM 10 steps.
**제안**: Rectified Flow 전환. 직선 경로(ODE) -> 적은 step으로 더 정확한 결과.

**기대**: 동일 NFE에서 **더 높은 SSIM, 더 낮은 MSE** (2024-2025 비교 연구 합의).
**위험**: **높음**. gaussian.py 재작성, per_channel_snr/aux loss 호환성 재검증. 2-3일 소요. 모든 기존 실험 무효화 위험.

### B2. Coarse-to-Fine Timestep Curriculum

**근거 문헌**:
- "Denoising Task Difficulty-based Curriculum for Training Diffusion Models," arXiv:2403.10348, 2024 — low-noise timestep이 더 어려움을 실증. easy-to-hard curriculum으로 수렴 속도 +30%, 최종 품질 개선
- Hang et al., "Improved Diffusion Training," ICCV 2025 (arXiv:2407.03297) — timestep sampling 최적화

**제안**: 학습 초반에 high-noise timestep 위주 -> 후반에 low-noise timestep 위주.
**구현**: timestep sampler에 curriculum schedule 추가 (~20줄).
**위험**: 낮음. **소요**: ~3h. prototyping 규모(100ep)에서는 curriculum 효과 부족 가능.

---

## 카테고리 C: 추론 품질 개선 (학습 불필요)

### C1. Wavelet-domain Guidance (LWD/HiWave 방식)

**근거 문헌**:
- "Latent Wavelet Diffusion (LWD)," arXiv:2506.00433, 2025 — DWT energy map 기반 spatially adaptive denoising. 아키텍처 변경 0, 추론 비용만 소폭 증가. 2K-4K 이미지에서 HF 디테일 개선
- Vontobel et al., "HiWave: Training-Free High-Resolution Image Generation via Wavelet-Based Diffusion Sampling," SIGGRAPH Asia 2025 (arXiv:2506.20452) — 추론 시 DWT로 LF 보존 + HF 선택적 guidance

**제안**: 추론 시 DDIM step마다 DWT energy map 계산 -> HF 서브밴드에 추가 guidance.
**핵심 장점**: 기존 체크포인트 그대로 사용. 학습 변경 0.
**위험**: 매우 낮음. **소요**: ~4h (스크립트 구현 + 평가).

### C2. Dynamic CFG Schedule

**근거 문헌**:
- Ho & Salimans, "Classifier-Free Diffusion Guidance," NeurIPS Workshop, 2022 (arXiv:2207.12598) — CFG 원논문
- Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models (EDM2)," arXiv:2312.02696, 2023 — timestep별 guidance 강도 변화의 효과 분석
- 본 thesis Ideas 0004_0001: CFG Schedule **채택 권장**이나 미실험 상태

**현재**: CFG gs=1.5 constant. **제안**: step별 guidance 변화 (초반 강, 후반 약).
**위험**: 매우 낮음 (추론만). **소요**: ~1h.

---

## 카테고리 D: 데이터/전처리

### D1. 3yr Full Data (진행 중)

**근거 문헌**:
- Kaplan et al., "Scaling Laws for Neural Language Models," arXiv:2001.08361, 2020 — 데이터 규모 확장의 일반적 효과
- 본 thesis 9G: 3yr FID 121 (P0의 215 대비 -44%). SSIM은 동등(0.540)

**상태**: Final_P0_3yr_D4 학습 중 (H200).

### D2. CSI 정규화 (Ideas 0005_0001)

**근거 문헌**:
- Ineichen & Perez, "A New Airmass Independent Formulation for the Linke Turbidity Coefficient," Solar Energy, 2002 — Clear-sky 모델 표준
- Lauret et al., "Solar Forecasting in a Challenging Insular Context," Atmosphere, 2022 — CSI 정규화가 ML 예측 정확도에 미치는 영향
- 일반 합의: CSI 정규화는 태양광 예보의 **표준 전처리**. 비정상성 제거 -> 모델이 구름 효과에 집중

**기대**: AC U-Net이 사용하는 방식과 일치 -> **공정 비교에 필수**. 논문 차별성 최대.
**위험**: 데이터 파이프라인 재구축. **소요**: 1-2일.

---

## 카테고리 E: 아키텍처 재설계

### E1. Cross-Scale Attention (A2의 확장)

A2와 동일. 문헌은 A2 참조.

### E2. Depth 증가 (D4 -> D6/D8)

**근거 문헌**:
- Peebles & Xie, "Scalable Diffusion Models with Transformers (DiT)," ICCV 2023 — Depth 증가에 따른 FID 개선
- 본 thesis: 9G(D8)는 FID 121이지만 SSIM 0.540으로 P0(D4, 0.538)과 동등

**분석**: D8이 SSIM 미개선인 이유는 **HF 트랙의 bottleneck이 depth와 무관**하기 때문. **A1 적용 후 재검토**.

---

## 카테고리 F: Adaptive 학습

### F1. Wavelet Energy Map 기반 Spatial Loss Masking

**근거 문헌**:
- "Latent Wavelet Diffusion (LWD)," arXiv:2506.00433, 2025 — DWT energy map으로 detail-rich 영역에 높은 loss 가중치. 아키텍처 변경 없이 HF 품질 개선
- 본 thesis [0004_adaptive_schedule_meeting.md](file:///home/oem/ST-DiT2/docs/0004_adaptive_schedule_meeting.md) 방향 A: "Loss-side 공간 가중치"가 **1순위 합의안**이었으나 Schedule-side(13C/13D)만 실험 후 전체 비채택 처리됨. **Loss-side 자체는 미실험**

> [!IMPORTANT]
> Adaptive Schedule 회의(2026-04-24)에서 방향 A(Loss-side 공간 가중치)가 1순위 합의안이었으나, 실제로는 방향 C/D(Schedule-side)만 실험됨. **합의된 1순위 방향이 아직 실험되지 않은 상태**.

**구현**: per_channel_snr 위에 **공간 차원 가중치** 추가 (~30줄). DWT energy map은 이미 latent에서 계산 가능.
**위험**: 낮음. **소요**: ~3h.

### F2. Timestep-dependent HF Loss Reweighting

**근거 문헌**:
- Hang et al., "Improved Noise Schedule for Diffusion Training (Min-SNR)," ICCV 2025 (arXiv:2407.03297) — timestep별 loss 가중치 최적화
- Sahoo et al., "ANT: Adaptive Noise Schedule for Time Series Diffusion Models," NeurIPS 2024 (arXiv:2410.14488) — 데이터 통계 기반 적응형 schedule
- 본 thesis Ideas 0002_0001, 0002_0010: 관련 아이디어 검토 완료

**현재**: per_channel_snr은 timestep별 가중치를 적용하지만 **채널별 균일 schedule**.
**제안**: low-noise timestep에서 HF 채널 가중치를 더 높이는 adaptive schedule.
**위험**: 낮음. **소요**: ~3h.

---

## 우선순위 정리

| 순위 | ID | 후보 | Jump 확률 | 비용 | 핵심 근거 |
|:---:|:---:|:---|:---:|:---:|:---|
| **1** | A1 | **HF Hetero MoE** | ★★★★★ | 2.5h | LF MoE +12% SSIM 전례. 구현 완료 |
| **2** | F1 | **Spatial Loss Masking** | ★★★★ | 3h | LWD(2025). 합의 1순위 미실험 |
| **3** | C2 | CFG Schedule | ★★★ | 1h | 채택 권장 미실험. 학습 불필요 |
| **4** | C1 | Wavelet-domain Guidance | ★★★ | 4h | LWD/HiWave(2025). 학습 불필요 |
| **5** | A2/E1 | Cross-Scale Attention | ★★★ | 3h | Restormer/HAT. A1과 결합 시 시너지 |
| **6** | B2 | Timestep Curriculum | ★★★ | 3h | arXiv:2403.10348. 위험 낮음 |
| **7** | D2 | CSI 정규화 | ★★★★ | 1-2일 | 태양광 예보 표준. 논문 가치 최대 |
| **8** | B1 | Flow Matching | ★★★★★ | 2-3일 | Lipman(2023)/SD3(2024). Deadline 위험 |
| ~~9~~ | ~~A3~~ | ~~HF Noise Schedule~~ | ~~-~~ | ~~-~~ | ~~**4회 실패 확정. 제외**~~ |

> [!IMPORTANT]
> **권장 전략**: A1(HF MoE) -> F1(Spatial Masking) -> C2(CFG Schedule) 순서.
> A1 단독으로 SSIM 0.60 미달 시 A2(Cross-Scale Attn)와 결합.
> B1(Flow Matching)은 deadline 여유가 있을 때만.

---

## 기존 실험과의 관계

| Jump 후보 | 기존 시도 | 차이점 |
|:---|:---|:---|
| A1 HF MoE | 미시도 | 신규. LF MoE 성공 전례 활용 |
| A2 Cross-Attn | Phase 5 CrossTrack (비채택) | Phase 5는 additive. 본 제안은 FiLM 자체 교체 |
| ~~A3 HF Noise~~ | **13B/13C/13D/ZTSNR (4회 실패)** | **제외. 구조적 부적합 확정** |
| F1 Spatial Masking | 회의 합의 1순위 (미실험) | Schedule-side만 실험됨. Loss-side는 미시도 |
| B2 Curriculum | 미시도 | 신규 |
| C1 Wavelet Guidance | 미시도 (추론만) | 학습 변경 0 |
