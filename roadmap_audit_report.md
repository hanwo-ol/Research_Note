# 로드맵 및 아이디어 문서 검수 리포트 (2026-04-29 기준)

요청하신 `road_map.md`, `road_map_appendix.md`, `docs/`, `docs/ideas/` 폴더 내의 모든 문서들을 대조 검수했습니다. 현재 대기 중인 실험과 로드맵에 아직 미반영된 아이디어들을 다음과 같이 분류했습니다.

---

## 1. 현재 로드맵상 대기/진행 중인 실험 (road_map.md / appendix)

### 🔄 최우선 진행 중
- Final_P0_3yr_Full_D4 [H200]: 3년 전체 데이터(19,682샘플) 정정 재학습
- A1_HFMoE_ZLoss_D4 [A6000]: HF MoE 및 Router z-loss 
- S1a_LatSSIM_w0.2 [A6000]: Latent SSIM 단조 악화 추세 최종 확인 (이후 LatSSIM 계열 폐기 수순)

### 📋 보완 실험
- Phase 7 (PhysInput): Per-frame Channel Concat (0007_0002) 적용
- Aux Loss 개선: AuxPixSSIM 개선(가중치 축소, warmup 적용 등), Aux L1 블록 위치 검증(16가지 조합)
- EMA 재검증: 3년 전체 데이터 규모에서 decay=0.9999로 EMA 모델 평가
- 논문 평가(Appendix): 예측 스텝 5/20/50 추론 평가, 3단계 DWT 재학습, PSD 및 VAE 오차 비교, Dense FFN 기준선 재학습

---

## 2. 로드맵에 미반영된 미수집 아이디어 (docs/ 검수 결과)

`docs/` 폴더 내의 분석 및 설계 문서들을 검수한 결과, 로드맵에는 등재되지 않았으나 매우 가치가 높은 후보 기법들이 발견되었습니다.

### ⭐ 0050: Brightness-Conditional HF MoE Gating
- 출처: `docs/0050_brightness_conditional_hfmoe.md`
- 내용: 기준 모델(main)과 고주파 MoE(aux)를 병렬 구성하고, 일사량(밝기) 신호에 따라 어두울 때만 고주파 MoE를 활성화하도록 학습.
- 제안 실험: 
  - Smoke_BrightnessGate (기본 테스트)
  - B1_BrightnessGate_HFMoE_D4 (정식 학습)
  - Ablation (고정 혼합 비율 비교)
- 비고: 로드맵에 완전히 누락되어 있음. 가장 유력한 개선 후보.

### ⭐ 0040: Jump Ideas 중 미실험 1순위
- 출처: `docs/0040_jump_ideas.md`
- 내용:
  - Spatial Loss Masking: 웨이블릿 에너지 맵 기반으로 공간적 손실 가중치 조절. (과거 회의에서 1순위로 합의했으나 미실험 상태임)
  - Cross-Scale Attention: 기존의 단순 조정을 대체하여 고주파와 저주파 간의 어텐션 도입.
  - DWT Channel Energy Redistribution Guidance: 추론 시 스텝마다 고주파 채널에 선택적 에너지 방향성 주입 (학습 불필요, 6-8시간 소요).

---

## 3. docs/ideas/ 폴더 검수 결과

`docs/ideas/0001` ~ `docs/ideas/0011`은 이전 리뷰 문서들을 통해 이미 채택/기각 여부가 정리되어 `road_map.md`에 편입되었습니다. 하지만 가장 최근인 0012번 아이디어들은 리뷰만 완료되고 로드맵 계획으로 이동하지 않았습니다.

### 💡 0012번 (외부 MoE 최적화 기법) 중 로드맵 누락 건
- 출처: `docs/0039_ideas_0012_review.md`
- 누락 항목:
  1. 0012_0008 Per-timestep 분석: 전문가(Expert) 모델들의 시간대별 분업 현상 분석 (학습 불필요, 기존 모델로 즉시 가능)
  2. 0012_0006 Hash Routing (Ablation): 시공간 라우터의 학술적 가치를 정당화하기 위한 랜덤 라우터 비교 실험.
  3. 0012_0002 Expert Importance Balancing: 현재 코드의 균형 조절 로직이 GShard 방식인지 확인하는 코드 검수 작업.

---

## 📋 제안: 다음 실행 계획
현재 GPU 서버가 비워지는 대로 곧바로 수행할 수 있도록, `road_map.md`의 [향후 계획] 섹션에 다음 항목들을 정식 편입시키는 것을 권장합니다:
1. B1_BrightnessGate_HFMoE_D4 (0050 설계 문서)
2. Spatial Loss Masking (0040 합의 1순위 미실험 건)
3. Per-timestep 분석 및 Hash Routing Ablation (0012 누락 건)
