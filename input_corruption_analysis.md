# 입력 데이터 스케일링 오염 및 스무딩 원인 분석 보고서

Raw Pixel 스윕 모델들에서 선명도(Sharpness) 개선이 미미하고 평활화(Smoothing) 현상이 지속되는 현상에 대하여, 코드와 입력 컨텍스트 및 모델 아키텍처를 엄밀하게 교차 검수한 결과입니다.

---

## 1. 소프트웨어 엔지니어 (Software Engineer) 관점: 연산 오염 (입력 스케일링 불일치 버그)

가장 크리티컬한 원인으로, **DWT 모드와 Raw Pixel 모드 간의 FFL 및 SGP 입력 텐서의 물리적 스케일 불일치**가 발견되었습니다.

* **DWT 모드 (정상)**:
  * [train.py](file:///home/oem/ST-DiT2/ST-DiT/stdit/engine/train.py)의 Line 388에서 DWT 잠재 텐서에 `mean_tensor`와 `std_tensor`를 적용하여 물리적 실제 스케일로 역변환(De-normalization)합니다.
  * 이후 IWT 모듈을 통과시켜 **실제 일사량 물리 범위인 `0 ~ 26.35` 스케일의 픽셀 이미지(`I_final`, `I_target`)**를 생성하고, 이를 FFL 및 SGP 손실 계산에 투입합니다.
* **Raw Pixel 모드 (스케일 오염)**:
  * [train.py](file:///home/oem/ST-DiT2/ST-DiT/stdit/engine/train.py)의 Line 451 `elif` 분기를 보면, `pred_noise`와 `target`은 데이터셋에서 반환된 **`[-1, 1]` 정규화 상태의 텐서**입니다.
  * 하지만 해당 분기에서 FFL 및 SGP를 계산할 때, **물리적 스케일(`0 ~ 26.35`)로의 역정규화(De-normalization) 과정을 누락한 채** `[-1, 1]` 텐서를 그대로 입력으로 전달하였습니다.
  * **수학적 여파**:
    1. 데이터 최댓값 범위가 `26.35`에서 `1.0`으로 축소되어, Sobel 필터의 에지 미분값(SGP) 및 2D FFT 진폭 오차(FFL)의 신호 강도가 **물리 스케일 대비 약 10배 이상 극도로 감쇄(Diluted)**되었습니다.
    2. 데이터 오프셋이 `-1.0`으로 내려가 음수가 포함되면서 FFL 계산을 위한 2D FFT의 저주파(DC component) 대역 스펙트럼 강도가 왜곡되었습니다.
  * **결론**: FFL 및 SGP 가중치 업데이트 신호가 실질적으로 무력화되어 모델 학습 시 에지와 디테일을 강제하지 못하고 스무딩 현상이 해결되지 않았습니다.

---

## 2. 도메인 전문가 (Domain Expert) 관점: 물리 범위와 변화율 보존 실패

일사량(GHI) 데이터와 바람장 물리 피처의 관계 및 전처리 상의 문제입니다.

* **물리 피처 per-sample min-max의 한계**:
  * [phys_features.py](file:///home/oem/ST-DiT2/ST-DiT/stdit/data/phys_features.py)의 `_normalize_to_context_range`는 배치 내 샘플별로 물리 피처 맵을 `[-1, 1]`로 강제 매핑합니다.
  * 일사량 변화가 0에 가깝거나 아예 없는 야간 시간대(`I_5 - I_4 = 0.0`)에도 min-max 수식상 강제로 전체 픽셀이 `-1.0`인 텐서로 채워지며, 미세한 노이즈만 존재하는 경우 노이즈가 `[-1, 1]`로 대폭 증폭되어 입력 컨텍스트를 오염시킵니다.
  * 이는 물리 변화의 절대적인 강도를 모델이 인지하지 못하게 방해하는 원인입니다.

---

## 3. 아키텍처 전문가 (Architecture Expert) 관점: Deep Stem의 구조적 평활화 성향

사용 중인 모델 아키텍처의 인풋 토큰화 계층이 가지는 한계입니다.

* **HierarchicalDeepStem의 스무딩 특성**:
  * 사용 중인 모델은 `--stem_type deep`을 적용하고 있습니다.
  * [stem.py](file:///home/oem/ST-DiT2/ST-DiT/stdit/models/stem.py)의 `HierarchicalDeepStem`은 패치 크기 `16`을 달성하기 위해 `Stride=2`를 갖는 3x3 Conv 레이어 4단을 점진적으로 통과시킵니다.
  * 점진적 CNN stem 믹싱은 receptive field가 패치 영역보다 넓어 토큰 경계를 부드럽게 만드는 장점이 있으나, 인풋 데이터에 포함된 고주파 에지 및 텍스처를 **Stem 수준에서 공간적으로 평활화(Spatial smoothing)**하여 DiT 백본 블록으로 넘겨버리는 아키텍처적 한계를 지닙니다.
  * 1번 항목에서 언급된 FFL 및 SGP 손실 신호가 스케일 감쇄로 인해 극도로 약화된 상태에서는, Deep Stem이 유발하는 이 구조적 스무딩 경향성을 이겨낼 수 없게 됩니다.

---

## 4. 리뷰어 (Reviewer) 종합 의견 및 제언

입력 데이터와 물리 연산 오차 전파 경로에 명백한 스케일링 결함이 존재하므로, 현재 진행 중인 스윕 실험 결과가 뭉개지는 것은 데이터 파이프라인 상 필연적인 결과입니다.

* **수정 방향 권고**:
  * Raw Pixel 모드 학습 시, FFL 및 SGP 손실을 계산하기 전에 `pred_noise`와 `target`을 실제 픽셀 스케일(`0 ~ 26.35`)로 복원하도록 역정규화 연산을 추가해야 합니다.
  * 역정규화 수식: `unscaled = (scaled + 1.0) * DATA_MAX / 2.0`
