# 코드·가중치·추가 학습의 구분

## 직접 작성한 코드

앱 화면, 파일·동영상 처리, GPU 신경망 연산, 모델 데이터 판독·변환, 이미지 전처리, 사람 좌표 복원, 화면 분할 탐색, 중복 제거, 관절 표시와 추가 보정 학습 코드를 직접 작성했습니다. 기존 프로젝트의 실행 코드를 복사해 넣지 않았습니다.

배포 앱은 브라우저 WebGPU·Canvas·Worker 표준 API를 사용합니다. ONNX Runtime, TensorFlow.js, OpenCV, 외부 JavaScript 패키지나 CDN 실행 코드를 앱에 포함하지 않습니다. OpenAI API를 호출하지 않습니다. 개발용 `scripts/reference-check.py`는 수치 비교 기준을 얻기 위해 ONNX Runtime/Numpy를 사용하지만, 앱 실행이나 이번 보정 학습에는 사용하지 않습니다.

`gpu-graph.js`는 아래 두 모델에 필요한 연산을 독립적으로 구현한 실행 코드입니다. 모든 ONNX 모델을 지원하는 범용 엔진이라는 뜻은 아닙니다.

## 활용한 사전학습 가중치

가중치는 원저작자들이 사전학습한 결과이며 Movira가 처음부터 학습한 것으로 표현하지 않습니다. 원래 ONNX의 텐서·연산 그래프를 읽어 JSON과 float32 데이터로 변환했습니다. 기본 신경망의 학습 파라미터는 변경하지 않았습니다.

| 용도 | 가중치 | 출처와 고지 |
|---|---|---|
| 여러 사람 검출 | YOLOX-Tiny, 416×416 | [공식 모델 배포 문서](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/README.md), [배포 프로젝트 Apache-2.0 고지](web/models/LICENSE-YOLOX.txt) |
| 사람별 26개 관절·뒤꿈치·발끝 | RTMPose-s Halpe26, 256×192 | [공식 모델 목록](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose), [배포 프로젝트 Apache-2.0 고지](web/models/LICENSE-MMPose.txt) |

원본 파일:

- [YOLOX-Tiny ONNX](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx)
- [RTMPose-s Halpe26 ONNX 패키지](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip)

파일 검증용 SHA-256:

| 파일 | SHA-256 |
|---|---|
| 원본 YOLOX ONNX | `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7` |
| 원본 RTMPose ONNX | `85c5d669aa0b0c7969799ea9e0620d3b8c1d3c2b6878243e98a1ef8351457986` |
| 배포 person.bin | `5186ac1d379b8b7e6b4ec1ae32ace0d3a9a84138eb8d28fd89d358cd1a8165ed` |
| 배포 pose.bin | `0f5edaf036d0b528db7953614ec07a2c444151b9c7ef7ebb74ae6bb651b18741` |

가중치의 배포 프로젝트 고지를 함께 제공합니다. 직접 작성한 앱 코드와 외부 가중치·학습 자료의 이용 조건은 구분해야 합니다. 코드 재작성만으로 외부 가중치나 수집 사진의 이용 조건이 없어지는 것은 아닙니다.

## 실제로 추가 학습한 부분

아래 인물 판별 보정과 별도로, 현재는 **손 신경망 전체를 자체 초기값부터 학습**했습니다. 기존 YOLOX·RTMPose의 가중치는 여전히 변경하지 않았습니다.

`calibration.js`와 `calibration-trainer.js`는 사람 검출 점수에 크기·비율·분할 화면 경계 등의 정보를 반영하는 잔차 판별 보정층입니다. 가중치 7개를 직접 작성한 경사하강법으로 학습했습니다.

1,400장 중 1,120장을 학습, 140장을 보정 선택 검증, 140장을 보정층 분리 시험에 사용했습니다. 사람/배경 예제 13,280개로 180회 학습했습니다. 기본 YOLOX와 RTMPose의 전체 신경망을 다시 학습하거나 미세 조정한 것은 아닙니다. 학습한 보정 결과는 `web/calibration.json`입니다.

이 1,400장은 COCO train2017에서 수집했으므로 기본 가중치의 사전학습 자료와 겹칩니다. 해당 140장 수치는 추가 보정층의 효과를 확인하는 값이며 전체 시스템의 독립 일반화 성능으로 취급하지 않습니다. 최종 별도 사진 시험은 COCO val2017의 새 사진으로 확인합니다. [검증 기록](SCALE-VALIDATION.md)

## 직접 학습한 손 모델

2026-11-18에는 별도의 손 영역 분류·방향 제안 모델 `web/models/hand-locator-own.json`도 자체 난수 초기값부터 35회 학습했습니다. 파라미터 27,843개·실제 3,465회 갱신이며 외부 손 가중치를 사용하지 않았습니다. 손가락 위치를 출력하는 아래 합성곱 가중치는 이번에 변경하지 않았습니다. [추가 학습과 전체 연결 비교](HAND-SEARCH-VALIDATION.md), [학습 기록](reports/hand-locator-training.json). 새 모델도 같은 HaGRID 자료·원문 라이선스·저자 고지를 메타데이터에 포함합니다.

`web/models/hand-own.json`은 손용 외부 가중치를 변환한 파일이 아닙니다. 직접 작성한 6개 합성곱 층·공간 확률 출력·손 존재 판별 모델의 파라미터 122,630개를 자체 난수 초기값부터 100회 학습한 결과입니다. 기존 몸 모델 파라미터도 손 모델의 초기값으로 사용하지 않았습니다. GPU 순전파·역전파·Adam 갱신·이미지 증강은 `hand-net.js`와 `hand-training.js`의 코드입니다. [학습 기록](HAND-TRAINING-REPORT.json)

학습 사진과 관절 좌표 데이터는 [HaGRIDv2](https://github.com/hukenovs/hagrid)에서 수집했습니다. 데이터 제공자가 MediaPipe로 자동 생성한 관절 주석이며 수작업으로 확인된 해부학 정답과 다릅니다. Movira에서 MediaPipe 코드·실행 환경·가중치나 OpenAI API를 실행한 것이 아닙니다. 학습 데이터가 외부에서 왔다는 사실과 직접 작성한 코드·학습 결과를 구분합니다.

HaGRID의 원문 고지는 [LICENSE-HaGRID-data.pdf](web/models/LICENSE-HaGRID-data.pdf)에 함께 보관합니다. 원문은 **Public license with attribution and conditions reserved**이며 표준 CC BY-SA 자체와 동일한 라이선스가 아닙니다. 데이터 저자 표기는 모델 메타데이터와 검증 문서에도 넣었습니다. 데이터의 이용 조건이 코드 재작성으로 없어지는 것은 아닙니다. [원문 출처](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)

초기 손 가중치 파일 SHA-256: `77e58a472f7918ddab48a1bed113c713956227144822565e3a9de6d8b7143329`. 파일에는 직접 학습한 수치와 학습·검증·출처 메타데이터가 들어 있습니다. 몸의 공개 가중치와 출처를 혼동하지 않도록 별도 파일로 유지합니다.

추가 손 영역 모델 SHA-256: `252dbbb10856b7dfe20ae68f0d5598e0755f442727d931ce34eb805606f36021`. 분류기 가중치와 개발 사진에서 선택한 탐색 품질 기준을 포함합니다.


2026-11-18 관절 오차 개선에서는 자체 100회 가중치에서 40회 추가 학습한 후보를 실제로 검사하고 35회째 체크포인트를 선택했습니다. 좌표·연결 벡터 손실, 평균 위치 복원과 여러 입력 간 보정을 적용했습니다. 현재 손 가중치 SHA-256은 `c8e369d15f86a77bf5a56b67632b7aefdcd236b19f0fc2fbc1dbcb7f68070519`입니다. [현재 학습 기록](HAND-TRAINING-REPORT.json) · [현재 전체 사진 검증](HAND-JOINT-VALIDATION.md). 앞의 100회·초기 해시는 부모 모델의 기록입니다.
