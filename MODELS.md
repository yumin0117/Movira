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

`calibration.js`와 `calibration-trainer.js`는 사람 검출 점수에 크기·비율·분할 화면 경계 등의 정보를 반영하는 잔차 판별 보정층입니다. 가중치 7개를 직접 작성한 경사하강법으로 학습했습니다.

1,400장 중 1,120장을 학습, 140장을 보정 선택 검증, 140장을 보정층 분리 시험에 사용했습니다. 사람/배경 예제 13,280개로 180회 학습했습니다. 기본 YOLOX와 RTMPose의 전체 신경망을 다시 학습하거나 미세 조정한 것은 아닙니다. 학습한 보정 결과는 `web/calibration.json`입니다.

이 1,400장은 COCO train2017에서 수집했으므로 기본 가중치의 사전학습 자료와 겹칩니다. 해당 140장 수치는 추가 보정층의 효과를 확인하는 값이며 전체 시스템의 독립 일반화 성능으로 취급하지 않습니다. 최종 별도 사진 시험은 COCO val2017의 새 사진으로 확인합니다. [검증 기록](SCALE-VALIDATION.md)
