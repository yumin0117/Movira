# Movira — 사진·동영상 포즈 스튜디오

## [▶ Movira 웹사이트 열기](https://movira.yuminkim9815.chatgpt.site)

**설치 없이 링크를 열고 사진이나 동영상을 선택하세요.**

1. **사진 또는 동영상 선택** 버튼을 누르거나 파일을 끌어 놓습니다.
2. 모델 준비 후 검출된 모든 사람의 관절과 뼈대가 나타납니다.
3. 동영상은 결과 화면 아래의 **▶** 버튼으로 재생합니다. 일시정지·위치 이동·반복 재생을 지원합니다.
4. 뼈대 선, 관절 점, 원본 화면을 각각 켜고 끌 수 있습니다. **현재 화면 저장**으로 결과 PNG를 받습니다.

웹사이트는 **로그인 없이 누구나 링크로 접속**할 수 있습니다. GitHub 소스 저장소는 비공개로 유지됩니다.

## 표시 방식

- 머리: **어깨 중앙(목) → 얼굴 중심**의 연결선 하나와 얼굴 점 하나로 표시합니다. 눈·귀의 개별 선이나 어깨에서 귀로 올라가는 선을 그리지 않습니다.
- 몸: 어깨, 팔꿈치, 손목, 골반, 무릎, 발목을 연결합니다.
- 발: 각 사람의 **실제 뒤꿈치·발끝 랜드마크를 추가로 추정**해 발목–뒤꿈치–발끝을 연결합니다. 고정 길이로 발을 임의 생성하지 않습니다.
- 발이 가려지거나 화면 밖에 있거나 신뢰도가 낮으면 발 표시가 생략될 수 있습니다. 신뢰도 슬라이더로 표시 기준을 조절할 수 있습니다.

## 로컬에서 웹사이트 실행 / VS Code

Node.js가 있는 PC에서 프로젝트 폴더를 열고:

```powershell
node scripts/serve-web.mjs
```

브라우저에서 **http://127.0.0.1:3000**을 엽니다. 서버 종료는 Ctrl+C입니다.
HTML을 파일 탐색기에서 직접 더블클릭하는 방식은 Worker / WASM 보안 제약 때문에 지원하지 않습니다.

VS Code의 **Run and Debug → Movira: Web → F5**는 배포된 웹사이트를 엽니다.
현재 웹 소스는 `web/index.html`, `web/styles.css`, `web/app.js`, `web/worker.js`, `web/pose-core.js`에 있습니다.
이전 C++ 참조 프로그램의 빌드와 실행 방법은 [README-native.md](README-native.md)를 참조하세요.

## 처리 방식과 개인정보

- 선택한 사진과 동영상은 **브라우저 메모리 안에서 처리**하며 서버로 업로드하지 않습니다.
- 이미지·영상의 파일 경로, 내용 및 분석 결과를 서버에 저장하지 않습니다.
- ONNX Runtime Web 1.20.1, YOLOv8n-pose, MediaPipe Pose 모델을 사이트에 포함했습니다.
- GPU(WebGPU)를 사용할 수 있으면 활용하고, 지원하지 않으면 CPU(WASM)로 처리합니다. 첫 분석은 모델 다운로드·초기화 때문에 더 오래 걸립니다.
- 영상에서는 처리 가능한 최신 프레임을 분석합니다. 프레임 대기열을 쌓지 않으며 30 FPS를 보장하지 않습니다. 속도는 기기와 검출 인원수에 따라 달라집니다.
- 브라우저가 지원하는 영상 코덱이 필요합니다. MP4(H.264) 또는 WebM을 권장합니다. AVI/MOV 등은 파일 확장자만으로 재생 가능 여부를 보장할 수 없습니다.
- 가림, 작은 인물, 역광, 화면 밖 신체 때문에 검출을 놓칠 수 있습니다. 3D 깊이, 손가락 개별 관절, 사람 ID 유지·재식별은 포함하지 않습니다.

## 검증

```powershell
node --test tests/web-pose.test.cjs
```

좌표 복원, 다중 인물/NMS, 출력 형식, 머리 단일 연결, 뒤꿈치·발끝 좌표 복원, 보이지 않는 발 처리에 관한 테스트가 포함됩니다.
실제 브라우저에서 인터넷 인물 사진과 동영상으로 검증한 내용은 [WEB-VALIDATION.md](WEB-VALIDATION.md)에 기록합니다.

## 라이선스 / 출처

Movira 소스 및 YOLOv8 모델: AGPL-3.0 (`LICENSE`). MediaPipe Pose 모델: Apache-2.0. ONNX Runtime: MIT.
모델 버전·해시와 전문은 [web/THIRD_PARTY_NOTICES.md](web/THIRD_PARTY_NOTICES.md) 및 `web/assets/LICENSE.*`에 있습니다.

- https://github.com/ultralytics/ultralytics
- https://huggingface.co/Xenova/yolov8n-pose
- https://github.com/opencv/opencv_zoo/tree/main/models/pose_estimation_mediapipe
- https://github.com/microsoft/onnxruntime
