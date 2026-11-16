# Movira Native (C++ 참조 버전)

C++17과 OpenCV로 사진·동영상·웹캠의 여러 사람에게 관절 점과 뼈대 선을 표시합니다.
VS Code용 빌드·실행 설정을 포함합니다. Python이나 외부 추론 API 없이 로컬 CPU에서 실행됩니다.

## 준비 (Windows x64)

- Visual Studio / Build Tools: **Desktop development with C++**, MSVC x64, Windows SDK, C++ CMake tools.
- Visual Studio Code + Microsoft **C/C++** 확장 (`ms-vscode.cpptools`).
- 최초 한 번 OpenCV 4.10 공식 Windows 바이너리 준비:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

위 명령의 실행 정책 옵션은 해당 프로세스에만 적용됩니다. 프로젝트 폴더에서 실행합니다.
의존성은 `.deps/`에 저장합니다. 시스템 PATH 변경은 필요 없습니다.
포함한 `models/yolov8n-pose.onnx`는 고정 640×640 입력 모델입니다.

## VS Code에서 빌드·실행

1. **File → Open Folder**로 `Movira` 폴더를 엽니다.
2. **Ctrl+Shift+B**: `Movira: Build and test`로 Release 빌드와 테스트를 수행합니다.
3. **Run and Debug** 목록에서 다음 중 하나를 골라 **F5**를 누릅니다.
   - **Movira: Photo** — 사진 경로 입력 (기본값 `samples/demo.jpg`)
   - **Movira: Webcam** — 기본 카메라 0
   - **Movira: Video** — 동영상 경로 입력
4. 결과 창에서 **Esc / Q** 또는 창 닫기로 종료합니다.

개인 사진·동영상은 `samples/`에 넣어 사용합니다. 해당 폴더의 미디어는 Git에서 제외되어 업로드되지 않습니다.
사진/영상은 저장소에 포함하지 않았으므로 처음 복제한 경우 본인의 미디어 경로를 지정하세요.

터미널에서 직접 실행:

```powershell
.\scripts\build.cmd
.\build\Movira.exe --source=0
.\build\Movira.exe "--source=C:/videos/people.mp4"
.\build\Movira.exe "--source=C:/photos/people.jpg" --save=results/photo.png
```

모든 실행은 **Movira 폴더를 작업 디렉터리**로 사용합니다. 두 번째 카메라는 `--source=1`입니다.

## 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--source` | `0` | 카메라 번호 또는 로컬 이미지·영상 파일 |
| `--model` | `models/yolov8n-pose.onnx` | 원시 pose 출력 모델 경로 |
| `--confidence` | `0.35` | 사람 검출 신뢰도 |
| `--keypoint` | `0.35` | 관절 표시 신뢰도 |
| `--save` | 없음 | 마지막 처리 프레임을 이미지로 저장 (예: PNG) |
| `--headless` | 없음 | 화면 창 없이 처리 |
| `--frames` | `0` | 영상 처리 프레임 수 제한. 0은 끝까지 |
| `--loop` | 없음 | 영상 끝에서 반복 재생 |

`--save`는 동영상 녹화가 아닌 **한 프레임 저장**입니다. 영상 끝에서는 마지막 화면을 유지합니다.

## 동작

1. 원본 종횡비를 유지한 640×640 letterbox, BGR → RGB / 255 전처리.
2. YOLOv8n-pose ONNX 모델로 여러 사람의 박스와 COCO 관절 17개 동시 추정.
3. NMS로 중복 검출을 제거하고 원본 픽셀 좌표로 복원.
4. 검출된 모든 사람의 관절과 뼈대를 표시. 어깨·골반 중점으로 중앙 척추 선 추가.

영상 캡처와 추론을 분리하여 최신 프레임 하나만 보관합니다. 동영상은 원본 FPS에 맞춰 입력하며 추론이 느리면 중간 프레임을 건너뜁니다. 지연 누적을 줄이기 위한 방식이며 모든 프레임을 저장하는 분석 용도와는 다릅니다.

CPU 기본 구현이므로 FPS는 하드웨어에 따라 다릅니다. 30 FPS는 보장하지 않습니다. 가림, 작은 인물, 역광, 흔들림, 화면 밖 신체로 인해 사람·관절을 놓칠 수 있습니다. 검출된 사람 모두에 표시하며, 화면 속 모든 사람의 검출을 보장하지는 않습니다.

손가락·발가락 개별 관절, 3D 깊이/캐릭터 리깅, 사람별 ID 유지·재식별은 포함하지 않습니다. 추가 모델은 고정 FLOAT `[1,3,640,640]` 입력과 원시 `[1,56,N]` / `[1,N,56]` 출력을 가진 pose 모델만 지원합니다. NMS 내장/YOLO26 end-to-end 모델은 그대로 대체할 수 없습니다.

## 테스트

`scripts/build.cmd`에서 CTest를 실행합니다. 좌표 역변환, 다중 검출 유지, 중복 제거, 신뢰도 필터, 출력 방향, 미지원 출력 거부, 낮은 신뢰도 관절 숨김을 확인합니다.

실제 모델 smoke test:

```powershell
.\build\Movira.exe "--source=C:/photos/multiple-people.jpg" --headless --save=results/test.png
.\build\Movira.exe "--source=C:/videos/people.mp4" --headless --frames=12 --save=results/video.png
```

## 모델·라이선스

소스: AGPL-3.0 (`LICENSE`). OpenCV: Apache-2.0.
모델: Ultralytics YOLOv8n-pose, Xenova ONNX 변환 배포본, AGPL-3.0.

- 원본: https://github.com/ultralytics/ultralytics
- 모델 게시자: https://huggingface.co/Xenova/yolov8n-pose
- 버전 고정 모델: https://huggingface.co/Xenova/yolov8n-pose/resolve/f0be3feb14d7d9cb44a7ec00ea6cc63899d02158/onnx/model.onnx
- SHA-256: `04f6d2416266f2aba6c5ba8b26de33ed9eba3279f972ac02e33f9f1366547586`
- OpenCV 공식 배포: https://github.com/opencv/opencv/releases/tag/4.10.0

Movira는 정확도나 처리 속도에 대한 보증 없이 제공됩니다.
