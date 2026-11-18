# 수집 자료의 출처

기본 폴더: 바탕화면 `Movira_test_Video`.

- `images/`: 내려받은 사진 원본.
- `pixels/`: Windows 이미지 기능으로 디코딩한 BGRA 픽셀. 직접 작성한 학습 CLI에서 읽음.
- `movira-dataset.json`: 브라우저에서 열 수 있는 사진·정답 묶음.
- `sources.json`: 각 원본 URL, 메타데이터의 라이선스, 파일 해시, 변환 내용.
- `person_keypoints_val2017.json`: 원본 정답 자료. 사전학습 모델이나 실행 코드가 아님.
- `training-report.json`: 실제로 실행한 자체 학습·검증 결과.

이미지 출처는 COCO 2017의 Flickr 사진입니다. 선택 시 COCO 메타데이터의 license id 4(CC BY 2.0)만 포함합니다. 출처 메타데이터가 제공하는 Flickr URL도 유지합니다. 이미지별 원작자 표기는 원본 Flickr 페이지에서 추가 확인해야 하며, 이 목록이 출시 권리 검토를 완료했다는 의미는 아닙니다. 현재 원본 사진은 Git 저장소나 공개 앱에 배포하지 않습니다.

정답 관절은 COCO Consortium의 CC BY 4.0 주석을 사용하고 Movira의 19개 표시점 순서로 변환합니다. 눈·귀를 별도로 표시하지 않으며, 목과 골반 중심은 각각 어깨·고관절 중점으로 만든 가상 기준점입니다. 보이는 점(v=2)과 위치가 주석된 가려진 점(v=1)을 구별합니다. 가려진 점의 좌표는 낮은 가중치로 학습하며, 주석 자체가 없는 점(v=0)은 학습에서 제외합니다. COCO에 없는 뒤꿈치·발끝은 정답으로 꾸며 넣지 않습니다. 자체 정답 편집기로 추가할 수 있습니다.

이전 관절 편집용 수집은 조건에 맞는 257장, 인물 790명입니다. 가상 목·골반을 포함해 보이는 좌표 7,696개와 가려졌지만 위치가 주석된 좌표 1,311개입니다. 이전 자체 CNN의 기록과 현재 인식 방식을 구분합니다. 원작자 메타데이터를 자동 확인하지 못한 사진은 `creditStatus`에 남깁니다.

현재 인물 판별 보정에는 `detection-train/`의 COCO train2017 사진 1,400장을 사용합니다. 학습 1,120장, 검증 140장, 내부 시험 140장입니다. 각 사진의 이미지 URL·Flickr URL·라이선스 ID·SHA-256·사람 영역을 `manifest.json`에 저장합니다. BGRA는 Windows 기본 이미지 기능으로 디코딩했습니다. 보정층은 사람 영역으로만 학습하며 이 작업이 새 관절 신경망 학습을 뜻하지는 않습니다.

`independent-val/`은 새로 수집한 COCO val2017 사진 150장과 평가 대상 인물 영역 329개입니다. 이번 보정층 학습·선택 및 이전 257장과 분리합니다. 매우 작은 사람과 군중 영역은 평가 시 무시합니다. `scale-training/`에는 실제 학습 전체 결과와 GPU 검증 결과를 보관합니다. [검증 조건과 수치](SCALE-VALIDATION.md)

자료 이용 조건은 코드 의존성과 구별합니다. 외부 코드·학습 가중치를 사용하지 않아도, 수집 자료 자체의 이용 조건은 유지됩니다.

- [COCO 원본 이용 조건](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)
- [이미지 CC BY 2.0 조건](https://creativecommons.org/licenses/by/2.0/)
- [정답 CC BY 4.0 조건](https://creativecommons.org/licenses/by/4.0/)

COCO-WholeBody의 발 주석은 별도 비상업적 사용 조건이 명시되어 있어 이번 수집에 포함하지 않았습니다. [원본 조건](https://github.com/jin-s13/COCO-WholeBody#terms-of-use)

## 손 데이터

`Movira_test_Video/hands/`에는 HaGRIDv2에서 수집한 사진 2,240장, 파일별 SHA-256·출처 기록, 원배포 고지를 보관합니다. 이 사진에서 손/배경 영역 3,316개를 만들었습니다. 학습 2,362개(손 1,833·배경 529), 검증 455개, 개발 중 분리 시험 499개입니다. 제공자의 `user_id`로 학습·검증·시험 사이 인물 중복을 검사했습니다. 자동 생성된 연령·성별·인종 항목은 이 작업에 사용하지 않으며 보관용 주석 파일에서도 제거했습니다.

손 관절 좌표는 데이터 제공자의 MediaPipe 자동 주석입니다. 손용 외부 실행 코드나 가중치를 가져오지 않았지만, 외부에서 생성된 정답 데이터를 사용했다는 점을 명시합니다. 출처나 정답 생성 방식을 직접 촬영·직접 수작업 주석한 자료로 표현하지 않습니다.

학습 모델 선택 후 기존 수집 자료의 인물 ID와 겹치지 않는 사진 160장을 추가 수집했습니다. `independent-sources.json`, `independent-hand-samples.json`, `independent-hands64.bgra`에 보관하며, 손 영역 검사와 몸에서 손을 찾는 전체 경로를 구분해 검사합니다. 상세 관절 예측과 원본 사진은 Git에 올리지 않습니다.

고지: [HaGRID 공식 자료](https://github.com/hukenovs/hagrid), [실제 라이선스 원문](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf). 데이터 저자: Alexander Kapitanov, Andrey Makhlyarchuk, Karina Kvanchiani, Aleksandr Nagaev, Roman Kraynov, Anton Nuzhdin. 표준 CC BY-SA라고 축약하지 않고 원문의 별도 조건을 유지합니다.


2026-11-18 손 영역 분류·방향 제안의 추가 학습은 기존 HaGRID 손/배경 3,316개와 같은 사람별 분할을 사용했습니다. 새 원본 사진 수집으로 표현하지 않습니다. `hands/locator-training.json`에 실제 35회 학습 모델과 기록을 보관합니다. `hands/hand-search-regression-full.json`은 이전 코드와 개선 코드의 160장 및 사용자 예제 비교입니다. Pinterest 사진 모음과 YouTube 관절 시연은 구조·움직임 참고용이며 해당 픽셀을 모델 학습에 사용하지 않았습니다. 사용자 첨부 사진도 회귀 검사에만 사용했습니다. [추가 학습·검증](HAND-SEARCH-VALIDATION.md)
