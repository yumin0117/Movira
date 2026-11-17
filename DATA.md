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

이번 수집: 조건에 맞는 257장, 인물 790명. 가상 목·골반을 포함해 보이는 좌표 7,696개와 가려졌지만 위치가 주석된 좌표 1,311개입니다. 각 사진의 모든 원본 인물 박스를 배경 표본 제외 영역으로 사용해, 가려져 관절 주석이 적은 사람을 배경 정답으로 잘못 학습하는 것을 줄입니다. 원작자 메타데이터를 자동 확인하지 못한 사진은 `creditStatus`에 남깁니다.

자료 이용 조건은 코드 의존성과 구별합니다. 외부 코드·학습 가중치를 사용하지 않아도, 수집 자료 자체의 이용 조건은 유지됩니다.

- [COCO 원본 이용 조건](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)
- [이미지 CC BY 2.0 조건](https://creativecommons.org/licenses/by/2.0/)
- [정답 CC BY 4.0 조건](https://creativecommons.org/licenses/by/4.0/)

COCO-WholeBody의 발 주석은 별도 비상업적 사용 조건이 명시되어 있어 이번 수집에 포함하지 않았습니다. [원본 조건](https://github.com/jin-s13/COCO-WholeBody#terms-of-use)
