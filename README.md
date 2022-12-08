## 데이터 제작 프로젝트
### 학습 데이터 추가 및 수정을 통한 이미지 속 글자 검출 성능 개선 대회
<img width="816" alt="dataan" src="https://user-images.githubusercontent.com/70750888/206357361-2117f476-fe81-4447-8f4b-f6406b191b24.png">


### Overview
스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

- 본 대회에서는 '글자 검출' task 만을 해결하게 됩니다.
- 예측 csv 파일 제출 (Evaluation) 방식이 아닌 model checkpoint 와 inference.py 를 제출하여 채점하는 방식입니다.
- Input : 글자가 포함된 전체 이미지
- Output : bbox 좌표가 포함된 UFO Format