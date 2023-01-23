# 데이터 제작 프로젝트
#### 학습 데이터 추가 및 수정을 통한 이미지 속 글자 검출 성능 개선 대회
<img width="816" alt="dataan" src="https://user-images.githubusercontent.com/70750888/206357361-2117f476-fe81-4447-8f4b-f6406b191b24.png">

<br/> 

## 프로젝트 개요
- 주제
    - OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 구성되지만, 본 대회에서는 **'글자 검출' task** 만을 해결
    - 본 대회는 **데이터를 구성하고 활용하는 방법**에 집중하는 것을 장려하는 취지에서, 제공되는 베이스 코드 중 **모델(EAST)과 관련한 부분을 변경하는 것이 금지**
- 데이터
    - 제공된 학습 데이터: 1510장
    - 평가 데이터 : 크롤링된 이미지 300장  

<br/> 

## 팀 구성
  - 김도윤, 김형석, 박근태, 양윤석, 정선규 (총 5인)  

<br/> 
  
## 프로젝트 상세 내용
  - 데이터 셋 제작 → 학습 데이터의 양이 적었기에, ICDAR 17 5k, ICDAR19 5K 추가
  - Synthetic data set인 UnrealText로 pretrain 후, target data로 fine tuning
  - Multi-scale-crop training → 512~1024 사이로 random crop 후 512로 resize
  - 다양한 글씨체, 글씨 변형에 대한 robustness 부여 위해 Elastic transform 사용  

<br/> 
  
## 모델 개요
  - Model : EAST
  - Optimizer : Adam
  - Scheduler : MultiStepLr
  - Pretrained on Unrealtext  

<br/> 

## 결과
  ||F1 score|순위|
|------|---|---|
|**Public Leaderbord**|0.6839|7/19|
|**Private Leaderbord**|0.6720|8/19|
