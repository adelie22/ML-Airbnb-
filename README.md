# 에어비앤비 가격 예측 프로젝트

## 개요
이 프로젝트는 다양한 특징들을 기반으로 에어비앤비 숙소의 가격을 예측하기 위해 설계되었습니다. 프로젝트는 데이터 전처리, 모델 학습, 평가 및 가격 예측을 포함합니다.

## 파일 설명

- **data_preprocessing.py**: CSV 파일들을 결합하고, 열 이름을 변경하고, 불필요한 열을 삭제하며, 결측값을 처리하고 데이터를 전처리하는 함수들을 포함합니다.
- **machine_learning.py**: 특징과 가격 간의 관계를 시각화하고, 선형 회귀, 랜덤 포레스트, 그래디언트 부스팅 모델을 학습 및 평가하며, 학습된 랜덤 포레스트 모델을 사용하여 가격을 예측하는 함수들을 포함합니다.
- **predict_price.py**: 에어비앤비 위치를 시각화하는 인터랙티브 지도를 생성하고, 특징 예측을 위한 랜덤 포레스트 모델을 학습하며, 위도와 경도를 기반으로 특징 값을 예측하는 함수들을 포함합니다.
- **main_menu.py**: 데이터 로드, 시각화, 모델 학습, 평가 및 가격 예측 등 프로젝트의 다양한 기능과 상호 작용할 수 있는 메뉴 기반 인터페이스를 제공합니다.
- **testing.py**: 데이터 로드, 전처리, 모델 학습 및 평가, 가격 예측 방법을 테스트하는 함수들을 포함합니다.

## 설치 및 설정

### 필수 조건

- Python 3.7 이상
- 필요한 라이브러리: `pandas`, `numpy`, `matplotlib`, `seaborn`, `folium`, `scikit-learn`

### 설치

1. 리포지토리를 클론하거나 프로젝트 파일을 다운로드합니다.
2. pip를 사용하여 필요한 라이브러리를 설치합니다:

   ```bash
   pip install pandas numpy matplotlib seaborn folium scikit-learn

### 데이터
- 데이터 파일이 프로젝트 루트에 `data`라는 디렉토리에 있어야 합니다. 각 CSV 파일은 `city_day.csv` 형식을 따라야 합니다.
- 데이터 링크: https://zenodo.org/records/4446043

## 사용법

## 메뉴 인터페이스 실행

1. main_menu.py 스크립트를 실행하여 메뉴 인터페이스를 시작합니다:

   ```bash
   python main_menu.py

2. 화면의 지시에 따라 데이터를 로드하고, 관계를 시각화하며, 모델을 학습하고 평가하며, 가격을 예측하고, 에어비앤비 위치를 시각화합니다.

### 메뉴 항목 설명

#### 주의사항

- 팝업으로 나오는 시각화 창들을 닫지 않으면 프로그램이 진행이 안됩니다. 
- 메뉴 순서대로 진행을 해야지 오류가 없습니다. 예) 데이터를 준비하는 1번을 실행하지 않고 3번을 실행한다면 먼저 1번을 실행하라고 나옵니다. 
- 3,4,5번은 room type을 입력하라고 나오는데 'Private room' 이나 'Entire home/apt' 둘중 하나를 입력하시면 됩니다.

1. **Load and preprocess data** - CSV 파일을 로드하고 전처리하여 데이터를 준비합니다.
    - 데이터 파일을 로드합니다.
    - 불필요한 열을 제거하고 결측값을 처리합니다.
    - 데이터를 병합하고 전처리합니다.
2. **Visualize relationships** - 다양한 특징과 가격 간의 관계를 시각화합니다.
    - 특징과 가격 간의 관계를 나타내는 산점도를 생성합니다.
3. **Train and evaluate Linear Regression model** - 선형 회귀 모델을 학습하고 평가합니다.
    - 학습 데이터를 사용하여 선형 회귀 모델을 학습합니다.
    - 테스트 데이터를 사용하여 모델을 평가합니다.
4. **Train and evaluate Random Forest model** - 랜덤 포레스트 모델을 학습하고 평가합니다.
    - 학습 데이터를 사용하여 랜덤 포레스트 모델을 학습합니다.
    - 테스트 데이터를 사용하여 모델을 평가합니다.
    - 추가로 가격 예측할 떄 사용할 특징들을 예측하는 모델도 생성합니다.
        - long과 lat을 사용해 attr_index_norm, rest_index_norm, metro_dist, center_dist 예측하는 모델 
5. **Train and evaluate Gradient Boosting model** - 그래디언트 부스팅 모델을 학습하고 평가합니다.
    - 학습 데이터를 사용하여 그래디언트 부스팅 모델을 학습합니다.
    - 테스트 데이터를 사용하여 모델을 평가합니다.
6. **Predict price** - 입력한 정보를 바탕으로 에어비앤비 숙소의 가격을 예측합니다. 7번에서 지도를 보고 원하는 위치의 위도와 경도를 확인할 수 있습니다. 
    - **Latitude (위도)**: 숙소의 위도 좌표입니다. 
    - **Longitude (경도)**: 숙소의 경도 좌표입니다.
    - **Number of Bedrooms (침실 수)**: 숙소에 있는 침실의 개수입니다. 0~4중에 입력하세요. 
    - **City Name (도시 이름)**: 숙소가 위치한 도시의 이름입니다. 총 10개 도시중에 입력하세요: amsterdam, athens, barcelona, berlin, budapest, lisbon, london, paris, rome, vienna.
    - **Person Capacity (수용 인원)**: 숙소가 수용할 수 있는 최대 인원 수입니다. 2~5중에 입력하세요. 
7. **Visualize Airbnb locations** - 에어비앤비 숙소의 위치를 인터랙티브 지도에 시각화합니다.
    - 숙소의 위치를 나타내는 마커를 지도에 추가합니다.
    - 추가로 클릭 했을떄 해당 지점의 경도와 위도를 나타내 줌으로써 가격 예측할 떄 이 정보를 사용할 수 있습니다. 
8. **Exit** - 프로그램을 종료합니다.
    - 메뉴 인터페이스를 종료합니다.
   
## 테스트 실행

1. testing.py 스크립트를 실행하여 데이터 로드, 전처리, 모델 학습, 평가 및 가격 예측 방법을 테스트합니다

    - 팝업으로 나오는 창들을 닫아야지 테스트가 계속 진행됩니다. 
    - Preprocessing test passed, All model accuracy test passed, Prediction successful 나오면서 테스트가 완료됩니다. 

   ```bash
   python testing.py





