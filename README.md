# SleepEventNet-tflite
YAMNet의 사전학습된 음향 특징(score)을 기반으로, 수면 이벤트별 패턴을 식별하는 커스텀 Keras 레이어를 설계하고, 해당 후처리 로직을 YAMNet에 통합하여 TFLite로 경량화한 프로젝트입니다.

## 💤 SleepEventNet
수면 중 발생하는 다양한 이벤트(이갈이, 코골이, 기침, 수면 중 말하기 등)를 감지하는 사운드 기반 분류 모델입니다.

## 🔍 프로젝트 특징
YAMNet 기반 사전학습 모델을 활용하여 음성 이벤트의 특징 추출합니다.

커스텀 Keras 레이어를 통해 5가지 수면 이벤트에 대한 점수 출력합니다.

TensorFlow 및 TFLite 지원으로 모바일 환경에서도 활용 가능합니다.

## 🎯 주요 기능
20초 길이의 .wav 오디오 파일을 입력받아 다섯 가지 수면 이벤트 점수를 반환합니다.

출력 이벤트: Bruxism (이갈이) , Snoring (코골이) , Cough (기침) , Sleep Speech (수면 중 말하기) , Normal (정상 상태) 에 대한 Score를 반환합니다.

예시 : 

```
bruxism_score: 11.7600
snoring_score: 0.0000
cough_score: 0.0000
speech_score: 0.0000
is_normal: 0.0000
```

## ⚙️ 환경 설정 (Setup)

```
conda create -n sleep_env python=3.11 -y
conda activate sleep_env
cd [다운받은 경로]
pip install -r requirements.txt
```

## 🔧 사용 방법
예제 실행: build_and_test_sleepnet.ipynb 참고

Python 스크립트 실행:

python main.py
main.py 내에서 분석할 .wav 경로 설정 필요합니다.
예시 :
```
wav_path = 'data/20_seconds/snoring/1/test3.wav'
main(wav_path)
```
TensorFlow 모델 및 TFLite 모델로 둘다 추론 가능합니다.

## ⚠️ 주의 사항
입력 파일은 반드시 20초 길이의 모노 채널1, .wav 파일이어야 하며,
샘플링 레이트는 16,000Hz, 데이터 타입은 float32, 값의 범위는 -1 ~ 1로 조정 필요합니다.

main.py 및 내부 함수에서는 자동 전처리가 포함되어 있으나,
TFLite 모델을 외부에서 사용할 경우 위 조건에 맞는 전처리가 선행되어야 합니다.

## 📐 모델 구조 설명

이 프로젝트는 YAMNet의 사전학습된 음향 특징(score)을 기반으로, **수면 중 발생 가능한 소리 이벤트(이갈이, 코골이, 기침, 수면 중 말하기)**를 식별하기 위한 후처리 레이어를 직접 설계하고, 이를 기존 모델에 통합하여 **경량화(TFLite 변환)**한 작업입니다.

주요 흐름

**1.입력**

20초 길이의 .wav 오디오 파일을 입력으로 받습니다.

**2.특징 추출**

OpenAI의 YAMNet 모델을 통해 오디오로부터 **프레임별 521개의 score(class 확률)**를 추출합니다.

**3.이벤트별 score 조합 정의 및 패턴 탐색**

각 소리 이벤트(이갈이, 코골이, 기침, 수면 중 말하기)에 대해,
해당 클래스에서 score에 자주 등장하는 상위 class 번호와 확률값을 수집합니다.

그 후 Grid Search 기반 랜덤 조합 탐색을 수행하여,
가장 높은 정확도로 해당 이벤트를 분류하는 최적 class 조합 패턴을 자동으로 도출합니다.

이 조합별로 전체 데이터에서의 **정답률을 기반으로 가중치(weight)**를 계산합니다.
(예: 조합 A가 이갈이 데이터 중 80%를 맞췄다면 → 가중치 0.8)

**4.이벤트별 사용자 정의 Keras 레이어 구성**

위에서 정의한 class 조합과 가중치를 기반으로,
각 이벤트(이갈이, 코골이, 기침, 말하기)에 대해
Custom Keras Layer를 구현했습니다.

각 레이어는 YAMNet의 score 출력을 입력 받아,
해당 이벤트의 **발생 여부 및 점수(score)**를 반환합니다.

**5.정상 상태(Normal) 판단**

네 가지 이벤트 중 모두 비발생일 경우, Normal로 분류되며,
정상 상태 여부도 함께 출력됩니다.
정상 상태라고 판단될 떄 점수는 1점으로 고정입니다.

**6.모델 통합 및 TFLite 변환**

YAMNet과 사용자 정의 레이어들을 하나의 모델로 통합하고,

최종적으로 TensorFlow Lite 형식으로 변환하여 모바일 환경에서도 실행 가능하도록 경량화했습니다.


## 🛠 사용 기술 스택 (Tech Stack)
Framework: TensorFlow 2.18

모델 구성 및 학습, 사용자 정의 레이어 설계에 활용

Deployment: TensorFlow Lite (TFLite)

모바일/임베디드 환경용 추론 모델 변환

Audio Processing: Librosa, SciPy

wav 파일 로딩, 샘플링 정규화, 특징 추출 등

Data Handling: NumPy, Pandas

score 분석 및 클래스 조합 탐색(Grid Search)

Visualization: Matplotlib, Seaborn

Class combination 시각화, 결과 확인용
