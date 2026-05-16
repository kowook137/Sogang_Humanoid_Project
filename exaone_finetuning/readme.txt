# EXAONE 4.0 1.2B Fine-tuning & Inference

이 디렉토리는 LG AI Research의 EXAONE 4.0 1.2B 모델을 한국어 데이터셋으로 파인튜닝하고 추론하기 위한 코드를 포함하고 있습니다.

## 주요 기능
- **EXAONE 4.0 지원**: 최신 transformers 라이브러리를 사용하여 EXAONE 4.0의 아키텍처와 기능을 지원합니다.
- **QLoRA 파인튜닝**: 4비트 양자화와 LoRA를 결합하여 적은 메모리로 효율적인 학습이 가능합니다.
- **추론 모드 (Reasoning Mode)**: EXAONE 4.0의 특화 기능인 <think> 태그를 활용한 추론 과정을 활성화할 수 있습니다.
- **Chat Template 적용**: 모델의 특성에 맞는 채팅 템플릿을 자동으로 적용합니다.

## 설치 방법
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 학습 (Fine-tuning)
train.py를 실행하여 silk-road/ChatML-Bactrian-ko 데이터셋으로 모델을 학습시킵니다.
```bash
python train.py
```
학습된 어댑터는 ./exaone-4.0-1.2b-finetuned 폴더에 저장됩니다.

### 2. 추론 (Inference)
inference.py를 실행하여 학습된 모델과 대화할 수 있습니다.
```bash
python inference.py
```
- 기본적으로 ./exaone-4.0-1.2b-finetuned의 어댑터를 로드합니다. (학습 전이라면 베이스 모델로 실행됩니다)
- think 명령어를 입력하여 추론 모드(Reasoning Mode)를 켜고 끌 수 있습니다.

## 파일 설명
- train.py: QLoRA 기반 파인튜닝 스크립트
- inference.py: 대화형 추론 인터페이스 (추론 모드 지원)
- requirements.txt: 필요한 라이브러리 목록
- guide.txt: EXAONE 4.0 공식 가이드 및 파라미터 정보
