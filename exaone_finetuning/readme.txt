# EXAONE 4.0 1.2B Fine-tuning & Inference

이 디렉토리는 LG AI Research의 EXAONE 4.0 1.2B 모델을 한국어 데이터셋으로 파인튜닝하고 추론하기 위한 코드를 포함하고 있습니다.

## 주요 기능
- **EXAONE 4.0 지원**: 최신 transformers 라이브러리를 사용하여 EXAONE 4.0의 아키텍처와 기능을 지원합니다.
- **QLoRA 파인튜닝**: 4비트 양자화와 LoRA를 결합하여 적은 메모리로 효율적인 학습이 가능합니다.
- **추론 모드 (Reasoning Mode)**: EXAONE 4.0의 특화 기능인 <think> 태그를 활용한 추론 과정을 활성화할 수 있습니다.
- **Chat Template 적용**: 모델의 특성에 맞는 채팅 템플릿을 자동으로 적용합니다.
- **다중 턴 대화**: 현재 세션의 이전 문답을 다음 생성 요청에 함께 전달합니다.
- **대화 자동 저장**: 각 문답을 `chat_logs/<session-id>.jsonl`에 즉시 저장합니다.
- **지역 말투 선택**: 표준어, 경상도, 전라도, 충청도 프롬프트를 선택합니다.

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
python exaone_finetuning/inference.py
```
- 기본적으로 `exaone_finetuning/exaone-4.0-1.2b-finetuned`의 어댑터를
  로드하며, 어댑터가 없으면 베이스 모델로 실행됩니다.
- 대화 기록은 개인정보가 포함될 수 있어 Git에 추가되지 않습니다.

지역 사투리를 선택해서 실행합니다.

```bash
python exaone_finetuning/inference.py --dialect gyeongsang
python exaone_finetuning/inference.py --dialect jeolla
python exaone_finetuning/inference.py --dialect chungcheong
```

세 지역 모두 현대적인 사투리, 친근한 존댓말, 보통 강도, 자연스러운 청년
말투를 기준으로 한다. 특정 어미를 반복하거나 방송식으로 과장하지 않으며,
긴급 상황에서는 명확한 안전 안내를 우선한다. 기본값은 `standard`다.

대화 기록은 선택한 지역에 따라 `chat_logs/<dialect>/`에 분리된다.

대화 중 사용할 수 있는 명령:

```text
think              추론 모드 켜기/끄기
new                새 대화 시작
history            현재 대화 보기
sessions           저장된 세션 목록 보기
load <session-id>  이전 세션 불러오기
help               명령 도움말
exit               종료
```

프로그램 시작과 동시에 이전 세션을 불러올 수도 있습니다.

```bash
python exaone_finetuning/inference.py --load-session <session-id>
```

전체 옵션은 다음 명령으로 확인합니다.

```bash
python exaone_finetuning/inference.py --help
```

## 파일 설명
- train.py: QLoRA 기반 파인튜닝 스크립트
- inference.py: 대화형 추론 인터페이스 (추론 모드 지원)
- chat_session.py: 다중 턴 상태, JSONL 저장 및 이전 세션 로드
- system_prompt.txt: 휴머노이드의 역할과 기본 안전 원칙
- prompts/: 경상도, 전라도, 충청도 말투별 시스템 프롬프트
- tests/test_chat_session.py: 모델 다운로드 없이 실행하는 세션 단위 테스트
- requirements.txt: 필요한 라이브러리 목록
- guide.txt: EXAONE 4.0 공식 가이드 및 파라미터 정보
