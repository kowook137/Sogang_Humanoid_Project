# 작업 기록

최종 갱신: 2026-08-21 (KST)

## 현재 초점

EXAONE 4.0 1.2B를 이용한 한국어 휴머노이드 대화 모델 제작을 이어갈 단계다.
아직 로봇 전용 데이터 학습을 시작한 상태는 아니며, 기존 범용 QLoRA 예제를
재현 가능한 대화·기록·학습 파이프라인으로 정비하는 것이 다음 작업이다.

## 2026-08-17 오후 11시 전후에 확인된 작업

- 프로젝트 루트에 Python 3.13 가상환경 `venv/`를 생성했다.
- PyTorch와 `exaone_finetuning/requirements.txt`의 패키지를 설치했다.
- PyTorch, Transformers, Datasets, PEFT, TRL import를 확인했다.
- `exaone_finetuning/inference.py`를 실행했다.
- 22:54:13에 `inference.py`가 수정됐다.
- 23:06:03에 `python -m py_compile exaone_finetuning/inference.py`를 실행한
  흔적이 생성됐다.

당시 확인된 주요 명령은 다음과 같다.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r exaone_finetuning/requirements.txt
python -c "import torch, transformers, datasets, peft, trl; print('all imports ok')"
python exaone_finetuning/inference.py
python -m py_compile exaone_finetuning/inference.py
python exaone_finetuning/inference.py
```

## EXAONE 코드 상태

- 베이스 모델: `LGAI-EXAONE/EXAONE-4.0-1.2B`
- 학습 방식: 4-bit QLoRA 예제
- 기존 학습 데이터: `silk-road/ChatML-Bactrian-ko`
- 어댑터 기본 경로: `./exaone-4.0-1.2b-finetuned`
- `inference.py`에는 `presence_penalty`를 지원하지 않는 Transformers 조합을
  위한 fallback이 미커밋 변경으로 남아 있다.
- 현재 추론 프로그램은 질문을 매번 독립적으로 처리한다. 다중 턴 기억과
  대화 자동 저장은 아직 구현되지 않았다.
- EXAONE 실행 중 터미널에서 주고받은 과거 문답은 저장되지 않아 복구하지
  못했다.

## 2026-08-18 이어서 완료한 작업

- `chat_session.py`로 모델과 독립적인 대화 세션 계층을 추가했다.
- 이전 사용자·어시스턴트 문답을 모델에 전달하는 다중 턴 대화를 구현했다.
- 메시지를 세션별 JSONL에 즉시 기록하도록 구현했다.
- `new`, `history`, `sessions`, `load`, `think`, `help`, `exit` 명령을 추가했다.
- 모델, 어댑터, 로그, 시스템 프롬프트, 생성 길이를 CLI 인자로 분리했다.
- 휴머노이드 역할과 안전 원칙을 `system_prompt.txt`로 분리했다.
- `chat_logs/`를 Git에서 제외했다.
- 모델 다운로드가 필요 없는 세션 단위 테스트 4개가 통과했다.
- Python 문법 검사와 `inference.py --help` 실행을 확인했다.

## 2026-08-20 사투리 선택 기능

- 말투 기준을 현대적 사투리, 친근한 존댓말, 강도 보통, 자연스러운 청년
  말투로 정했다.
- `--dialect standard|gyeongsang|jeolla|chungcheong` 옵션을 추가했다.
- 경상도, 전라도, 충청도 시스템 프롬프트를 별도 파일로 추가했다.
- 특정 어미 반복, 방송식 과장, 다른 지역 말투 혼용을 피하도록 명시했다.
- 긴급 상황에서는 사투리보다 명확한 안전 안내를 우선하도록 했다.
- 사투리 대화 로그를 지역별 디렉터리로 분리하고 JSONL에 지역 값을 기록한다.
- 다른 지역으로 저장한 세션을 잘못 불러오는 것을 차단했다.
- 관련 단위 테스트를 포함해 총 6개 테스트가 통과했다.

## 저장소 주의 사항

현재 `main`에는 EXAONE 변경 외에도 낙상 감지 관련 미커밋 변경과 새 파일이
있다. 이 변경들은 기존 사용자 작업이므로 EXAONE 작업 중 수정하거나 버리지
않는다. 작업 종료 시점에는 커밋하지 않았다.

상태 확인 명령:

```bash
git status --short --branch
git diff -- exaone_finetuning/inference.py
```

## 2026-08-21 경상도 LoRA 데이터 준비

- AI-Hub 경상도 라벨 ZIP을 압축 해제하지 않고 분석했다.
- 부산 주 성장지 10~30대 화자의 표준어·방언 대응 후보를 추출했다.
- 단순 인접 발화 대화 후보는 품질이 낮아 주 학습 자료로 쓰지 않기로 했다.
- 존댓말 판정에서 `그니까`를 `니까` 존댓말로 오인하던 문제를 수정했다.
- weak 자료와 반복 전사를 제외한 `gyeongsang_lora_v1`을 생성했다.
- v1 학습 파일은 1,544개(변환 1,534개, 대화 10개), 검증은 348개다.
- `train.py`가 외부 범용 데이터 대신 로컬 messages JSONL을 읽도록 변경했다.
- 생성 데이터는 원본 라이선스와 용량 관리를 위해 Git에서 제외된다.
- 문법 검사, Git diff 검사, 기존 단위 테스트 6개가 모두 통과했다.
