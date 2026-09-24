# 다음 작업

## 재개 방법

```bash
cd /home/hanseo501/projects/Sogang_Humanoid_Project
source venv/bin/activate
git status --short --branch
```

새 Codex 대화에서는 다음과 같이 요청한다.

> `WORK_LOG.md`와 `NEXT_STEPS.md`, Git 상태를 읽고 EXAONE 대화 모델 작업을
> 이어가줘. 기존 미커밋 변경은 덮어쓰지 마.

## 우선순위

1. `gyeongsang_lora_v1`으로 짧은 파일럿 QLoRA 학습을 실행한다.
2. 고정 평가 질문으로 베이스 모델과 경상도 어댑터를 비교한다.
3. 말투 재현성, 답변 능력 저하, 과장된 어미 반복을 평가한다.
4. 대화 SFT 10개를 다양한 주제로 확장해 변환 자료 편중을 줄인다.
5. 실제 시험 대화를 수집하되 원본과 검수 완료 데이터를 분리한다.
6. 긴 대화에서 컨텍스트 길이를 제한하는 정책을 추가한다.

데이터 재생성:

```bash
python exaone_finetuning/build_gyeongsang_lora_dataset.py
```

학습 실행(가상환경과 CUDA 확인 후):

```bash
python exaone_finetuning/train.py
```

## 첫 번째 구현 완료 조건

- 코드와 단위 테스트 기준으로 완료했다.
- 실제 모델 생성 품질과 GPU/메모리 동작은 다음 실행에서 확인한다.

## 보류된 결정

- 로봇의 이름과 구체적인 성격
- 주 사용 대상과 허용할 대화 범위
- 음성 입출력 및 ROS 연결 시점
- 수집 대화를 학습 데이터로 채택하는 검수 기준
