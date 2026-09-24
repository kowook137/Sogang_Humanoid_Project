# 연구실 GPU 작업 인수인계

## 가장 중요한 실행 원칙

- 이 문서를 읽은 Codex는 어떤 설치나 수정도 하기 전에 반드시 `hostname`, `pwd`, `git status --short --branch`, `nvidia-smi`를 실행한다.
- `hostname`과 GPU 출력이 연구실 Ubuntu 컴퓨터임을 사용자가 확인하기 전에는 패키지를 설치하거나 학습을 시작하지 않는다.
- 기존 작업 트리의 수정 파일과 미추적 파일은 사용자 작업이므로 덮어쓰거나 삭제하지 않는다.
- 기존 개인 컴퓨터 `BOOK-USPOJIF1LF`에서 작업하지 않는다.

## 프로젝트 목표

노인 사용자가 부산 사람과 대화하는 느낌을 받을 수 있는 대화형 로봇을 완성하는 것이 목표다.

- 기반 모델: EXAONE 3.5 7.8B Instruct
- 대화 기능: 일반적인 다중 턴 대화, 사용자 이름·일정 기억
- 안전 기능: 낙상 등 위험 질문에 대한 안전 응답
- 센서 기능: 실제 센서 정보가 없을 때 날씨·실내 온도를 추측하지 않음
- 말투: 모든 문장을 억지 사투리로 바꾸지 않고, 자연스럽고 확실한 현대 부산 표현만 보수적으로 사용
- 음성: 텍스트 사투리 강도보다 부산 억양 TTS가 더 중요한 최종 목표
- 대상: 부산·경남 지역 어르신

## 확정된 설계 방향

1. 기반 LLM이 내용·기억·안전 응답을 생성한다.
2. 규칙 기반 부산말 렌더러가 확실한 표현만 제한적으로 적용한다.
3. TTS 큐에 `busan_contemporary_gentle_elder` 스타일과 부산 억양 메타데이터를 전달한다.
4. 실제 부산 화자 음성 또는 적절한 부산 억양 TTS를 통해 최종 인상을 만든다.
5. 근거가 약한 `~예`, `~노`, 과장된 사투리는 자동으로 남발하지 않는다.

## 저장소 정보

- 저장소: `https://github.com/kowook137/Sogang_Humanoid_Project.git`
- 작업 브랜치: `exaone35-78b-qlora-v3`
- 이전 핵심 커밋: `f90aaa2`
- 주요 코드:
  - `exaone_finetuning/inference.py`
  - `exaone_finetuning/chat_session.py`
  - `exaone_finetuning/robot_policy.py`
  - `exaone_finetuning/busan_style_renderer.py`
  - `exaone_finetuning/tts_queue.py`
  - `exaone_finetuning/prompts/busan.txt`

## 확인된 이전 결과

- 경상도 스타일 v4 LoRA 학습은 280/280 step 완료 이력이 있다.
- 당시 최종 지표는 대략 `eval_loss=0.22595`, `token_accuracy=0.95485`, `train_loss=0.45868`이었다.
- 실제 대화에서는 `마이소`, `보이소`, `주이소` 등이 일부 출력됐지만 모든 상황에서 일관된 부산 말투를 만들지는 못했다.
- 데이터 생성 교사 모델로 EXAONE을 사용했을 때 품질과 통과율이 불안정했다.
- 이후 Gemini 2.5 Pro를 이용한 후보 생성과 사람 검수를 시도했다.
- 최종 방향은 과도한 텍스트 사투리 학습보다 대화 품질·안전·기억을 유지하고 부산 억양 TTS를 결합하는 구조다.

## 보유했던 음성·TTS 자료

Cloud Shell 백업에 다음 파일이 있었다.

- `consent.m4a`
- `busan_reference.m4a`
- `busan_tts_queue.jsonl`
- `busan_tts_audio_test.zip`

이 파일들은 GitHub에 없을 수 있으므로 연구실 컴퓨터에 존재하는지 먼저 확인한다. 없다면 사용자에게 백업 파일 위치를 물어본다.

## 지금 연구실 컴퓨터에서 할 일

### 1. 실행 환경 확인

```bash
hostname
pwd
git status --short --branch
nvidia-smi
python3 --version
df -h .
```

GPU 이름, VRAM, 드라이버 버전을 사용자에게 먼저 보고한다.

### 2. 코드 상태와 테스트 확인

- 저장소의 기존 변경사항을 보존한다.
- `exaone_finetuning`의 대화·정책·부산말·TTS 관련 테스트부터 실행한다.
- 의존성 설치는 별도 가상환경에서 수행한다.
- GPU 종류에 맞는 PyTorch 설치 명령을 선택한다. NVIDIA일 때만 CUDA 빌드를 사용한다.

### 3. 모델 작업 판단

- VRAM이 충분하면 EXAONE 3.5 7.8B 4-bit 추론부터 검증한다.
- 기존 어댑터 파일이 연구실 컴퓨터에 없다면 먼저 다운로드·복구 경로를 확인한다.
- 학습을 바로 재시작하지 말고 기반 모델, 기존 어댑터, 테스트 데이터의 존재 여부를 먼저 점검한다.
- 데이터와 어댑터가 준비된 뒤에만 QLoRA 재학습 또는 추가 학습을 결정한다.

### 4. 완료 기준

- 표준어 질문에도 자연스러운 대화가 가능해야 한다.
- 이름과 일정을 다음 턴에서 기억해야 한다.
- 센서가 없으면 값을 추측하지 않아야 한다.
- 안전 질문은 정확하고 보수적으로 답해야 한다.
- 확실한 경우에만 자연스러운 부산 표현을 사용해야 한다.
- 최종 음성에서 부산 억양이 인지되어야 한다.

## 새 Codex에 보낼 첫 지시문

```text
LAB_GPU_HANDOFF.md를 끝까지 읽고 지침을 따르세요. 먼저 어떤 설치나 수정도 하지 말고 hostname, pwd, git status --short --branch, nvidia-smi, python3 --version, df -h .만 실행하여 이 환경이 연구실 GPU 컴퓨터인지 확인해 주세요. 기존 변경사항은 절대 삭제하거나 덮어쓰지 마세요.
```
