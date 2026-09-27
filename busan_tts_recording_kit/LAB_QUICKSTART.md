# 연구실 GPU 빠른 재개 절차

목표는 VS Code/Codex 로그아웃이나 컴퓨터 변경 후에도 GitHub만으로 같은 작업을 복구하는 것입니다. 명령은 저장소 루트에서 실행합니다.

## 처음 받는 컴퓨터

```bash
git clone --depth 1 --branch exaone35-78b-qlora-v3 https://github.com/kowook137/Sogang_Humanoid_Project.git
cd Sogang_Humanoid_Project
bash busan_tts_recording_kit/lab_tts.sh status
```

`status`는 설치 없이 GPU, 녹음 수, 환경, 가중치와 로그를 한 번에 보여줍니다.

## 최초 한 번만 설치

Conda와 NVIDIA 드라이버가 설치된 Ubuntu에서 다음 한 줄을 실행합니다.

```bash
bash busan_tts_recording_kit/lab_tts.sh setup
```

공식 GPT-SoVITS `20250606v2pro`의 정확한 커밋, `GPTSoVits` 환경, 사전학습 모델을 설치하고 CUDA 접근을 검증합니다. 다운로드는 시간이 걸릴 수 있지만 명령을 다시 조립할 필요는 없습니다. 완료 표식이 있으면 재실행 시 설치를 건너뜁니다.

## 녹음이 GitHub에 올라온 뒤

```bash
git pull --ff-only origin exaone35-78b-qlora-v3
bash busan_tts_recording_kit/lab_tts.sh prepare
bash busan_tts_recording_kit/lab_tts.sh start
```

`prepare`는 `status=ready`, `transcript_verified=true`이고 WAV가 실제로 존재하는 행만 공식 `vocal_path|speaker_name|language|text` 형식으로 변환합니다. WebUI에서 버전은 `v2Pro`, 실험 이름은 `busan_voice_v1`로 고정하고, 먼저 SoVITS와 GPT가 각각 1 epoch를 완료하는지만 확인합니다.

## 로그아웃 전에 반드시 실행

```bash
bash busan_tts_recording_kit/lab_tts.sh save-audio
bash busan_tts_recording_kit/lab_tts.sh backup-results
```

`backup-results`는 로그인된 GitHub CLI(`gh`)를 사용합니다. 일반 Git은 파일당 100 MiB 제한이 있으므로 큰 학습 결과를 커밋하지 않고 `busan-tts-backup` Release에 압축 파일과 SHA-256을 올립니다.

복구 범위는 다음과 같습니다.

- GitHub 브랜치: 코드, 대본, 매니페스트, 녹음과 설정
- GitHub Release: GPT/SoVITS 학습 가중치와 실험 로그
- 자동 재다운로드: 공식 GPT-SoVITS 코드와 사전학습 모델
- 저장하지 않음: Conda 캐시와 재설치 가능한 패키지

로그아웃 자체는 Ubuntu 파일을 지우지 않지만 컴퓨터 초기화나 계정 정리에 대비해 학습 직후 `backup-results`를 실행해야 합니다.
