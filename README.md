# Sogang Humanoid Project

Berkeley Humanoid Lite를 기반으로 보행, 대화, 시각 인지 기능을 통합하는 휴머노이드 프로젝트입니다. 시각 인지 분야의 현재 우선 과제는 카메라 영상에서 사람의 자세를 추정하고 낙상 여부를 판단해 로봇 시스템에 전달하는 것입니다.

## 목표 시스템

```text
카메라 영상
  -> 사람 자세 추정
  -> 시간에 따른 골격 변화 분석
  -> NORMAL / FALLING / FALLEN 판정
  -> ROS를 통한 로봇 행동 시스템 전달
```

최종 운용 환경은 Linux 기반 로봇 컴퓨터와 NVIDIA Jetson Orin Nano 계열 장치를 고려합니다. 로봇의 저수준 보행 제어와 낙상 감지 프로그램은 분리하고, ROS 메시지로 결과를 연동하는 구조를 목표로 합니다.

## 주요 디렉터리

| 경로 | 역할 |
| --- | --- |
| `Berkeley-Humanoid-Lite/` | 휴머노이드 설계 및 보행 관련 코드 |
| `exaone_finetuning/` | 대화형 LLM 관련 코드 |
| `openpose/` | CMU OpenPose 소스와 실행 환경 |
| `fall_detection/` | 영상 골격 추출 및 낙상 감지 개발 코드 |

OpenPose 자체는 낙상 여부를 판정하지 않습니다. OpenPose가 생성한 관절 좌표와 신뢰도를 이용해 별도의 규칙 기반 또는 학습 기반 낙상 감지기를 구현해야 합니다.

## 현재 상태

- Windows에서 OpenPose BODY_25 골격 추출 확인
- 녹화 영상에서 OpenPose JSON 및 NumPy 형식의 관절 데이터 생성 확인
- 처리된 골격 영상 생성 및 결과 프레임 검증 완료
- 규칙 기반 낙상 판정, 개발 세트 평가, Windows 실시간 카메라 프로토타입 구현 완료
- Linux OpenPose 실행 경로 자동 탐색 지원; Linux 빌드와 독립 테스트 데이터는 준비 필요
- Linux CPU용 YOLO pose 실행기와 기존 낙상 상태 머신 연결 완료
- Jetson용 TensorRT 실행 경로, 사전점검 및 벤치마크 도구 준비 완료; 실제 Orin Nano 측정과 ROS 통합 필요

현재 단계는 규칙 기반 실시간 기준선과 공개 데이터셋 GRU 영상 분류 기준선을 구현한 프로토타입입니다. GMDCSA-24의 미학습 Subject 4에서 앙상블 정확도 73.0%, 낙상 재현율 94.1%를 얻었지만 영상 37개뿐인 단일 사람 결과입니다. Linux/Jetson 실시간 운용과 시간당 오경보는 별도 검증이 필요합니다.

## 먼저 읽을 문서

- [낙상 감지 모듈 실행 방법](fall_detection/README.md)
- [현서 담당 개발 이력 및 상세 계획](HISTORY_AND_PLAN_HYEONSEO.md)
- [Berkeley Humanoid Lite 안내](Berkeley-Humanoid-Lite/README.md)
- [OpenPose 안내](openpose/README.md)

각 기능의 상세 설치법과 사용법은 해당 디렉터리의 README에 기록합니다. 개인별 작업 과정, 조사 내용, 시행착오 및 장기 계획은 담당자의 이력·계획 문서에 기록합니다.

## 데이터 및 개인정보 관리

다음 항목은 공개 Git 저장소에 커밋하지 않습니다.

- 원본 및 처리된 촬영 영상
- 실명이 포함된 `participants.csv`
- 한글 원본 입력이나 개인정보가 포함된 `labels.csv`
- Python 가상환경과 캐시 파일
- 다운로드한 OpenPose 바이너리, 모델 외 생성물

공유가 필요한 데이터는 사람을 `p01`, `p02` 같은 익명 ID로 바꾸고, 저장소 밖의 별도 데이터 저장소에서 버전과 출처를 관리합니다.

## 문서 운영 원칙

- 루트 README: 모든 참여자가 반드시 알아야 할 목표, 구조, 실행 진입점만 기록
- 모듈 README: 해당 기능의 설치법, 명령어, 입출력 형식 기록
- 개인 이력·계획 문서: 담당자의 상세 진행 과정, 판단 근거, 향후 계획 기록

코드나 문서를 변경할 때는 기존 사용자 작업을 보존하고, 검증 결과를 확인한 뒤 커밋합니다.
