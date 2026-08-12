# 현서 담당 낙상 감지 개발 이력 및 계획

최종 갱신일: 2026-08-11

## 1. 프로젝트 목표

이 프로젝트의 최종 목표는 Berkeley Humanoid Lite에 장착된 Depth Camera와 NVIDIA Jetson Orin Nano를 사용하여 사람의 낙상을 온디바이스로 감지하는 것이다.

목표 처리 흐름은 다음과 같다.

```text
Depth Camera
    -> 영상 입력
    -> 사람 자세 추정(Pose Estimation)
    -> 관절 좌표 추출
    -> 시간에 따른 자세 변화 분석
    -> NORMAL / FALLING / FALLEN 판정
    -> ROS를 통해 로봇 행동 시스템에 결과 전달
```

낙상 감지 중 로봇이 움직이면 카메라 자체 움직임과 사람의 움직임이 섞일 수 있다. 첫 번째 버전은 로봇이 정지하고 진동이 안정된 상태에서 2~3초 동안 사람을 관찰한다고 가정한다. 이동 중 실시간 감지는 이후 IMU, 영상 안정화 또는 optical flow를 이용한 카메라 움직임 보정을 추가하여 확장한다.

## 2. 저장소 구성

현재 저장소의 주요 폴더는 다음과 같다.

```text
Sogang_Humanoid_Project/
├── Berkeley-Humanoid-Lite/  # 휴머노이드 플랫폼 및 보행 관련 코드
├── exaone_finetuning/       # 대화형 LLM 파인튜닝 관련 코드
├── openpose/                # CMU OpenPose 1.7.0 소스
├── README.md                # 팀 전체가 공유할 프로젝트 핵심 안내
└── HISTORY_AND_PLAN_HYEONSEO.md  # 현서 담당 작업의 상세 이력과 계획
```

`openpose/`는 CMU OpenPose 1.7.0 소스 트리다. 주요 구성은 다음과 같다.

- `src/`: OpenPose C++ 구현
- `include/openpose/`: 공개 C++ API
- `examples/openpose/openpose.cpp`: C++ 데모 진입점
- `examples/tutorial_api_python/`: Python API 예제
- `python/openpose/`: `pyopenpose` 바인딩 소스
- `models/`: 네트워크 정의 및 모델 다운로드 스크립트
- `doc/`: 설치, 실행, 출력 형식 문서
- `3rdparty/`: Caffe 및 관련 의존성

OpenPose 자체에는 낙상 판단 알고리즘이 없다. OpenPose는 사람의 관절 좌표와 confidence를 제공하고, 낙상 감지는 별도 모듈에서 구현해야 한다.

## 3. 지금까지 완료한 작업

### 3.1 OpenPose 분석

- OpenPose 버전 1.7.0 확인
- Windows 실행 구조와 C++/Python 진입점 확인
- BODY_25 관절 출력 형식 확인
- 기존 코드에 낙상 판단 로직이 없음을 확인
- 원본 소스 체크아웃에는 실행 파일, Python 바인딩, 모델 가중치가 없음을 확인

### 3.2 Windows OpenPose 실행 환경

소스 빌드는 기존 배포 서버의 HTTP 오류와 오래된 Caffe 의존성 문제로 중단했다. 대신 공식 OpenPose 1.7.0 Windows GPU 포터블 패키지를 사용했다.

현재 실행 환경:

```text
openpose/openpose-portable/openpose/
├── bin/OpenPoseDemo.exe
├── models/pose/body_25/pose_iter_584000.caffemodel
├── examples/
└── 3rdparty/
```

BODY_25 모델 검증 결과:

```text
파일 크기: 104,715,850 bytes
MD5: 78287B57CF85FA89C03F1393D368E5B7
```

완료된 실행 검증:

- RTX 2070 SUPER를 OpenPose가 정상 인식
- 샘플 이미지에서 BODY_25 골격 렌더링 성공
- 카메라 0번에서 실시간 골격 추출 성공
- 테스트 출력 이미지 생성 성공

카메라 실행 명령:

```powershell
cd openpose\openpose-portable\openpose

.\bin\OpenPoseDemo.exe `
  --camera 0 `
  --model_folder .\models `
  --model_pose BODY_25 `
  --net_resolution "-1x256"
```

OpenPose 창에서 `Esc`를 누르면 종료된다. 카메라가 여러 개라면 `--camera 0`을 `1`, `2` 등으로 변경한다.

### 3.3 Windows 학습 환경

Miniconda 기반 전용 환경을 구성하고 GPU 연산까지 확인했다.

```text
Conda 환경: fall-detection
Python: 3.10.20
PyTorch: 2.5.0+cu124
CUDA Runtime: 12.4
GPU: NVIDIA GeForce RTX 2070 SUPER
VRAM: 8GB
OpenCV: 5.0.0
NumPy: 2.2.6
pandas: 2.3.3
scikit-learn: 1.7.2
```

현재 PC에서는 Conda의 `fall-detection` 환경을 사용한다. VS Code에서는 해당 환경의 Python 인터프리터를 선택하고, Jupyter에서는 `Python (fall-detection)` 커널을 선택한다.

환경 활성화:

```powershell
conda activate fall-detection
```

GPU 확인:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

정상 결과:

```text
True
NVIDIA GeForce RTX 2070 SUPER
```

### 3.4 연구 결과보고서 검토

2026학년도 1학기 연구 결과보고서에서 다음 내용을 확인했다.

- 최종 하드웨어는 Berkeley Humanoid Lite + Depth Camera + Jetson Orin Nano
- OpenPose로 주요 관절을 추출
- 머리와 골반 높이의 급격한 하강 정도 사용
- 몸체가 수직에서 수평에 가까워지는 정도 사용
- 위 특징을 weighted sum으로 결합
- 보고된 낙상 감지 성능은 AUROC 87%
- 2026학년도 2학기에 ROS 기반으로 각 모듈을 통합할 계획
- 보고서에는 OpenPose, Exaone LLM, 보행 모델을 Orin Nano에서 온보드 구동했다고 기재

그러나 보고서에는 다음 재현 정보가 없다.

- Depth Camera 모델명과 설치 위치
- RGB만 사용했는지 실제 depth 값을 사용했는지
- OpenPose 입력 및 네트워크 해상도
- 영상 FPS
- 데이터 수, 촬영 인원, 라벨 기준
- 학습/검증/테스트 분리 방식
- 시간 윈도 길이
- 특징 정규화 공식
- 가중치와 최종 임곗값
- ROC 곡선과 AUROC 계산 코드
- Jetson 처리속도 및 메모리 사용량

따라서 AUROC 87%는 목표 및 참고 결과로 사용하되, 원본 코드와 데이터가 없다면 자체 데이터로 다시 검증해야 한다.

## 4. 낙상 감지에 사용할 관절과 특징

OpenPose BODY_25 전체 관절을 저장하고, 첫 번째 규칙 기반 감지기에서는 다음 관절을 우선 사용한다.

- Nose
- Neck
- Left/Right Shoulder
- MidHip 또는 Left/Right Hip
- Left/Right Knee
- Left/Right Ankle

파생 중심점:

```text
hip_center = (left_hip + right_hip) / 2
shoulder_center = (left_shoulder + right_shoulder) / 2
```

관절 confidence가 기준보다 낮으면 해당 좌표를 0으로 정상 처리하지 않는다. invalid/missing으로 표시하고, 짧은 누락은 보간하되 긴 누락에서는 판단을 보류한다.

계획된 특징:

1. 머리의 수직 이동량과 속도
2. 골반 중심의 수직 이동량과 속도
3. 어깨 중심과 골반 중심을 연결한 몸통 벡터의 각도
4. 관절 바운딩 박스의 폭/높이 비율
5. 신체 높이의 급격한 감소
6. 낮은 자세의 지속시간
7. 선택적으로 depth 기반 실제 거리 및 높이 변화

영상 좌표는 오른쪽이 `+x`, 아래쪽이 `+y`인 것이 일반적이다. 구현과 테스트에서 이 좌표 규약을 명시적으로 확인해야 한다.

## 5. 권장 모듈 구조

Windows와 Jetson, 영상 파일과 카메라 입력을 쉽게 교체할 수 있도록 다음 경계를 유지한다.

```text
VideoFileSource / CameraSource / DepthCameraSource
                    |
                    v
              PoseEstimator
                    |
                    v
            KeypointProcessor
                    |
                    v
              FallDetector
                    |
                    v
        Visualizer / Logger / ROS Publisher
```

처음부터 복잡한 클래스 계층을 만들 필요는 없다. 핵심은 입력 소스, 자세 추정, 낙상 판단을 서로 직접 결합하지 않는 것이다.

PoseEstimator의 공통 출력은 다음 형태로 유지한다.

```python
keypoints.shape == (number_of_people, 25, 3)
# 마지막 축: x, y, confidence
```

이 인터페이스를 유지하면 Jetson에서 OpenPose 빌드가 어렵거나 성능이 부족할 때 PoseEstimator만 TensorRT 기반 경량 모델로 교체할 수 있다.

## 6. 구현된 최소 파이프라인

아직 LSTM/TCN 학습이나 최종 낙상 임곗값 결정부터 시작하지 않는다. 다음 최소 파이프라인을 먼저 구현하고 두 테스트 영상으로 검증했다.

```text
입력 MP4
    -> OpenPose BODY_25 실행
    -> 골격이 표시된 결과 MP4 저장
    -> 프레임별 BODY_25 JSON 저장
    -> JSON 폴더를 NumPy 배열로 변환
    -> 좌표 및 confidence 검증
```

목표 NumPy 형식:

```python
keypoints.shape == (frame_count, 25, 3)
```

영상과 JSON을 출력하는 기본 명령:

```powershell
python fall_detection/process_video.py `
  --input "<입력 MP4 경로>" `
  --output-dir "<출력 폴더>"
```

구현 파일은 경로를 하드코딩하지 않고 CLI 인자를 받는다.

```text
python <script>.py --input <video> --output-dir <directory>
```

구현 위치:

```text
fall_detection/process_video.py
```

검증 결과:

```text
p01_fall_side_001
- 원본/JSON/NPZ/렌더링: 521 frames
- keypoints shape: (521, 25, 3)
- 사람 검출: 521/521 frames

p01_normal_walk_sit_001
- 원본/JSON/NPZ/렌더링: 753 frames
- keypoints shape: (753, 25, 3)
- 사람 검출: 753/753 frames
```

휴대폰 세로 영상의 회전 메타데이터를 OpenPose가 무시하는 문제와 Windows OpenPose가 MP4를 직접 기록하지 못하는 문제도 처리 스크립트에서 자동 보정한다.

## 7. 데이터 수집 및 라벨링 계획

최소 파이프라인을 확인한 뒤 소량 데이터로 규칙 기반 감지기를 검증한다.

정상 동작 예시:

- 서기와 걷기
- 의자에 앉기/일어서기
- 빠르게 앉기
- 물건 줍기
- 허리 숙이기
- 무릎 꿇기
- 바닥이나 침대에 자발적으로 눕기
- 카메라 밖으로 이동하기

낙상 예시:

- 앞으로 넘어짐
- 뒤로 넘어짐
- 좌우로 넘어짐
- 걷다가 넘어짐
- 의자에서 떨어짐
- 천천히 주저앉음
- 넘어진 뒤 움직이지 않음
- 넘어진 뒤 바로 일어남

낙상 촬영은 안전 매트, 보호 장비, 보조자와 함께 연출된 동작으로만 수행한다. 고령자나 낙상 위험이 있는 사람에게 실제 낙상 동작을 요청하지 않는다.

권장 라벨 형식:

```csv
video,person_id,action,label,fall_start_frame,impact_frame,fall_end_frame
p01_fall_side_001.mp4,p01,fall_side,fall,65,82,120
p01_normal_sit_001.mp4,p01,sit,normal,,,
```

동일한 사람이 학습 데이터와 테스트 데이터에 동시에 들어가지 않도록 사람 단위로 분리한다.

## 8. 낙상 판단 개발 순서

### 8.1 규칙 기반 기준선

먼저 다음 상태 머신을 구현한다.

```text
NORMAL -> FALLING -> FALL_CANDIDATE -> FALL_CONFIRMED
```

초기 판단 흐름:

```text
빠른 머리/골반 하강
    + 몸통 각도의 급격한 변화
    + 신체 높이 감소
    + 낮은 자세 지속
    = 낙상 후보 또는 확정
```

최종 구조는 다음과 같이 확장 가능하게 만든다.

```text
FallScore =
    w1 * head_drop_score
  + w2 * hip_drop_score
  + w3 * torso_angle_score
  + w4 * bbox_score
```

현재는 임의의 가중치와 임곗값을 최종값으로 확정하지 않는다. 수집 데이터의 ROC와 오탐 사례를 기반으로 결정한다.

### 8.2 학습 기반 모델

규칙 기반 기준선과 데이터 수집이 완료된 뒤 최근 약 2초의 관절 좌표를 입력으로 사용하는 GRU, LSTM 또는 TCN을 학습한다.

예시 입력:

```text
60 frames x 25 joints x (x, y, confidence)
```

원본 픽셀 좌표는 골반 중심과 신체 크기를 기준으로 정규화한다. RTX 2070 SUPER 8GB는 이 규모의 관절 시계열 모델 학습에 충분하다.

평가 지표:

- AUROC
- Recall
- Precision
- F1 score
- 시간당 오경보 횟수
- 낙상 발생부터 알림까지의 지연시간
- 사람 및 카메라 방향별 성능

## 9. Jetson Orin Nano 및 ROS 이식 계획

Windows에서 만든 Python 로직과 모델 가중치는 이식할 수 있지만 Windows용 `OpenPoseDemo.exe`는 Jetson에서 사용할 수 없다. Jetson은 ARM64 Linux이므로 자세 추정 런타임을 Jetson용으로 별도 빌드하거나 TensorRT 기반 모델로 교체해야 한다.

권장 최종 프로세스 구성:

```text
[저수준 모터 제어기/MCU]
          |
          v
[Jetson 보행 및 상태 관리 노드]
          ^
          | ROS 2 Topic
          v
[Jetson 낙상 감지 노드]
  - Depth Camera 입력
  - Pose Estimation
  - Fall Detection
  - Event Publisher
```

OpenPose 추론은 처리시간 변동이 있으므로 모터의 저수준 실시간 제어 주기를 직접 담당하면 안 된다. 비전 프로세스가 느려지거나 종료돼도 모터 제어가 영향을 받지 않도록 프로세스와 책임을 분리한다.

낙상 감지 노드가 발행할 최소 메시지 예시:

```yaml
timestamp: 0.0
person_id: 0
state: NORMAL       # NORMAL / FALLING / FALLEN / UNKNOWN
fall_detected: false
confidence: 0.0
```

Jetson 최적화 방향:

- Jetson Orin Nano 8GB 이상 권장
- BODY_25 신체 관절만 사용
- 얼굴/손 검출 비활성화
- 초기에는 최대 인원 1명
- 입력 및 네트워크 해상도를 낮게 시작
- 전력과 발열, GPU/메모리 사용량 측정
- 가능하면 TensorRT FP16/INT8 최적화 검토
- 보행, LLM, 비전 동시 구동 시 자원 경쟁 측정

OpenPose 1.7.0은 오래된 Caffe 기반 코드이므로 최신 JetPack/CUDA/cuDNN에서 빌드 문제가 생길 수 있다. FallDetector가 OpenPose 고유 코드에 종속되지 않도록 공통 keypoint 인터페이스를 유지한다.

## 10. Git 및 파일 관리 주의사항

현재 `git status` 기준으로 다음 폴더가 추적되지 않는다.

```text
openpose/openpose-portable/
tmp/
```

`openpose/openpose-portable/`에는 대용량 바이너리와 모델이 들어 있으므로 일반 Git 커밋에 포함하지 않는 것이 좋다. 설치 방법과 모델 해시만 문서화하고 실제 바이너리는 각 개발 PC와 Jetson에서 준비한다.

`tmp/`는 PDF 검토용 렌더링 등 임시 산출물이므로 커밋하지 않는다.

원본 영상은 용량이 커질 수 있으므로 다음 원칙을 사용한다.

- 매우 짧은 공개 테스트 영상만 일반 Git으로 관리
- 실제 촬영 영상은 별도 스토리지 또는 데이터셋 저장소 사용
- 반드시 GitHub로 공유해야 하면 Git LFS 검토
- 개인정보가 포함된 영상은 공개 저장소에 업로드하지 않음

## 11. 사용자가 해야 할 일

현재 테스트 영상 2개와 OpenPose 처리 결과가 준비되었으므로, 당장 영상을 수백 개 촬영할 필요는 없다. 먼저 두 영상으로 규칙 기반 낙상 감지기를 구현하고 실제로 구분 가능한지 확인한다.

### 지금 해야 할 일

1. 현재 원본 영상 2개를 삭제하거나 편집하지 않고 보관한다.

```text
data/raw_videos/p01_normal_walk_sit_001.mp4
data/raw_videos/p01_fall_side_001.mp4
```

2. 최종 하드웨어 정보를 확인한다.

- Depth Camera의 정확한 제품명
- Jetson Orin Nano의 메모리 용량(4GB/8GB)
- 카메라를 로봇에 장착할 예상 높이
- 카메라의 예상 하향 각도
- 보행 코드가 ROS 1 또는 ROS 2 중 무엇을 사용하는지
- JetPack 또는 Ubuntu 버전이 이미 정해졌는지

3. 이전 낙상 감지 구현 자료가 남아 있는지 팀원에게 확인한다.

- AUROC 87%를 계산한 코드 또는 노트북
- 기존 OpenPose/Depth Camera 코드
- 학습 및 테스트 영상
- keypoint JSON/CSV/NPZ
- 가중치와 임곗값
- ROC 곡선과 평가 결과
- Jetson 및 ROS 실행 스크립트

자료가 없다면 현재 코드로 새로 구현한다.

4. 다음 규칙 기반 프로토타입이 완성될 때까지 추가 촬영은 잠시 보류한다. 프로토타입에서 오탐과 미탐 원인을 확인한 뒤 필요한 동작을 중심으로 촬영해야 데이터 낭비를 줄일 수 있다.

### 규칙 기반 프로토타입 이후 해야 할 일

첫 프로토타입이 정상적으로 작동하면 총 30~60개의 소규모 검증 데이터를 준비한다.

```text
정상 영상: 20~40개
낙상 영상: 10~20개
촬영 인원: 3~5명
영상 길이: 약 5~15초
```

정상 동작에는 서기, 걷기뿐 아니라 낙상과 혼동하기 쉬운 다음 동작을 포함한다.

- 빠르게 의자에 앉기
- 물건 줍기
- 허리 숙이기
- 무릎 꿇기
- 바닥이나 침대에 자발적으로 눕기
- 누웠다가 다시 일어나기
- 카메라 밖으로 이동하기

낙상 동작에는 다음 유형을 포함한다.

- 앞으로 넘어짐
- 뒤로 넘어짐
- 좌우로 넘어짐
- 걷다가 넘어짐
- 천천히 주저앉음
- 넘어진 뒤 움직이지 않음
- 넘어진 뒤 다시 일어남

낙상 영상은 안전 매트, 보호 장비, 보조자가 준비된 상태에서 연출된 동작으로만 촬영한다. 고령자나 낙상 위험이 있는 사람에게 실제 낙상 동작을 요청하지 않는다.

### 학습 모델 단계에서 해야 할 일

규칙 기반 결과를 확인한 뒤 GRU/LSTM/TCN 학습용으로 총 300~500개를 목표로 확장한다.

```text
정상 영상: 200~300개
낙상 영상: 100~200개
촬영 인원: 최소 10명
```

동일한 사람의 영상이 학습과 테스트에 동시에 포함되지 않도록 사람 ID를 정확히 기록한다. 새 영상을 추가할 때 다음 파일명 규칙을 사용한다.

```text
p<사람번호>_<normal|fall>_<동작>_<순번>.mp4

예시:
p02_normal_pickup_001.mp4
p02_normal_fast_sit_001.mp4
p03_fall_forward_001.mp4
p03_fall_backward_001.mp4
```

영상은 편집하지 않은 원본으로 `data/raw_videos/`에 넣는다. 코드에서 파생 영상과 keypoint를 별도로 생성하므로 원본을 덮어쓰지 않는다.

### 사용자가 하지 않아도 되는 일

다음 작업은 코드와 자동화 파이프라인에서 처리한다.

- 영상 방향과 코덱 변환
- OpenPose 일괄 실행
- 골격 영상과 JSON 생성
- JSON을 NPZ로 변환
- 좌표 정규화와 특징 계산
- 규칙 기반 낙상 감지 구현
- 학습 및 평가 코드 구현
- 데이터 증가에 따른 성능 변화 측정
- 실시간 카메라 감지 코드 구현
- Jetson/ROS 통합 인터페이스 구현

현재 사용자가 바로 준비할 것은 **최종 카메라·Jetson·ROS 정보와 기존 낙상 코드의 존재 여부**다. 추가 영상은 규칙 기반 프로토타입을 확인한 뒤 촬영한다.

## 12. 향후 작업 체크리스트

### 최소 파이프라인 - 완료

- [x] 낙상/정상 테스트 MP4 한 개 이상 준비
- [x] MP4에서 골격 결과 MP4와 BODY_25 JSON 생성
- [x] OpenPose JSON 파서 구현
- [x] `(frames, 25, 3)` NPZ 저장 구현
- [x] confidence 및 누락 관절 처리 구현
- [x] 주요 관절 및 골격 렌더링 시각 검증

### 규칙 기반 감지

- [ ] 좌표 정규화
- [ ] 머리/골반 하강 속도 계산
- [ ] 몸통 각도 계산
- [ ] 바운딩 박스 비율 계산
- [ ] 상태 머신 구현
- [ ] 결과 영상에 상태와 특징값 표시
- [ ] 오탐/미탐 기록

### 데이터와 학습

- [ ] 다양한 사람의 정상/낙상 영상 수집
- [ ] 프레임 구간 라벨링
- [ ] 사람 단위 train/validation/test 분리
- [ ] 규칙 기반 기준선 평가
- [ ] GRU/LSTM/TCN 학습
- [ ] AUROC 및 실사용 지표 평가

### 최종 배포

- [ ] Depth Camera 모델과 Linux SDK 확정
- [ ] JetPack 버전 확정
- [ ] Jetson pose estimator 벤치마크
- [ ] ROS 메시지 인터페이스 확정
- [ ] 비전/보행/LLM 동시 부하 테스트
- [ ] 로봇 정지 상태 통합 테스트
- [ ] 이동 중 감지 또는 ego-motion 보정 검토
- [ ] 실제 돌봄 시나리오 기반 안전 검증

## 13. 다음 작업의 완료 기준

다음 개발 단계는 아래 조건이 모두 충족되면 완료로 본다.

1. 상대경로 또는 CLI 인자로 MP4 입력을 지정할 수 있다.
2. OpenPose가 모든 프레임에서 BODY_25 추론을 수행한다.
3. 골격이 표시된 결과 MP4가 생성된다.
4. 프레임별 JSON이 생성된다.
5. JSON을 `(frame_count, 25, 3)` NumPy 배열로 변환할 수 있다.
6. 사람이 없거나 confidence가 낮은 프레임도 시간축을 깨뜨리지 않고 처리한다.
7. Windows의 절대 사용자 경로를 코드에 하드코딩하지 않는다.
8. 실행 명령과 출력 위치가 문서화되어 다른 PC에서도 재현 가능하다.

## 14. 2인·20개 영상 수집 및 OpenPose 일괄 처리

작업일: 2026-08-12

초기 규칙 기반 낙상 감지기를 검증하기 위해 `p01`, `p02` 두 사람이 동일한 장소에서 각각 10개씩, 총 20개 영상을 촬영했다. 구성은 사람마다 낙상 5개와 정상 5개다.

```text
낙상: backward, collapse, forward, left, right
정상: bend, lie_down, pickup, sit, walk
```

원본은 `data/raw_videos/`에 저장했으며 개인정보와 대용량 영상이 공개 저장소에 포함되지 않도록 Git 추적에서 제외했다.

### 원본 영상 검사 결과

- 영상 수: 20개 (`p01` 10개, `p02` 10개)
- 클래스 수: 낙상 10개, 정상 10개
- 해상도: 전부 1280×720
- 실제 FPS: 약 9.95~10.03 FPS
- 영상 길이: 약 8.53~14.67초
- 총 프레임 수: 2,169프레임
- 전체 프레임 디코딩: 20개 모두 성공
- 대표 프레임 육안 검사: 동작 전·수행 중·종료 자세와 전신 구도 확인

촬영 목표는 30FPS였지만 실제 저장값은 약 10FPS였다. 현재 규칙 기반 프로토타입에는 사용할 수 있으나, 빠른 하강 속도를 더 정밀하게 측정하려면 향후 촬영 프로그램과 카메라의 실측 FPS를 개선해야 한다.

배경의 책상, 의자와 화면 오른쪽 기둥이 일부 자세 추정을 방해할 가능성이 있다. 이후 데이터 수집에서는 가능한 한 단순한 배경을 사용하고 사람이 화면 중앙에서 바닥에 누울 공간을 충분히 확보한다.

### OpenPose 일괄 처리 결과

`fall_detection/process_video.py`를 사용하여 20개를 BODY_25, `--net_resolution -1x256` 설정으로 순차 처리했다.

- GPU: NVIDIA GeForce RTX 2070 SUPER
- 전체 일괄 처리시간: 약 162초
- 영상당 처리시간: 약 5.65~8.72초
- 생성 결과: 영상별 `rendered.mp4`, `keypoints_json/`, `keypoints.npz`, `metadata.json`
- NPZ 형상: 모든 영상에서 `(원본 프레임 수, 25, 3)`
- 원본·렌더링·JSON·NPZ 프레임 수: 20개 모두 일치
- 렌더링 영상 전체 프레임 디코딩: 모두 성공
- 원본과 렌더링 FPS: 모두 일치

18개 영상의 사람 검출 프레임 비율은 97.3~100%였다. 다음 두 정면 낙상 영상은 완전히 엎드린 뒤 BODY_25가 사람을 놓쳐 검출률이 낮았다.

```text
p01_fall_forward_001.mp4: 64/112프레임, 57.1%
p02_fall_forward_001.mp4: 73/106프레임, 68.9%
```

두 영상 모두 서 있는 구간과 앞으로 넘어지는 중간 과정에서는 골격이 검출됐다. 누운 뒤의 신체 중첩과 정면 엎드린 자세에서 검출이 중단된 것으로, 파일 손상이나 처리 파이프라인 오류는 아니다. 낙상 판정기는 단일 프레임의 최종 자세에만 의존하지 않고, 검출이 유지되는 낙상 전후의 하강 속도·몸통 각도 변화와 검출 소실 자체를 함께 다뤄야 한다.

### 이 결과에 따른 다음 작업

1. 20개 NPZ에서 골반·어깨·머리 중심과 몸통 각도 특징을 계산한다.
2. 낮은 FPS에서도 사용할 수 있도록 프레임 차이가 아닌 초 단위 속도로 정규화한다.
3. 긴 관절 누락 구간은 정상 좌표 0으로 처리하지 않고 `UNKNOWN` 또는 검출 소실 상태로 유지한다.
4. 규칙 기반 상태 머신을 구현해 20개 영상의 정상/낙상 분류 결과를 확인한다.
5. 데이터가 늘어나면 사람 단위로 학습·검증·테스트를 분리한다. 현재는 사람이 2명뿐이므로 학습 모델의 일반화 성능을 평가하기에는 부족하다.

## 15. 첫 번째 규칙 기반 낙상 감지기 구현

작업일: 2026-08-12

20개 영상의 BODY_25 데이터를 이용해 설명 가능한 규칙 기반 기준선을 구현했다. 구현 파일은 다음과 같다.

```text
fall_detection/
├── features.py   # 자세 특징 계산과 짧은 누락 보간
├── detector.py   # 낙상 상태 머신
└── evaluate.py   # 데이터셋 평가와 결과 영상 생성
```

### 특징 설계

카메라 해상도, 사람과 카메라 사이 거리 및 FPS 변화에 덜 민감하도록 다음 원칙을 적용했다.

- 시작 약 2초에서 검출한 사람의 신체 높이를 정규화 기준으로 사용
- 프레임당 픽셀 이동이 아니라 `신체 높이/초` 단위의 하강 속도 사용
- `MidHip`이 없으면 좌우 Hip의 신뢰도 가중 중심 사용
- Nose/Neck이 없으면 Shoulder 중심을 머리 위치의 대체값으로 사용
- 신뢰도 0.2 미만 관절은 무효 처리
- 약 0.3초 이하의 짧은 누락만 선형 보간
- 긴 누락은 NaN과 검출 소실 상태로 보존

계산 특징:

1. 머리 하강 속도와 약 1초 하강량
2. 골반 하강 속도와 약 1초 하강량
3. 어깨 중심-골반 중심 몸통 벡터가 수직선과 이루는 각도
4. 주요 신체 관절 바운딩 박스의 폭/높이 비율
5. 초기 골반 위치 대비 현재 골반의 낮아진 정도

### 상태 머신

```text
NORMAL -> FALLING -> FALL_CANDIDATE -> FALLEN
   └-------------------------------> UNKNOWN
```

- 빠른 하강과 충분한 하강량이 동시에 나타나면 `FALLING`
- 최근 하강 증거가 있고 몸통이 수평에 가까우며 골반이 낮으면 `FALL_CANDIDATE`
- 후보 자세가 약 0.5초 유지되면 `FALLEN`
- 낙상 움직임 직후 골격이 약 0.5초 이상 사라져도 `FALLEN`
- 사전 하강 증거 없이 사람만 사라지면 `UNKNOWN`이며 낙상으로 판정하지 않음

초기 주요 임곗값:

```text
하강 속도: 0.55 신체 높이/초 이상
1초 하강량: 신체 높이의 0.30 이상
몸통 각도: 수직선 기준 48도 이상
바운딩 박스 폭/높이: 0.85 이상
골반 하강 위치: 신체 높이의 0.28 이상
```

### 개발 세트 평가

첫 하강 속도 기준 0.65에서는 다음 결과가 나왔다.

```text
TP=9, TN=10, FP=0, FN=1
Accuracy=95.0%, Precision=100%, Recall=90.0%, F1=94.7%
미탐: p01_fall_collapse_001
```

`p01_fall_collapse`의 최대 하강 속도는 약 0.567 신체 높이/초였다. 정상 동작과 비교하면서 속도 기준을 0.55로 낮췄고, 이후 낮은 골반 위치와 수평 자세의 지속 조건은 그대로 유지했다. 최종 개발 세트 결과는 다음과 같다.

```text
TP=10, TN=10, FP=0, FN=0
Accuracy=100%, Precision=100%, Recall=100%, F1=100%
```

20개 판정 영상의 프레임 수와 FPS를 원본 OpenPose 결과와 대조했으며 모두 일치했다. `collapse`와 `normal_lie_down`의 대표 프레임도 비교했다. 주저앉기 영상은 하강 후 `FALLEN`이 되었고, 천천히 눕는 정상 영상은 전 구간 `NORMAL`을 유지했다.

이 100%는 실제 정확도가 아니다. 같은 20개 영상으로 임곗값을 선택하고 다시 같은 영상에서 평가한 개발 세트 적합 결과다. 독립된 사람과 장소의 영상이 없으므로 일반화 성능, 신뢰구간, 실제 오경보율을 아직 말할 수 없다.

### 다음 검증 조건

다음에 촬영하는 영상은 현재 임곗값을 고정한 상태에서 별도 테스트 세트로 사용한다. 최소한 새로운 사람, 다른 복장, 다른 조명과 다음 어려운 정상 동작을 포함한다.

- 빠르게 의자에 앉기
- 쪼그려 앉기와 무릎 꿇기
- 물건을 급하게 줍기
- 바닥에 빠르게 앉되 눕지 않기
- 화면 밖으로 빠르게 이동하기
- 가구 뒤에서 관절이 가려지기

독립 테스트에서 발생한 실패는 영상별로 기록하고, 테스트 결과를 본 뒤 임곗값을 다시 조정한다면 해당 데이터는 더 이상 최종 테스트 세트로 사용하지 않는다.

## 16. Windows 실시간 카메라 판정 프로토타입

작업일: 2026-08-12

녹화 영상에서 검증한 규칙 엔진을 카메라 입력에 연결하기 위해 `fall_detection/live_detect.py`를 구현했다.

처리 흐름:

```text
Windows 카메라
  -> OpenPoseDemo.exe 실시간 BODY_25 처리
  -> 프레임별 JSON 생성
  -> 처리 FPS 자동 추정
  -> features.py 특징 계산
  -> detector.py 상태 머신 판정
  -> OpenPose 골격 창 + Fall Detection Status 창
```

카메라 0번에서 640×480 입력을 확인했고, OpenPose가 실시간 JSON을 지속적으로 생성하는 것까지 검증했다. 처음 백그라운드 프로세스로 실행했을 때 영상 처리는 됐지만 GUI가 사용자 화면에 나타나지 않았다. 이후 보이는 PowerShell 세션에서 실행하도록 변경해 OpenPose와 판정 상태 창을 표시했다.

다른 Windows 컴퓨터에서 절대경로 없이 실행할 수 있도록 다음 파일도 추가했다.

```text
fall_detection/requirements.txt
fall_detection/run_live.bat
```

실행 배치 파일은 `%~dp0`을 사용해 배치 파일 위치를 기준으로 프로젝트 루트를 찾는다. Python 가상환경, OpenPose 포터블 바이너리, BODY_25 모델과 원본 영상은 Git에 포함하지 않는다.

현재 실시간 판정은 초기 프로토타입이며 다음 한계가 있다.

- 2명·20개 개발 영상에서 조정한 임곗값 사용
- 카메라와 사람이 모두 정지한 환경 가정
- 시작 약 2초 동안 서 있는 기준 자세 필요
- OpenPose 처리 FPS와 검출 품질에 따라 지연 발생 가능
- 장시간 실행 시 전체 프레임을 다시 계산하는 현재 구조의 최적화 필요
- Windows OpenPose 실행 방식은 Jetson에 직접 이식할 수 없음

따라서 현 단계의 정확한 표현은 학습 모델 완성이 아니라 **초기 데이터로 규칙 기반 낙상 기준선을 구현하고 개발 세트에 맞춰 임곗값을 조정한 상태**다.
