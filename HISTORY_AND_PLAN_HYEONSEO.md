# 현서 담당 낙상 감지 개발 이력 및 계획

최종 갱신일: 2026-08-22

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

## 17. Linux 작업 재개 준비

작업일: 2026-08-12

Windows 포터블 실행 경로에 고정되어 있던 영상 및 실시간 파이프라인을 Linux에서도 사용할 수 있도록 실행 계층을 분리했다. `--openpose-root`를 생략하면 저장소의 Linux 빌드 `openpose/build/examples/openpose/openpose.bin`을 먼저 찾고, 없으면 Windows 포터블 `OpenPoseDemo.exe`를 찾는다. OpenPose 소스 루트 또는 빌드 디렉터리를 직접 지정할 수도 있다.

경로 탐색과 핵심 keypoint 처리를 실제 OpenPose 없이 검증하는 `unittest`도 추가했다. 현재 Linux 환경에는 OpenPose 빌드와 BODY_25 모델이 없고 WSL GPU 접근도 확인되지 않았으므로, 다음 단계는 GPU·카메라가 노출되는 환경을 확정하고 Linux OpenPose를 빌드하여 녹화 영상 1개를 종단 간 처리하는 것이다.

OpenPose 빌드와 별개로 Linux에서 즉시 검증 가능한 YOLO11n pose 백엔드도 추가했다. COCO 17관절을 기존 BODY_25 기반 특징 입력으로 변환하며 녹화 영상과 카메라 번호를 동일한 실행기로 처리한다. WSL CPU에서 저장소 샘플 영상 120프레임을 실제 추론한 결과 약 20.2 FPS였고, 첫 3프레임에서 1~2명의 자세와 최대 17개 관절 검출을 확인했다. 결과 영상도 120프레임으로 정상 생성됐다.

동일 환경에서 카메라 0번 열기를 시도했지만 `/dev/video0`와 WSL GPU 장치 `/dev/dxg`가 존재하지 않아 장치를 열 수 없었다. 따라서 모델과 웹캠 연결 코드는 준비됐지만 실제 카메라 구동 완료를 위해서는 Windows 웹캠을 WSL USB 장치로 전달하거나 실물 Linux 컴퓨터에서 실행해야 한다.

이후 Windows에 `usbipd-win 5.3.0`을 설치하고 Logitech HD Pro Webcam C920(BUSID `2-7`)만 공유해 WSL에 연결했다. WSL 커널이 UVC 장치와 `/dev/video0`, `/dev/video1`을 인식했고 V4L2에서 640×480 스트리밍 기능을 확인했다. 기본 YUYV는 timeout이 발생해 MJPEG 640×480 15 FPS를 요청하도록 실행기를 수정했다. 실제 웹캠 10프레임을 YOLO pose 파이프라인으로 처리하고 640×480 결과 MP4를 생성·재디코딩하는 데 성공했다. 다만 USB/IP 전송에서 실제 처리 속도는 약 0.4 FPS였고 손상 JPEG 경고도 발생했다. 모델 통합 검증에는 성공했지만 실시간 운용 성능을 위해서는 네이티브 Linux/Jetson 또는 더 안정적인 카메라 전달 방식이 필요하다.

## 18. Windows YOLO pose GPU 실시간 검증

작업일: 2026-08-12

Linux `.venv`의 실험용 `torch 2.13.0+cu130`을 공식 CUDA 12.4 조합인 `torch 2.5.1+cu124`, `torchvision 0.20.1+cu124`로 교체했다. 현재 Codex WSL 세션은 운영체제 정책으로 `/dev/dxg`와 `/dev/video*`가 노출되지 않아 WSL 안에서는 CUDA를 사용할 수 없지만, Windows 네이티브 환경은 다음과 같이 정상임을 확인했다.

```text
GPU: NVIDIA GeForce RTX 2070 SUPER 8GB
Driver: 560.94
Windows Conda: fall-detection
PyTorch: 2.5.0+cu124
torch.cuda.is_available(): True
```

Windows Conda 환경에 `ultralytics 8.4.118`을 설치하고 저장소 샘플 영상 60프레임을 GPU 0에서 처리했다. 이어서 Windows 카메라 0번을 640×480으로 열어 GPU 포즈 추론, 규칙 기반 상태 머신, 주석 MP4 저장까지 60프레임 종단 간 검증했다. 생성 영상은 MPEG-4 640×480, 60프레임이며 화면에 표시된 처리 FPS는 약 15.7이었다. 당시 카메라 화면에 전신 사람이 없었으므로 상태가 `UNKNOWN`인 것은 정상이다.

`yolo_pose.py`에는 CUDA 자동 선택, 잘못된 GPU 요청의 조기 진단, 녹화 영상을 가상 웹캠처럼 반복하는 `--loop --realtime` 옵션을 추가했다. `run_yolo_gpu.bat`은 Windows에서 기본 카메라 0과 GPU 0으로 즉시 실행한다. 단위 테스트는 정적 자세, 관절 대응, 장치 인자, 합성 낙상 시퀀스의 `FALLEN` 전이를 포함해 8개가 통과했다.

## 19. 공개 데이터셋 학습 기준선

작업일: 2026-08-21

2명·20개 자체 개발 영상에 맞춘 규칙 기반 결과만으로 일반화 성능을 주장할 수 없어서
공개 데이터셋을 이용한 별도 학습·평가 경로를 구현했다.

### GMDCSA-24

Zenodo의 GMDCSA-24 v2.0, 총 160개 영상을 사용했다. 영상 복제 없이 상대 심볼릭 링크와
매니페스트를 생성하고, YOLO11n-pose 결과를 BODY_25 호환 `(frame, 25, 3)` NPZ로
저장했다. 같은 사람이 train과 test에 섞이지 않도록 다음과 같이 고정했다.

```text
train: Subject 1, 2 — 80개
validation: Subject 3 — 43개
test: Subject 4 — 37개
```

`prepare_gmdcsa24.py`, `extract_gmdcsa24_poses.py`, `train_gmdcsa24.py`를 추가했다. 학습
모델은 96프레임으로 리샘플링한 정규화 관절 시계열을 입력으로 받는 2-layer
bidirectional GRU다. validation F1으로 최적 epoch를 선택하고 early stopping을 적용했다.

### FallVision 보조 데이터

Harvard Dataverse의 20개 keypoint RAR를 검증·압축 해제하고 COCO 관절 CSV를 BODY_25
호환 NPZ로 변환했다. 총 5,864개 시퀀스이며 낙상 3,000개, 정상 2,864개다. GMDCSA-24의
validation/test 독립성을 보존하기 위해 전부 train에만 추가했다. 데이터 다운로드와
생성물은 `data/datasets/` 아래에 두고 Git에서 제외한다.

UR Fall RGB 70개 시퀀스의 공식 다운로드 도구도 준비했지만, 이번 보고 성능에는 UR Fall을
사용하지 않았다.

### 고정 test 결과

원시 pose 특징 모델과 13개 engineered 특징 모델을 학습했다. Subject 3 validation에서만
앙상블 가중치와 임곗값을 선택하고 Subject 4 test를 한 번 평가했다.

```text
앙상블: pose 0.3 + engineered 0.7
threshold: 0.35

Subject 4 test (37개)
TP=16, TN=11, FP=9, FN=1
Accuracy=72.97%
Precision=64.00%
Recall=94.12%
F1=76.19%
```

현재 모델은 낙상을 거의 놓치지 않는 대신 정상 행동 오경보가 많은 방향이다. 이 수치는
새 사람 1명의 37개 영상에 대한 첫 독립 결과로는 의미가 있지만 표본이 작다. 또한 영상
전체 정상/낙상 분류 결과이며 실시간 낙상 시작 프레임 정확도가 아니다. 최종 연구 결과에는
4-fold leave-one-subject-out 평균·표준편차와 실제 카메라 연속 스트림의 시간당 오경보를
추가해야 한다.

## 20. Jetson Orin Nano 배포 사전작업

작업일: 2026-08-21

실제 Jetson 장치가 연결되기 전에 PC에서 가능한 배포 준비를 완료했다. 초기 온디바이스
경로는 다음과 같이 결정했다.

```text
V4L2 카메라
  -> YOLO11n-pose TensorRT FP16 (416x416)
  -> BODY_25 호환 관절
  -> 규칙 기반 실시간 상태 머신
  -> 이후 ROS 2 publisher 연결
```

GRU는 현재 영상 전체 분류기이므로 실시간 경보 경로에 억지로 연결하지 않는다. Jetson에서
실시간 사용할 모델은 우선 프레임 단위 YOLO pose와 기존 상태 머신이다.

추가한 배포 기능:

- `jetson_preflight.py`: ARM64/L4T, CUDA PyTorch, OpenCV, Ultralytics, 카메라, 모델 점검
- `run_jetson.sh`: TensorRT engine, GPU 0, 416 입력, 640×480 15 FPS 기본 실행
- `--no-render`: 헤드리스 운용에서 골격 렌더링 비용 제거
- 워밍업 시간과 steady-state inference median/p95 분리 출력
- V4L2 MJPEG 설정, 카메라 버퍼 최소화 및 선택적 FFmpeg 입력
- JetPack의 PyTorch/OpenCV를 pip wheel로 덮어쓰지 않는 설치 절차

TensorRT `.engine`은 JetPack, TensorRT 및 GPU에 종속되므로 PC에서 생성해 커밋하지 않는다.
실행할 Orin Nano 본체에서 `yolo export format=engine imgsz=416 half=True device=0`으로
생성한다.

PC에서 사전점검 JSON 출력, shell 문법, 전체 Python compile, 단위 테스트 10개와
YOLO11n-pose 헤드리스 비렌더링 영상 추론을 확인했다. CPU 3프레임 smoke test에서는 첫
워밍업 약 940 ms와 이후 추론 중앙값 약 29 ms가 분리되어 출력됐다. 이 수치는 Jetson
성능 수치가 아니며 코드 경로 검증용이다.

### Jetson 본체에서 남은 작업

1. JetPack 버전과 Orin Nano 메모리 용량 확인
2. NVIDIA aarch64 PyTorch 및 카메라 드라이버 설치
3. 본체에서 TensorRT FP16 엔진 생성
4. `tegrastats`와 함께 최소 10분 연속 실행
5. FPS, median/p95, RAM, GPU 사용률, 온도 및 throttling 기록
6. 로봇 정지·진동·보행 조건별 오탐과 미탐 측정
7. ROS 2 topic 이름과 메시지 형식을 확정한 뒤 publisher 구현

첫 벤치마크 목표는 입력 포함 10 FPS 이상과 p95 약 100 ms지만, 이는 실제 측정 전의
출발점일 뿐 합격을 주장하는 수치가 아니다.

## 21. Jetson Orin Nano 실측 및 실시간 경로 재구현

작업일: 2026-08-22

실제 Jetson Orin Nano Super 8GB(L4T R36.4.7, JetPack 6.2 계열)에서 코드를 직접 실행하며
배포 준비 상태를 확인했다. 20절까지의 준비는 PC에서 작성한 것이고, 이번에는 본체에서
측정한 값으로 대체한다.

### 확인된 환경

```text
보드      Jetson Orin Nano Engineering Reference Developer Kit Super (8GB)
L4T       R36.4.7, Ubuntu 22.04, 전력 모드 MAXN_SUPER, 6코어, NVMe 467GB
PyTorch   2.5.0a0+nv24.08 / CUDA 12.6 / cuda_available=True / Orin sm_87
OpenCV    4.11.0   NumPy 1.26.4   ROS 2 Humble + rclpy 정상
미설치    Ultralytics(설치 완료), TensorRT, 카메라
```

### 발견한 실행 차단 요인

`PATH` 앞쪽의 `~/jetconda3/bin/python3`가 **Python 3.6.1**이라, `python3 fall_detection/...`
형태의 명령이 전부 문법 오류로 죽는다. CUDA PyTorch를 가진 인터프리터는 JetPack의
`/usr/bin/python3.10` 하나뿐이다. `run_jetson.sh`가 PATH를 신뢰하지 않고 `torch.cuda`,
OpenCV, Ultralytics가 모두 import되는 인터프리터를 직접 찾도록 고쳤다.

또한 `requirements.txt`를 Jetson에서 그대로 설치하면 pip가 JetPack PyTorch를 CUDA 없는
일반 aarch64 wheel로 교체한다. Ultralytics만 `--no-deps`로 설치하는 절차로 문서를 바꿨다.

### 규칙 계층이 실제 병목이었음

Orin Nano에서 프레임당 비용을 측정했다.

```text
윈도우   extract_features   detect_fall   합계
   60         32.6 ms         1.2 ms     33.8 ms
  120         62.6 ms         2.3 ms     64.8 ms
  225        115.6 ms         4.2 ms    119.8 ms
  450        230.0 ms         8.4 ms    238.3 ms
```

GPU 포즈 추론은 35 ms인데 규칙 계층이 매 프레임 전체 윈도우를 다시 계산하느라 그보다 큰
비용을 쓰고 있었다. 30 FPS 카메라 기본값인 450프레임 윈도우에서는 프레임당 238 ms로,
TensorRT를 붙여도 약 4 FPS가 천장이었다.

### `streaming.py` 추가

같은 특징, 같은 `DetectorConfig` 임곗값, 같은 상태 전이를 프레임당 O(1)로 갱신하는
`StreamingFallDetector`를 추가했다. 저장하는 이력은 하강량 계산에 실제로 필요한 약 1초뿐이다.

```text
윈도우   기존         스트리밍     배수
   60    37.1 ms      0.924 ms     40x
  120    65.9 ms      0.907 ms     73x
  225   121.9 ms      0.901 ms    135x
  450   242.2 ms      0.911 ms    266x
```

30분(27,000프레임) 연속 처리에서 프레임당 0.917 → 0.903 ms로 평탄했고 RSS는 28 MB로
변하지 않았다. 저장소 샘플 영상 end-to-end는 10.3 FPS에서 18.6 FPS로 올랐다.

특징 수식이 두 벌로 갈라지지 않도록 `features.py`에서 프레임 단위 계산을 `frame_*`
함수로 분리하고 오프라인 경로도 같은 함수를 쓰게 했다. 무작위 12회 시험에서 리팩터 전후
출력이 모든 필드에서 완전히 동일함을 확인했다.

### 실시간 판정의 두 가지 오류 수정

첫째, 기준 신체 높이와 골반 위치가 슬라이딩 윈도우에서 다시 측정되고 있었다. 낙상 몇 초
뒤에는 쓰러진 자세가 "서 있는" 기준이 되어 버린다. 이제 세션 시작의 보정 구간에서 한 번만
정하고 고정한다.

둘째, 더 심각한 문제로 **경보가 스스로 해제됐다.** 15 FPS·40초 시나리오(5초에 낙상 후
계속 정지)를 재현한 결과는 다음과 같다.

```text
기존   6.2초 FALLEN  ->  19.8초에 NORMAL로 복귀 (사람은 계속 바닥에 있음)
수정   6.3초 FALLEN  ->  40초까지 FALLEN 유지 (latched)
```

낙상 장면이 15초 윈도우 밖으로 밀려나면 근거가 사라져 상태가 되돌아가고 있었다. 이제
`FALLEN`은 세션 동안 유지되며 `reset_alarm()`으로만 해제한다. 사람이 스스로 일어난 경우를
위해 `--auto-clear-seconds`를 두었고 기본값은 해제하지 않음이다. 이 시나리오를 회귀
테스트로 추가했다.

### 해상도는 성능과 무관함

순수 추론시간을 40회씩 측정한 결과 imgsz 256/320/416/640이 모두 30~32 ms였다. 이 모델은
연산량이 아니라 커널 실행 오버헤드에 묶여 있다. 따라서 20절에서 정한 416 입력은 이득이
없어 기본값을 640으로 되돌렸고, 속도를 올리는 유일한 수단은 TensorRT 엔진임을 확인했다.

### 그 외 수정

- `evaluate.py`의 `round()` 인자 결합 오류를 고쳤다. 삼항 연산자가 값이 아니라 `ndigits`에
  붙어 있어서, 사람이 한 번도 검출되지 않은 영상이 있으면 배치 평가 전체가 `TypeError`로
  죽었다.
- `live_detect.py`가 부분 기록된 JSON을 건너뛰고 뒤 프레임을 먼저 소비해 시간축이 섞이던
  문제를 고쳤다. 이제 미완성 파일을 만나면 멈춘다. 무한히 커지던 소비 목록도 제거했다.
- `jetson_preflight.py`를 blocker와 warning으로 나누고, 인터프리터 불일치, TensorRT 부재,
  가용 메모리 부족, rclpy 미로딩을 함께 보고하도록 다시 썼다.
- 단위 테스트 10개 → 19개.

### 메모리 제약

Orin Nano는 CPU와 GPU가 8GB를 공유한다. 데스크톱 세션이 떠 있으면 가용 메모리가 2.7GB로
떨어지고, 그 상태에서 CUDA 초기화가 `CUBLAS_STATUS_ALLOC_FAILED`로 실패했다. 실제 운용은
헤드리스로 해야 한다. 보행 정책과 EXAONE LLM을 동시에 올리는 구성은 이 8GB 안에서
별도 측정이 필요하다.

### 학습·구동 역할 분담 확정

Jetson에서 학습까지 하려는 시도는 하지 않기로 했다. 근거는 다음과 같다.

- Orin Nano는 CPU와 GPU가 8GB를 공유하고, 데스크톱 세션만 떠 있어도 가용 메모리가 2.7GB로
  떨어진다. GRU 학습 배치를 올릴 여유가 없다.
- 공개 데이터셋 원본이 약 1.1GB이고 포즈 추출까지 하면 더 커진다. Jetson에 둘 이유가 없다.
- 학습은 재현성이 중요한데, PC의 CUDA 12.4 조합에서 이미 검증한 절차가 있다.

따라서 역할을 다음과 같이 고정한다.

```text
PC (NVIDIA GPU)          데이터 준비, 포즈 추출, GRU 학습, 정확도 검증
Jetson Orin Nano         카메라 실시간 판정만
공유                     features.py, detector.py, streaming.py의 수식과 임곗값
```

이 분담을 `fall_detection/README.md` 최상단의 역할 표와 파일별 실행 위치 표로 옮겼다.
문서도 `## PC: 데이터 준비, 학습, 검증`과 `## Jetson: 실시간 구동`으로 나눴다. 이전에는
Windows → Linux → WSL → Jetson 순의 작업 이력 순서로 쓰여 있어서, 새로 합류한 사람이
자기 장비에서 무엇을 해야 하는지 찾기 어려웠다.

### Jetson 런타임 설치 정리

설치 절차를 `fall_detection/setup_jetson.sh` 하나로 묶었다. 여러 번 실행해도 안전하다.

```bash
fall_detection/setup_jetson.sh          # 사용자 권한 설치만
fall_detection/setup_jetson.sh --apt    # sudo apt 단계까지
```

이 보드에서 실제로 설치한 내용은 다음과 같다.

```text
ultralytics 8.4.126, ultralytics-thop, py-cpuinfo   --no-deps 설치
onnx 1.22.0, onnxslim 0.1.96                        numpy==1.26.4 제약 설치
yolo11n-pose.pt                                     내려받음
```

`--no-deps`와 NumPy 제약이 핵심이다. 제약 없이 `onnx`를 설치하면 pip가 NumPy 2.2.6을
끌어와 JetPack NumPy 1.26.4를 가리고, JetPack OpenCV가 그 ABI로 빌드되어 있어 함께
깨진다. 실제로 dry-run에서 이 동작을 확인하고 제약을 걸었다. 설치 후 NumPy 1.26.4,
OpenCV 4.11.0, `torch.cuda.is_available()=True`, cv2↔numpy 변환을 모두 재확인했다.

`tensorrt`와 `ffmpeg`는 root 권한이 필요해 설치하지 못했다. 스크립트가 명령만 출력한다.

### 아직 못 하는 것

1. **카메라 없음.** `/dev/video*`가 없고 USB에도 카메라가 없다. 실시간 종단 검증, 카메라
   포함 FPS, 시간당 오경보, 거리별 오탐·미탐을 측정할 수 없다. 위 18.6 FPS는 녹화 영상
   기준이다.
2. **TensorRT 미설치.** root 권한이 필요하다(`sudo apt install -y tensorrt ffmpeg`, 또는
   `setup_jetson.sh --apt`). 엔진 빌드에 필요한 `onnx`/`onnxslim`은 설치해 두었으므로
   TensorRT만 깔면 바로 엔진을 만들 수 있다. 엔진 없이도 실행되지만 현재 수치는 PyTorch
   eager 기준이며, 해상도가 성능에 영향이 없는 이상 이것이 유일한 속도 개선 수단이다.
3. **Depth Camera 미확정.** USB에 depth 카메라가 없어 depth 기반 특징은 착수할 수 없다.
4. **공개 데이터셋 부재.** `data/`는 Git에서 제외되어 이 보드에 없다. GMDCSA-24 재평가나
   재학습을 하려면 약 1.1GB를 다시 내려받아야 한다.
5. **Jetson OpenPose는 포기 권장.** OpenPose 1.7.0은 오래된 Caffe 기반이라 CUDA 12.6과
   JetPack 6에서 빌드가 사실상 불가능하다. Jetson 경로는 YOLO pose로 확정하고, OpenPose는
   Windows 녹화 데이터 처리 용도로만 남긴다.
6. **ROS 2 연동 미착수.** rclpy는 정상 동작하므로 차단 요인은 없다. 다만 보행 노드의
   토픽 이름과 메시지 형식이 정해져야 publisher를 확정할 수 있다.
## 22. FallVision 재학습 및 고재현율 앙상블

작업일: 2026-08-22

Jetson 실시간 경로의 규칙 기반 판정만으로 충분한지 확인하기 위해 Harvard Dataverse의
FallVision 공개 관절 데이터 20개 RAR를 실제로 다운로드하고 검증·압축 해제했다. 시스템에
`7z`가 없는 환경도 처리할 수 있도록 다운로드 도구에 `7zz` 탐색과 Python 압축 해제
fallback을 추가했다.

준비된 데이터는 총 5,864개 시퀀스다.

```text
낙상: 3,000개
정상: 2,864개
bed: 1,883개
chair: 1,951개
stand: 2,030개
```

FallVision 공개 매니페스트에는 참가자 ID가 없어서 사람 단위 분할을 만들 수 없었다.
따라서 라벨과 동작 종류별 비율을 유지하는 결정적 clip-level 분할을 별도로 만들었다.

```text
train: 4,104개
validation: 879개
test: 881개
```

빈 시퀀스와 15프레임 미만 시퀀스는 학습·평가에서 제외되어 실제 validation/test 평가
샘플은 각각 875개와 878개다. 이 분할은 같은 참가자가 여러 split에 섞일 가능성이 있으므로
참가자 독립 일반화 성능으로 해석하면 안 된다.

기존 규칙 기반 상태 머신을 FallVision 전체에 30 FPS로 가정해 평가한 결과는 다음과 같다.

```text
TP=1,139, TN=2,802, FP=62, FN=1,861
Accuracy=67.21%
Precision=94.84%
Recall=37.97%
F1=54.23%
```

규칙은 오경보가 적고 정밀도가 높지만 낙상을 많이 놓치는 보수적인 판정기였다. 임곗값을
완화한 후보도 recall 44.27%에 그쳐 기본 임곗값을 공개 데이터 결과만 보고 교체하지 않았다.

대신 다음 두 개의 2-layer bidirectional GRU를 각각 학습했다.

- 원시 pose 모델: 정규화된 25개 관절의 `(x, y, confidence)` 75차원 시계열
- engineered 모델: 골반·머리·몸통·bbox·속도 등 13차원 시계열

단일 모델 test 결과:

```text
engineered GRU
Accuracy=92.03%, Precision=89.68%, Recall=95.30%, F1=92.41%

pose GRU
Accuracy=92.60%, Precision=91.34%, Recall=94.41%, F1=92.85%
```

test 결과를 보지 않고 validation에서 낙상 recall 95% 이상을 만족하면서 precision을
최대화하도록 가중치와 임곗값을 선택했다.

```text
pose weight: 0.45
engineered weight: 0.55
threshold: 0.475

고정 test 878개
TP=428, TN=390, FP=41, FN=19
Accuracy=93.17%
Precision=91.26%
Recall=95.75%
F1=93.45%
```

이 결과는 잘라진 영상 전체의 `normal/fall` 분류 성능이다. 실시간 연속 스트림의 낙상
시작 시점, 감지 지연, 시간당 오경보와는 다른 지표다. 또한 clip-level 분할 한계 때문에
최종 보고 수치로 사용하려면 참가자 ID가 있는 GMDCSA-24의 4-fold 평가 또는 자체 촬영
참가자 독립 테스트가 필요하다.

## 23. 규칙+GRU 실시간 통합 및 배포 번들

작업일: 2026-08-22

영상 전체 분류용 GRU를 그대로 단독 경보기로 사용하지 않고, 최근 4초 관절 윈도에 대한
보조 증거로 실시간 실행기에 연결했다.

```text
V4L2 카메라
  -> YOLO11n-pose TensorRT FP16 416x416
  -> BODY_25 호환 관절
  -> 규칙 기반 상태 머신
  -> pose 0.45 + engineered 0.55 GRU 앙상블
  -> 높은 확률 + 수평 자세 0.5초 지속
  -> FALLEN
```

Jetson CPU 부하를 줄이기 위해 작은 GRU 두 개는 기본적으로 3포즈 프레임마다 실행한다.
GRU 확률만으로 `FALLEN`을 선언하지 않고 몸통 각도 또는 bbox 비율의 수평 자세 증거를 함께
요구한다. 규칙 기반 감지 결과도 유지하므로 두 경로 중 하나가 확정 조건을 충족하면 경보한다.

실시간 운용을 위해 다음도 보완했다.

- 초기 약 2초의 신체 높이와 골반 위치를 고정하여 rolling window가 이동해도 기준이 변하지 않음
- 여러 사람이 보이면 관절 신뢰도와 화면 면적을 함께 사용해 가장 크게 보이는 사람 선택
- `FALLEN` 기본 30초 latch, `--fall-hold-seconds 0`이면 재시작 전까지 유지
- 상태 변화와 heartbeat를 `FALL_STATUS` JSON Lines로 표준 출력
- `outputs/live_fall_status.json`을 임시 파일 교체 방식으로 원자적 갱신
- 규칙 임곗값, GRU 가중치·임곗값·윈도·실행 간격을 CLI로 조정 가능
- 모델과 실행 코드를 묶는 `make_jetson_bundle.sh` 추가

공식 Ultralytics 샘플을 이용한 PC CPU 종단 간 smoke test에서 두 GRU 앙상블까지 실행해
`UNKNOWN -> NORMAL`, 상태 JSON 갱신을 확인했다. 35프레임 결과는 다음과 같다.

```text
average FPS=26.34
YOLO warmup=546.29 ms
steady inference median=19.00 ms
steady inference p95=23.89 ms
final state=NORMAL
```

Windows DirectShow 카메라 0에서도 640x480, 15 FPS로 118프레임을 녹화하고 WSL의 전체
파이프라인에 전달했다. 전체 프레임 처리와 주석 영상·상태 JSON 생성을 확인했으며 YOLO
steady median은 19.04 ms, p95는 23.30 ms였다. 촬영 구도에 얼굴과 상체만 보이고 골반·다리가
보이지 않아 마지막 숙임 동작이 `FALLING`으로 끝났다. 이는 실제 낙상 확정이 아니라 전신이
보이지 않는 구도에서 규칙 기반 초기 기준이 불안정해지는 사례로 기록한다.

최신 Jetson 번들은 다음 로컬 산출물로 생성했다.

```text
outputs/sogang_fall_jetson_bundle.tar.gz
크기: 약 7.2 MB
SHA-256: 850db2cb6a788c943fce2c07afadd42a53d40c989c49185a06a333daea5a7915
```

번들에는 낙상 감지 소스, YOLO11n-pose `.pt`, pose GRU와 engineered GRU가 포함된다.
TensorRT `.engine`은 JetPack·TensorRT·GPU에 종속되므로 Jetson 본체에서 생성한다. 공개
데이터, 가상환경, 모델 가중치와 생성 결과는 Git에서 제외하며 배포 번들로 별도 전달한다.

### 현재 성능 표현과 목표

```text
현재 공개 clip-level recall: 95.75%
현재 공개 clip-level precision: 91.26%
Jetson 실제 연속 스트림 감지율: 아직 미측정
```

1차 현장 목표:

- 참가자 독립 낙상 recall 90% 이상
- precision 85% 이상, F1 88% 이상
- 낙상 후 2초 이내 감지
- 시간당 오경보 1회 이하
- Jetson 입력 포함 10 FPS 이상, p95 약 100 ms 이하

최종 목표는 recall 95% 이상, precision 90% 이상, 시간당 오경보 0.1회 이하, 감지 지연
1.5초 이하, Jetson 15 FPS 지속 실행이다.

### 남은 필수 작업

1. Jetson 본체에서 JetPack·PyTorch·카메라 환경 확정
2. 본체에서 YOLO11n-pose TensorRT FP16 엔진 생성
3. `tegrastats`와 함께 최소 10분 연속 실행 및 온도·throttling 기록
4. 전신이 보이는 실제 설치 구도에서 새 참가자 낙상·정상 혼동 동작 촬영
5. 참가자 독립 recall/precision/F1과 시간당 오경보 측정
6. 로봇 정지·진동·보행 조건별 성능 비교
7. ROS 2 topic과 메시지 형식 확정 후 publisher 연결

## 24. Git 반영 상태

작업일: 2026-08-22

Jetson 실시간 낙상 감지 개선 코드는 다음 로컬 커밋으로 정리했다.

```text
ffdbd30694f4f0cc59c75b2f5c31ca8f36ca6268
feat: improve Jetson real-time fall detection
```

단위 테스트 11개, 전체 Python compile, `run_jetson.sh`와 `make_jetson_bundle.sh` 문법 검사,
두 GRU 앙상블 smoke test를 통과했다. GitHub 원격 `origin/main` 푸시는 현재 WSL 환경에
GitHub HTTPS 인증 정보가 없어 완료되지 않았다. 인증 후 `git push origin main`이 필요하다.

## 25. 두 갈래 작업 병합

작업일: 2026-08-23

21절(Jetson 본체 실측·스트리밍 재구현)과 22~24절(PC에서의 FallVision 재학습·GRU 앙상블)은
같은 `304182e`에서 각각 갈라져 나왔다. 24절이 푸시 미완으로 기록했던 커밋은 이후
`origin/main`에 반영되었고(`ffdbd30`, `b1f0b41`), 이번에 로컬 Jetson 브랜치
`hyeonseo/jetson-realtime`으로 병합했다. 겹친 파일 5개에서 내린 결정은 다음과 같다.

1. **실시간 루프는 스트리밍 경로로 통일.** 원격 쪽은 매 프레임 `extract_features` +
   `detect_fall`을 버퍼 전체에 다시 돌렸는데, 이것이 21절에서 프레임당 238 ms로 측정되어
   제거한 병목 자체다. `StreamingFallDetector`(0.9 ms 고정)를 유지했다.
2. **GRU 앙상블은 규칙 계층 위에 얹었다.** 분류기는 경보를 올리기만 하며, 자세 증거는
   `StreamingUpdate.torso_angle` / `bbox_aspect`에서 읽는다. 확정 시
   `StreamingFallDetector.force_fallen()`을 호출해 latch 주체를 detector 하나로 유지한다.
3. **해제 정책은 둘 다 남겼다.** 시간 기반 `--fall-hold-seconds`(원격, 기본 30초)를
   detector 안으로 옮기고, 자세 기반 `--auto-clear-seconds`(로컬, 기본 0)와 병존시켰다.
   무인 감시 목적이면 `--fall-hold-seconds 0`이 필요하다는 점을 README에 명시했다.
4. **baseline 주입 파라미터 수용.** `extract_features`가 `baseline_body_height` /
   `baseline_hip_y` override를 받되, 측정 경로는 스트리밍과 공유하는
   `resolve_baseline_height()`를 그대로 쓴다.
5. **`--status-interval` 폐기.** 같은 개념인 `--heartbeat-seconds`로 일원화하고,
   `FALL_STATUS` JSON에 `fall_latched`·`calibrating`을 추가해 soak 테스트 정보를 유지했다.
6. **인터프리터 탐색 유지.** `run_jetson.sh`의 `PYTHON_BIN="${JETSON_PYTHON:-python3}"`는
   버리고 21절의 `find_python()`을 유지했다. 이 보드의 `python3`는 여전히 conda 3.6이다.

## 26. 카메라 실시간 구동과 화면 표시

작업일: 2026-08-23

병합된 코드를 본체에 연결된 카메라로 처음 끝까지 돌리고, 실시간 화면 표시를 붙였다.

### 확인된 환경

```text
카메라    Logitech HD Pro Webcam C920 (/dev/video0, UVC, MJPG/YUYV 640x480 30fps)
디스플레이 X11 :0, GNOME 세션 활성 (XDG_SESSION_TYPE=x11)
```

### 발견한 차단 요인: headless OpenCV

`~/.local`에 **`opencv-python-headless` 4.11.0.86**이 설치되어 있었다. 이 빌드는
`GUI: NONE`이라 `cv2.imshow`가 아예 구현되어 있지 않고, 실행 도중 예외로만 그 사실을
알린다. 무인 로봇에는 맞는 선택이지만 화면 확인에는 쓸 수 없다.

같은 버전의 GUI 빌드(Qt5)로 교체했다. `--no-deps`가 핵심이다. 빼면 NumPy 2.x가 딸려
들어와 OpenCV ABI가 깨진다.

```bash
/usr/bin/python3.10 -m pip uninstall -y opencv-python-headless
/usr/bin/python3.10 -m pip install --user --no-deps opencv-python==4.11.0.86
```

교체 후에도 `numpy 1.26.4`, `torch 2.5.0a0+nv24.08 cuda_available=True`,
`ultralytics 8.4.126`은 그대로다. 같은 실수를 반복하지 않도록 `overlay.gui_available()`이
빌드 정보를 읽어 판정하고, 두 실행기와 `setup_jetson.sh`가 시작 시점에 이를 확인해
교체 명령을 안내한다.

### 정리한 구조

```text
jetson_env.sh    인터프리터·모델·GRU 체크포인트 탐색 (두 실행기가 source)
run_camera.sh    데스크톱 창에 띄우는 실시간 화면
run_jetson.sh    무인 운용, 헤드리스, 상태 파일 발행
overlay.py       상태 색과 화면 표시 (yolo_pose.py와 live_detect.py가 공유)
```

`run_jetson.sh`에 있던 60여 줄의 환경 탐색이 `jetson_env.sh` 하나로 모였고,
`live_detect.py`에 중복돼 있던 `STATE_COLORS`가 사라졌다. GRU 임계값 `0.475`는 앙상블
기준으로 튜닝된 값이므로 체크포인트 두 개가 모두 있을 때만 전달하도록 고쳤다. 하나만
있으면 조용히 다른 임계값으로 돌던 문제다.

### 고친 표시 문제

사람이 한 프레임 검출되지 않을 때마다 상태가 `NORMAL`↔`UNKNOWN`으로 튀었다. 포즈 네트워크의
한 프레임 실패이지 사람이 나간 것이 아니므로, 0.5초 연속으로 검출되지 않아야 `UNKNOWN`으로
내려가도록 debounce를 넣었다. 200프레임 구동에서 상태 발행이 수십 회에서 1회로 줄었다.

### 측정 (C920, 640x480, 창 표시 + 골격 렌더링 포함)

```text
처리 속도       20~23 FPS
추론 median     32.0 ms   p95 38.5 ms
워밍업          약 2.3 s
```

헤드리스(`run_jetson.sh`)보다 낮은 것은 `prediction.plot()` 골격 그리기와 창 갱신 비용이다.
실제 낙상 동작을 카메라 앞에서 수행해 `FALLING` → `FALLEN` 전이와 latch, 그리고 다시
일어섰을 때 `--auto-clear-seconds 5`에 의한 자동 해제까지 확인했다.

### 남은 것

1. GRU 체크포인트가 이 보드에 없다. `make_jetson_bundle.sh`로 PC에서 옮겨야 앙상블이 켜진다.
   현재 카메라 판정은 규칙 계층 단독이다.
2. TensorRT 미설치. `.pt` 폴백이라 추론이 32 ms에 묶여 있다.
3. 10분 이상 연속 구동, `tegrastats` 동시 기록, 거리별 오탐·미탐은 아직 측정하지 않았다.
