# Fall Detection Video Pipeline

이 폴더는 OpenPose 원본 코드와 분리된 낙상 감지 파이프라인을 제공한다. 처리 경로는 두
가지이며 실행하는 장비가 다르다.

## 역할 분담: PC와 Jetson

학습과 검증은 GPU가 있는 PC에서, 실시간 구동은 Jetson에서 한다. Jetson은 학습하지 않는다.

| | PC (NVIDIA GPU) | Jetson Orin Nano |
| --- | --- | --- |
| 목적 | 데이터 준비, 학습, 정확도 검증 | 로봇 탑재 실시간 판정 |
| 입력 | 녹화 영상 파일, 공개 데이터셋 | 카메라 스트림 |
| 자세 추정 | OpenPose BODY_25 또는 YOLO pose | YOLO pose (OpenPose는 빌드 불가) |
| 판정 | `detector.py` 일괄 처리 | `streaming.py` 프레임당 O(1) |
| 산출물 | NPZ, 평가 CSV, 학습 체크포인트 | 실시간 상태와 경보 |

PC 경로 (일괄 처리):

```text
MP4/AVI 입력
  -> OpenPose BODY_25 또는 YOLO pose
  -> 골격 렌더링 MP4 + 프레임별 JSON
  -> (frames, 25, 3) NPZ + 처리 메타데이터
  -> 정규화된 자세 특징 계산
  -> 규칙 기반 상태 머신 낙상 판정 / GRU 학습
  -> 평가 CSV, 상태 표시 영상, 체크포인트
```

Jetson 경로 (실시간):

```text
V4L2 카메라
  -> YOLO11n-pose (TensorRT 엔진 또는 PyTorch 가중치)
  -> BODY_25 호환 (25, 3) 관절
  -> 프레임당 O(1) 스트리밍 특징 갱신
  -> NORMAL / FALLING / FALL_CANDIDATE / FALLEN / UNKNOWN
  -> (예정) ROS 2 publisher
```

구현 파일과 실행 위치:

| 파일 | 역할 | 실행 위치 |
| --- | --- | --- |
| `process_video.py` | OpenPose 일괄 처리, JSON을 NPZ로 변환 | PC |
| `evaluate.py` | 규칙 기반 데이터셋 일괄 평가 | PC |
| `prepare_gmdcsa24.py` | 공개 데이터셋 매니페스트와 분할 생성 | PC |
| `extract_gmdcsa24_poses.py` | 데이터셋 전체 GPU 포즈 추출 | PC |
| `prepare_fallvision.py` | FallVision 관절 CSV를 NPZ로 변환 | PC |
| `download_*.py` | 공개 데이터셋 내려받기 | PC |
| `train_gmdcsa24.py` | 양방향 GRU 학습 | PC |
| `evaluate_fall_ensemble.py` | 앙상블 선택과 고정 test 평가 | PC |
| `live_detect.py` | Windows OpenPose 실시간 판정 | PC |
| `setup_jetson.sh` | Jetson 런타임 설치 | Jetson |
| `jetson_preflight.py` | 구동 가능 여부 사전점검 | Jetson |
| `run_jetson.sh` | 실시간 검출기 실행 | Jetson |
| `features.py` | 자세 특징 계산 | 공통 |
| `detector.py` | 상태 머신과 임곗값 정의 | 공통 |
| `streaming.py` | 프레임당 O(1) 실시간 검출기 | 공통 (주로 Jetson) |
| `yolo_pose.py` | YOLO pose 실행기 (영상 파일 / 카메라) | 공통 |

`features.py`와 `detector.py`의 수식·임곗값은 두 경로가 공유한다. PC에서 녹화 영상으로
맞춘 임곗값이 Jetson 실시간에서도 같은 의미를 갖는다.

## PC: 데이터 준비, 학습, 검증

### Windows 환경

Windows 개발 환경의 Python 인터프리터:

```text
conda activate fall-detection
python --version
```

필요한 Python 패키지:

- NumPy
- OpenCV Python

프로젝트 루트에서 설치할 수 있다.

```powershell
python -m pip install -r fall_detection/requirements.txt
```

OpenPose Windows 포터블 패키지는 기본적으로 다음 상대경로에 둔다.

```text
openpose/openpose-portable/openpose/
```

### Linux 환경

Python 환경은 프로젝트 루트에서 준비한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r fall_detection/requirements.txt
```

이 절차는 GPU가 있는 PC용이다. Jetson에서는 `requirements.txt`를 쓰지 않고
`setup_jetson.sh`를 실행한다.

OpenPose 소스를 `openpose/build/`에 빌드하고 BODY_25 모델을 내려받으면 실행 파일과 모델을 자동으로 찾는다. 이 경로는 PC 전용이다. OpenPose 1.7.0은 구형 Caffe 기반이라 JetPack 6의 CUDA 12.6/cuDNN 9에서는 빌드가 성립하지 않으므로, Jetson에서는 YOLO pose를 쓴다.

```text
openpose/build/examples/openpose/openpose.bin
openpose/models/pose/body_25/pose_iter_584000.caffemodel
```

```bash
python fall_detection/process_video.py \
  --input data/raw_videos/p01_fall_side_001.mp4
```

다른 위치에 빌드했다면 OpenPose 소스 루트 또는 빌드 디렉터리를 지정한다.

```bash
python fall_detection/process_video.py \
  --input data/raw_videos/p01_fall_side_001.mp4 \
  --openpose-root /path/to/openpose
```

실시간 카메라도 같은 실행 파일 탐색 방식을 사용한다.

```bash
python fall_detection/live_detect.py --camera 0 --camera-resolution 640x480
```

WSL에서 실행할 때는 Windows 드라이버, WSL GPU 전달, 카메라 장치와 GUI 전달이 모두 준비되어야 한다. 실물 Linux 또는 Jetson과 장치 접근 방식이 다르므로 각각 확인한다.

### 녹화 영상 OpenPose 처리

프로젝트 루트에서 실행한다.

```powershell
python fall_detection/process_video.py `
  --input data/raw_videos/p01_fall_side_001.mp4
```

Conda 환경을 현재 셸에서 활성화하지 않고 실행하려면:

```powershell
conda run -n fall-detection python `
  fall_detection/process_video.py `
  --input data/raw_videos/p01_fall_side_001.mp4
```

출력 경로를 직접 지정할 수도 있다.

```powershell
python fall_detection/process_video.py `
  --input data/raw_videos/p01_fall_side_001.mp4 `
  --output-dir outputs/custom_fall_test `
  --net-resolution "-1x256"
```

기존 결과를 실수로 덮어쓰지 않도록 출력 파일이 이미 존재하면 실행을 중단한다.


Windows 포터블 OpenPose는 MP4 직접 기록을 지원하지 않으므로 내부적으로 임시 AVI를
생성한 뒤 OpenCV로 `rendered.mp4`를 만들고 임시 AVI를 제거한다.

휴대폰 세로 영상처럼 MP4 회전 메타데이터가 있는 입력은 OpenPose가 방향을 무시할
수 있다. 스크립트는 회전값을 픽셀에 적용한 임시 입력 영상을 만들어 처리하고 완료 후
임시 입력을 제거한다. 따라서 저장된 keypoint 좌표의 위/아래 방향은 결과 영상과
일치한다.
### 출력 형식

기본 출력 위치는 `outputs/<입력 파일명>/`이다.

```text
outputs/p01_fall_side_001/
├── rendered.mp4
├── keypoints_json/
│   └── *_keypoints.json
├── keypoints.npz
└── metadata.json
```

`keypoints.npz`의 주요 배열:

- `keypoints`: `(frame_count, 25, 3)` float32
- `frame_files`: 원본 OpenPose JSON 파일명
- `joint_names`: BODY_25 관절명
- `fps`: 원본 FPS
- `frame_width`, `frame_height`: 원본 해상도

좌표 순서는 `(x, y, confidence)`다. 사람이 없거나 관절을 검출하지 못한 경우 `x`, `y`는 `NaN`, confidence는 `0`으로 저장한다. 프레임은 제거하지 않으므로 원본 영상과 시간축이 유지된다.

### 규칙 기반 일괄 평가와 판정 방식

OpenPose 처리가 완료된 데이터셋 전체를 평가한다.

```powershell
python fall_detection/evaluate.py
```

프레임마다 판정 상태와 특징값이 표시된 영상을 함께 만들려면 다음과 같이 실행한다.

```powershell
python fall_detection/evaluate.py --write-videos
```

기본적으로 `data/raw_videos/`에 존재하는 영상과 같은 이름의 `outputs/<영상명>/`만 평가한다. 파일명의 `_fall_`, `_normal_` 부분에서 개발용 정답 라벨을 읽는다.

주요 구현 파일:

- `features.py`: BODY_25 좌표에서 골반·머리 하강 속도, 몸통 각도, 바운딩 박스 비율 계산
- `detector.py`: `NORMAL`, `FALLING`, `FALL_CANDIDATE`, `FALLEN`, `UNKNOWN` 상태 머신
- `streaming.py`: 같은 특징·임곗값을 프레임당 O(1)로 갱신하는 실시간 검출기
- `evaluate.py`: 데이터셋 일괄 평가, CSV/JSON/NPZ/판정 영상 생성

녹화 영상 일괄 평가는 `features.py` + `detector.py`를, 카메라 실시간 판정은
`streaming.py`를 쓴다. 특징 수식과 `DetectorConfig` 임곗값은 완전히 공유하므로 녹화
영상으로 맞춘 임곗값이 실시간에서도 같은 의미를 갖는다. 실시간 경로에서만 다른 두 가지는
의도적이다. 첫째, 기준 신체 높이와 골반 위치를 세션 시작의 보정 구간에서 한 번만 정하고
고정한다. 둘째, `FALLEN`은 낙상 근거가 오래 지나도 유지되며 `reset_alarm()`으로만 해제된다.
이전 실시간 구현은 슬라이딩 윈도우로 매 프레임 전체를 다시 계산했기 때문에, 낙상 장면이
윈도우 밖으로 밀려나는 약 20초 뒤에 쓰러진 사람을 다시 `NORMAL`로 보고했다.

추가 출력:

```text
outputs/
├── fall_evaluation.csv
└── <영상 파일명>/
    ├── fall_detection.mp4
    ├── fall_detection.json
    └── fall_detection.npz
```

판정기는 다음 증거를 함께 사용한다.

1. 초기 신체 높이로 정규화한 머리·골반 하강 속도
2. 약 1초 동안의 머리·골반 하강량
3. 어깨 중심과 골반 중심으로 계산한 몸통 각도
4. 관절 바운딩 박스의 폭/높이 비율
5. 초기 위치 대비 낮아진 골반 위치
6. 낙상 움직임 직후 바닥 근처에서 발생한 골격 검출 소실

속도는 프레임당 픽셀이 아닌 `신체 높이/초` 단위로 계산하므로 입력 FPS가 달라도 같은 기준을 사용할 수 있다. 짧은 관절 누락만 보간하고, 긴 누락은 정상 좌표 0으로 바꾸지 않는다. 검출 소실만으로 낙상을 선언하지 않으며 직전에 빠른 하강이 있었을 때만 보조 증거로 사용한다.

현재 규칙은 2명·20개 개발 영상으로 만든 초기 기준선이다. 같은 데이터로 임곗값을 조정했기 때문에 여기서 계산된 성능을 실제 환경 정확도로 해석하면 안 된다. 새로운 사람과 장소에서 촬영한 별도 테스트 세트로 반드시 재평가해야 한다.

### GMDCSA-24 공개 데이터셋

공식 Zenodo v2.0 레코드에서 ZIP을 내려받아 다음 위치에 압축 해제한다.

- DOI: `10.5281/zenodo.12921216`
- 파일: `gmdcsa24-v2.0.zip` (1,107,543,412 bytes)
- MD5: `49bf4eb15a84cc84cb0a4f9c6ddd59e6`
- 라이선스: CC BY 4.0 (논문 및 데이터셋 인용 필요)

원본과 생성된 매니페스트는 `data/datasets/`에 두며 Git에는 포함하지 않는다.
압축 해제 후 아래 명령은 160개 영상을 기존 파이프라인용 이름으로 연결하고
`manifest.csv`를 만든다. 영상 파일은 복사하지 않고 상대 심볼릭 링크를 사용한다.

```bash
mkdir -p data/datasets/gmdcsa24/extracted
curl -fL --continue-at - \
  --output data/datasets/gmdcsa24/gmdcsa24-v2.0.zip \
  https://zenodo.org/api/records/12921216/files/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-v2.0.zip/content
echo "49bf4eb15a84cc84cb0a4f9c6ddd59e6  data/datasets/gmdcsa24/gmdcsa24-v2.0.zip" | md5sum --check
unzip data/datasets/gmdcsa24/gmdcsa24-v2.0.zip \
  -d data/datasets/gmdcsa24/extracted
python3 fall_detection/prepare_gmdcsa24.py
```

기본 참가자 분할은 Subject 1·2를 train, Subject 3을 validation, Subject 4를
test로 사용한다. 같은 사람의 영상이 서로 다른 분할에 섞여 성능이 부풀려지는 것을
막기 위한 기준선이며, 최종 보고에는 4-fold leave-one-subject-out 평가를 권장한다.

#### GPU 포즈 추출 및 GRU 학습

GMDCSA-24 전체 영상에서 YOLO 관절 좌표를 추출한다. 중간 NPZ는
`data/datasets/gmdcsa24/poses/`에 저장되며 이미 완료된 영상은 다시 처리하지 않는다.

```bash
.venv/bin/python fall_detection/extract_gmdcsa24_poses.py \
  --device 0 --batch-size 16
```

추출한 관절의 정규화된 `(x, y, confidence)` 시계열로 양방향 GRU 영상 분류기를
학습한다. validation F1을 기준으로 최적 모델을 저장하고 30 epoch 동안 개선이 없으면
중단한다.

```bash
.venv/bin/python fall_detection/train_gmdcsa24.py \
  --device cuda --epochs 200 --patience 30
```

산출물은 Git에서 제외된 `outputs/gmdcsa24_training/`에 생성된다.

```text
outputs/gmdcsa24_training/
├── best_model.pt
└── metrics.json
```

이 모델은 영상 전체를 입력으로 `normal/fall`을 분류한다. GMDCSA-24에는 정확한 낙상
시작 프레임 라벨이 없으므로 실시간 `FALLING` 시작 시점을 학습한 모델은 아니다.
실시간 상태 전이는 기존 규칙 기반 판정기와 결합하거나 프레임 라벨 데이터로 별도
학습해야 한다.

#### FallVision 보조 학습 데이터

FallVision의 공개 관절 CSV는 영상 원본을 다시 추론하지 않고 학습 데이터로 변환할 수
있다. Harvard Dataverse에서 20개 RAR를 내려받고 압축을 해제하려면 `7z`가 필요하다.

```bash
.venv/bin/python fall_detection/download_fallvision_keypoints.py
.venv/bin/python fall_detection/prepare_fallvision.py
```

현재 공개 파일에서는 총 5,864개 시퀀스(낙상 3,000, 정상 2,864)가 생성된다. 이
시퀀스는 모두 GMDCSA-24의 **train split에만** 추가되며 validation/test subject에는
섞이지 않는다. 기본 학습 명령은 FallVision 매니페스트가 있으면 자동으로 사용한다.
GMDCSA-24만 사용하려면 `--no-fallvision`을 지정한다.

두 종류의 입력 특징으로 모델을 학습하고 validation subject에서 앙상블 가중치와
임곗값을 고른 뒤, test subject를 한 번 평가할 수 있다.

```bash
.venv/bin/python fall_detection/train_gmdcsa24.py \
  --device cuda --output-dir outputs/combined_training
.venv/bin/python fall_detection/train_gmdcsa24.py \
  --device cuda --feature-mode engineered \
  --output-dir outputs/engineered_combined_training
.venv/bin/python fall_detection/evaluate_fall_ensemble.py --device cuda
```

2026-08-21 기준 고정 분할 실험에서 validation으로 선택된 앙상블은 pose 0.3,
engineered 0.7, threshold 0.35였다. 한 번만 평가한 Subject 4 test 결과는 accuracy
0.730, precision 0.640, recall 0.941, F1 0.762다. 표본이 37개뿐인 단일 subject
결과이므로 일반화 성능으로 단정하지 말고, 최종 보고에는 4-fold
leave-one-subject-out 평가와 fold별 평균·표준편차를 사용한다.

### YOLO pose 백엔드 (녹화 영상 확인용)

최신 Linux와 CPU에서도 간단히 실행할 수 있도록 YOLO pose 실행기를 제공한다. COCO 17개 관절을 기존 낙상 판정기가 사용하는 BODY_25 배열에 대응시키고, Neck과 MidHip은 좌우 관절 중심으로 계산한다.

녹화 영상:

```bash
python fall_detection/yolo_pose.py \
  --source openpose/examples/media/video.avi \
  --device cpu \
  --output outputs/yolo_pose.mp4
```

웹캠:

```bash
python fall_detection/yolo_pose.py \
  --source 0 --device cpu \
  --camera-width 640 --camera-height 480 --camera-fps 15
```

GUI가 없는 환경에서는 `--headless`를 사용하고 `--output`으로 결과를 저장한다. NVIDIA GPU가 PyTorch에서 정상 인식되면 `--device 0`을 사용한다. 기본 모델은 `openpose/models/yolo11n-pose.pt`이며 Git에는 포함하지 않는다.

기본값 `--device auto`는 CUDA가 실제로 사용 가능하면 GPU 0을 선택하고, 그렇지 않으면 CPU를 선택한다. GPU 사용을 강제해 환경 문제를 즉시 확인하려면 `--device 0`을 지정한다. 이 프로젝트에서 검증한 CUDA 12.4 조합은 다음과 같이 별도로 설치한다.

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

녹화 영상을 웹캠처럼 반복·실시간 재생하여 전체 파이프라인을 시험할 수도 있다.

```bash
python fall_detection/yolo_pose.py \
  --source data/raw_videos/p01_fall_side_001.mp4 \
  --loop --realtime --device auto
```

실제 가상 카메라(OBS Virtual Camera 등)는 운영체제에서 일반 카메라 번호로 보이므로 `--source 0`, `--source 1`처럼 지정한다.

Windows에서는 `fall_detection\run_yolo_gpu.bat`을 실행하면 기본 카메라 0번과 GPU 0번으로 바로 시작한다. 다른 카메라를 쓰려면 뒤에 인자를 추가한다(같은 인자가 여러 번 있으면 마지막 값이 적용된다).

```powershell
fall_detection\run_yolo_gpu.bat --source 1
```

WSL에서 `/dev/video0`가 없다면 웹캠 코드 문제가 아니라 Windows 카메라가 Linux에 전달되지 않은 상태다. 카메라를 USB 장치로 WSL에 연결하거나 실물 Linux 환경에서 실행해야 한다.

### Windows 실시간 카메라 판정 (OpenPose)

실시간 판정은 Windows용 `OpenPoseDemo.exe`를 실행하면서 생성되는 BODY_25 JSON을 감시하고 동일한 규칙 기반 상태 머신에 전달한다.

사전 준비:

1. Python 3.10 이상 환경을 준비한다.
2. `requirements.txt`의 패키지를 설치한다.
3. OpenPose 1.7.0 Windows GPU 포터블 패키지와 BODY_25 모델을 다음 위치에 둔다.

```text
openpose/openpose-portable/openpose/
├── bin/OpenPoseDemo.exe
└── models/pose/body_25/pose_iter_584000.caffemodel
```

OpenPose 실행 파일과 모델은 용량이 크므로 이 Git 저장소에 포함되지 않는다. 다른 컴퓨터에서도 위 상대경로에 별도로 준비해야 한다. NVIDIA GPU 드라이버와 OpenPose 포터블 패키지가 지원하는 CUDA 런타임도 필요하다.

실행:

```powershell
python fall_detection/live_detect.py --camera 0 --camera-resolution 640x480
```

또는 `fall_detection/run_live.bat`을 실행한다. 배치 파일은 자신의 위치를 기준으로 프로젝트 루트로 이동하므로 사용자명이나 저장소 절대경로에 의존하지 않는다.

카메라가 여러 개면 번호를 변경한다.

```powershell
python fall_detection/live_detect.py --camera 1 --camera-resolution 640x480
```

실행 후 처음 2초간 전신이 보이도록 서 있어야 초기 신체 높이와 골반 위치가 설정된다. OpenPose 창에는 골격이, `Fall Detection Status` 창에는 다음 상태가 표시된다.

```text
NORMAL / FALLING / FALL_CANDIDATE / FALLEN / UNKNOWN
```

판정 창에서 `Q` 또는 `Esc`를 누르면 종료된다. 실시간 JSON은 Git에서 제외된 `tmp/live_detection/<실행시각>/`에 저장된다.

실행 계층은 Windows 포터블의 `OpenPoseDemo.exe`와 Linux 빌드의 `openpose.bin`을 모두 찾는다. 다만 실제 카메라·GPU·GUI 동작은 현재 Windows에서만 확인된 초기 프로토타입이다. Jetson에서는 OpenPose 빌드 성능을 측정한 뒤 필요하면 TensorRT 기반 자세 추정기로 교체해야 한다.

### WSL에서 USB 웹캠 연결

Windows에 `usbipd-win`을 설치하고 관리자 PowerShell에서 최초 한 번 카메라를 공유한다. BUSID는 PC마다 달라지므로 `usbipd list` 결과를 사용한다.

```powershell
winget install --exact --id dorssel.usbipd-win
usbipd list
usbipd bind --busid <BUSID>
```

일반 PowerShell에서 WSL이 실행 중인 상태로 카메라를 연결한다. `bind`는 유지되지만 `attach`는 Windows나 WSL 재시작 및 USB 재연결 후 다시 실행해야 한다.

```powershell
usbipd attach --wsl --busid <BUSID>
```

이 PC의 Logitech C920은 확인 당시 BUSID `2-7`이었다. 연결 중에는 Windows 프로그램에서 같은 카메라를 사용할 수 없다. 반환하려면 다음을 실행한다.

```powershell
usbipd detach --busid <BUSID>
```

C920을 WSL USB/IP로 사용할 때 기본 YUYV 스트림은 timeout이 발생할 수 있어 `yolo_pose.py`는 카메라 입력에 MJPEG 640×480 15 FPS를 기본 요청한다. USB/IP에서는 실제 전송 속도가 설정값보다 크게 낮거나 손상 프레임이 생길 수 있으므로 실시간 운용은 네이티브 Linux를 권장한다.

테스트는 실제 OpenPose나 카메라 없이 실행할 수 있다.

```bash
python -m unittest discover -s fall_detection/tests -v
```

## Jetson: 실시간 구동

Jetson에서는 Windows OpenPose 패키지를 사용하지 않는다. 온디바이스 경로는
`카메라 → YOLO11n-pose → 규칙 기반 스트리밍 상태 머신`으로 고정한다. 학습된 GRU는
영상 전체 분류기라 실시간 경보 노드에는 연결하지 않는다.

### 검증된 실행 환경

2026-08-22에 실제 보드에서 확인한 값이다.

```text
보드      NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super (8GB)
L4T       R36.4.7 (JetPack 6.2 계열), Ubuntu 22.04, 전력 모드 MAXN_SUPER
인터프리터 /usr/bin/python3.10 (3.10.12)
PyTorch   2.5.0a0+872d972e41.nv24.08 / CUDA 12.6 / cuda_available=True / Orin sm_87
OpenCV    4.11.0   NumPy 1.26.4   Ultralytics 8.4.126
```

### 인터프리터 주의

**`python3`를 그냥 쓰면 안 된다.** 이 보드의 `PATH` 앞쪽에는
`~/jetconda3/bin/python3`(Python 3.6.1)가 있어서, 그대로 실행하면 이 프로젝트 코드가
문법 오류로 죽는다. CUDA PyTorch를 가진 인터프리터는 JetPack의 `/usr/bin/python3.10`
하나뿐이다.

```bash
/usr/bin/python3.10 -c "import torch; print(torch.cuda.is_available())"   # True 여야 한다
```

`run_jetson.sh`는 PATH를 신뢰하지 않고 `torch.cuda` + OpenCV + Ultralytics가 모두
import되는 인터프리터를 직접 찾는다. 다른 것을 쓰려면 `JETSON_PYTHON`으로 지정한다.

### 설치

설치 스크립트 하나로 끝난다. 여러 번 실행해도 안전하다.

```bash
fall_detection/setup_jetson.sh          # 사용자 권한 설치만
fall_detection/setup_jetson.sh --apt    # sudo apt 단계까지 함께 실행
```

스크립트는 JetPack 인터프리터를 직접 찾고, Ultralytics를 `--no-deps`로 설치하고,
`onnx`/`onnxslim`은 NumPy를 고정한 채 설치하고, 포즈 가중치를 내려받은 뒤 사전점검까지
돌린다. 마지막에 CUDA와 NumPy ABI를 다시 확인하므로 설치가 스택을 망가뜨렸는지 즉시 알 수 있다.

**`requirements.txt`를 Jetson에서 설치하면 안 된다.** JetPack이 PyTorch, OpenCV, NumPy를
서로 맞춰 제공하는데 pip가 이들을 CUDA 없는 일반 aarch64 wheel로 덮어쓴다. 특히 `onnx`를
제약 없이 설치하면 NumPy 2.x가 딸려 들어와 JetPack OpenCV의 ABI가 깨진다.

`sudo`가 필요한 항목(`tensorrt`, `ffmpeg`)은 스크립트가 명령을 출력만 하고 넘어간다.

### 사전점검

```bash
/usr/bin/python3.10 fall_detection/jetson_preflight.py
```

blocker(실행 불가)와 warning(성능 저하)을 분리해 보고한다. 카메라 없이 영상 파일로만
시험할 때는 `--ignore-camera`를 붙인다. 반드시 실행에 사용할 인터프리터로 점검한다.

### 실행

```bash
fall_detection/run_jetson.sh                 # 카메라 0, GPU 0, 헤드리스
fall_detection/run_jetson.sh --source 1      # 뒤에 붙인 인자가 기본값을 덮어쓴다
```

TensorRT 엔진이 있으면 그것을, 없으면 `.pt` 가중치를 자동으로 쓴다. 엔진이 없다고
실패하지 않는다.

### 측정된 성능 (2026-08-22, 데스크톱 세션이 함께 떠 있는 상태)

`openpose/examples/media/video.avi` 205프레임, `--headless --no-render`, GPU 0 기준이다.

```text
end-to-end          18.6 FPS  (개선 전 10.3 FPS)
추론 median/p95     35.1 ms / 42.3 ms
워밍업 첫 프레임    약 2.4 s
```

입력 해상도는 성능에 영향이 없다. 순수 추론시간을 40회씩 측정한 결과 imgsz 256/320/416/640이
모두 30~32 ms였다. 즉 이 모델은 연산량이 아니라 **커널 실행 오버헤드에 묶여 있다.**
해상도를 낮춰도 프레임레이트가 오르지 않으므로 기본값을 640으로 둔다. 속도를 올리는
유일한 수단은 TensorRT 엔진이다.

규칙 계층은 프레임당 **0.9 ms 고정**이다. 30분(27,000프레임) 연속 처리에서 프레임당
시간이 0.917 → 0.903 ms로 평탄했고 RSS는 28 MB로 변하지 않았다.

본체를 바꾸거나 TensorRT 엔진을 만든 뒤에는 카메라 없이 같은 영상으로 다시 측정한다.
헤드리스·비렌더링 모드는 골격 그리기와 GUI 비용을 제외한다.

```bash
/usr/bin/python3.10 fall_detection/yolo_pose.py \
  --source openpose/examples/media/video.avi \
  --device 0 --headless --no-render --max-frames 100
```

출력 끝줄의 `average_fps`, `inference_median_ms`, `inference_p95_ms`를 기록한다.
`warmup_ms`는 첫 프레임의 CUDA 초기화 비용이라 정상 운용 수치에서 제외한다.

### TensorRT FP16 엔진

현재 이 보드에는 TensorRT가 설치되어 있지 않다(`tensorrt` 미설치, `trtexec` 없음).
엔진은 JetPack·TensorRT·GPU에 종속되므로 커밋하지 않고 **실행할 본체에서** 만든다.

```bash
sudo apt install tensorrt
/usr/bin/python3.10 -m ultralytics export \
  model=openpose/models/yolo11n-pose.pt \
  format=engine imgsz=640 half=True device=0
```

생성 후 `run_jetson.sh`가 자동으로 엔진을 집는다.

### 메모리 주의

Orin Nano는 CPU와 GPU가 8GB를 공유한다. 데스크톱 세션(Chrome, VS Code, GNOME)이 떠 있으면
가용 메모리가 2.7GB까지 떨어지고, 그 상태에서 CUDA 초기화가
`CUBLAS_STATUS_ALLOC_FAILED`로 실패한 사례를 확인했다. 실제 운용은 헤드리스로 하고,
사전점검의 가용 메모리 warning을 확인한다.

### 본체에서 반드시 기록할 항목

- `tegrastats`의 RAM, GPU 사용률, 온도와 throttling 여부
- 실제 카메라로 최소 10분 연속 실행한 average FPS, inference median/p95
- 카메라 실제 FPS와 장시간 프레임 누락 여부
- 사람 전신이 보이는 거리별 오탐·미탐과 시간당 오경보
- 로봇 정지/진동/보행 상태별 결과

위 18.6 FPS는 녹화 영상 기준이며 카메라 입력·조명·거리 조건이 빠져 있다. 실제 카메라를
연결한 뒤 다시 측정해야 운용 수치로 쓸 수 있다.
