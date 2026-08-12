# Fall Detection Video Pipeline

이 폴더는 OpenPose 원본 코드와 분리된 낙상 감지 전처리 파이프라인을 제공한다.

현재 구현 범위:

```text
MP4/AVI 입력
  -> OpenPose BODY_25
  -> 골격 렌더링 MP4
  -> 프레임별 OpenPose JSON
  -> (frames, 25, 3) NPZ
  -> 처리 메타데이터
  -> 정규화된 자세 특징 계산
  -> 규칙 기반 상태 머신 낙상 판정
  -> 평가 CSV 및 상태 표시 영상
```

Windows 포터블 OpenPose는 MP4 직접 기록을 지원하지 않으므로 내부적으로 임시 AVI를
생성한 뒤 OpenCV로 `rendered.mp4`를 만들고 임시 AVI를 제거한다.

휴대폰 세로 영상처럼 MP4 회전 메타데이터가 있는 입력은 OpenPose가 방향을 무시할
수 있다. 스크립트는 회전값을 픽셀에 적용한 임시 입력 영상을 만들어 처리하고 완료 후
임시 입력을 제거한다. 따라서 저장된 keypoint 좌표의 위/아래 방향은 결과 영상과
일치한다.

## 환경

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

## 실행

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

## 출력

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

## 규칙 기반 낙상 판정

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
- `evaluate.py`: 데이터셋 일괄 평가, CSV/JSON/NPZ/판정 영상 생성

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

## Linux에서 이어서 실행

Python 환경은 프로젝트 루트에서 준비한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r fall_detection/requirements.txt
```

OpenPose 소스를 `openpose/build/`에 빌드하고 BODY_25 모델을 내려받으면 실행 파일과 모델을 자동으로 찾는다.

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

### 권장 Linux 백엔드: YOLO pose

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

## Windows 실시간 카메라 판정

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
