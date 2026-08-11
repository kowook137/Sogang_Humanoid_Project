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
