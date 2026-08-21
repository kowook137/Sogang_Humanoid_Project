#!/usr/bin/env bash
# Launch the on-device fall detector on a Jetson.
#
# Deliberately does NOT trust `python3` from PATH: a conda or venv shim early in
# PATH shadows the JetPack interpreter, and only the JetPack one has the aarch64
# CUDA build of PyTorch. Override with JETSON_PYTHON=/path/to/python if needed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

find_python() {
  if [[ -n "${JETSON_PYTHON:-}" ]]; then
    echo "${JETSON_PYTHON}"
    return
  fi
  local candidate
  for candidate in /usr/bin/python3.10 /usr/bin/python3 "$(command -v python3 || true)"; do
    [[ -x "${candidate}" ]] || continue
    if "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
if sys.version_info < (3, 10):
    raise SystemExit(1)
import torch, cv2, ultralytics
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    then
      echo "${candidate}"
      return
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(find_python)"; then
  echo "No interpreter with CUDA PyTorch + OpenCV + Ultralytics was found." >&2
  echo "On JetPack the working one is usually /usr/bin/python3.10. Check with:" >&2
  echo "  /usr/bin/python3.10 -c 'import torch;print(torch.cuda.is_available())'" >&2
  echo "Install Ultralytics WITHOUT pulling a non-Jetson torch wheel:" >&2
  echo "  /usr/bin/python3.10 -m pip install --user --no-deps ultralytics ultralytics-thop py-cpuinfo" >&2
  exit 1
fi

# Ultralytics writes settings at import time; give it somewhere writable.
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${HOME}/.config/Ultralytics}"
mkdir -p "${YOLO_CONFIG_DIR}"

ENGINE="${JETSON_POSE_MODEL:-${PROJECT_DIR}/openpose/models/yolo11n-pose.engine}"
WEIGHTS="${ENGINE%.engine}.pt"
if [[ -f "${ENGINE}" ]]; then
  MODEL="${ENGINE}"
elif [[ -f "${WEIGHTS}" ]]; then
  # A TensorRT engine is faster, but it is built per JetPack/GPU and cannot be
  # shipped in git. The PyTorch weights are a valid fallback, not an error.
  MODEL="${WEIGHTS}"
  echo "TensorRT engine not found, using PyTorch weights: ${MODEL}" >&2
  echo "Build the engine on this Jetson for lower latency:" >&2
  echo "  sudo apt install tensorrt" >&2
  echo "  ${PYTHON_BIN} -m ultralytics export model=${WEIGHTS} format=engine imgsz=640 half=True device=0" >&2
else
  echo "No pose model found. Expected one of:" >&2
  echo "  ${ENGINE}" >&2
  echo "  ${WEIGHTS}" >&2
  echo "Download the weights:" >&2
  echo "  curl -fL -o ${WEIGHTS} https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt" >&2
  exit 1
fi

# Measured on an Orin Nano Super: eager-mode inference costs ~31 ms at every
# input size from 256 to 640, so it is launch-overhead bound, not compute bound.
# Shrinking the input buys no frame rate - only a TensorRT engine does.
echo "python=${PYTHON_BIN} model=${MODEL}" >&2

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/yolo_pose.py" \
  --source 0 \
  --model "${MODEL}" \
  --device 0 \
  --image-size 640 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 15 \
  --headless \
  --no-render \
  --status-interval 30 \
  "$@"
