#!/usr/bin/env bash
# Shared environment discovery for the on-device runners. Source it, do not run
# it: it exports PYTHON_BIN, MODEL and CLASSIFIER_ARGS for the caller to use.
#
# Deliberately does NOT trust `python3` from PATH: a conda or venv shim early in
# PATH shadows the JetPack interpreter, and only the JetPack one has the aarch64
# CUDA build of PyTorch. Override with JETSON_PYTHON=/path/to/python if needed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CLASSIFIER="${JETSON_FALL_CLASSIFIER:-${PROJECT_DIR}/outputs/fallvision_engineered_training/best_model.pt}"
POSE_CLASSIFIER="${JETSON_POSE_CLASSIFIER:-${PROJECT_DIR}/outputs/fallvision_pose_training/best_model.pt}"

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

# The GRU checkpoints are optional: without them the rule layer runs alone. The
# threshold belongs to the *ensemble* it was tuned on, so it is only applied
# when the pair is complete - a single checkpoint keeps yolo_pose.py's default.
CLASSIFIER_ARGS=()
if [[ -f "${CLASSIFIER}" ]]; then
  CLASSIFIER_ARGS+=(--classifier "${CLASSIFIER}")
fi
if [[ -f "${POSE_CLASSIFIER}" ]]; then
  CLASSIFIER_ARGS+=(--pose-classifier "${POSE_CLASSIFIER}")
fi
if (( ${#CLASSIFIER_ARGS[@]} > 0 )); then
  # The GPU is already the pose network's; the GRU is small enough for the CPU.
  CLASSIFIER_ARGS+=(--classifier-device cpu)
fi
if [[ -f "${CLASSIFIER}" && -f "${POSE_CLASSIFIER}" ]]; then
  CLASSIFIER_ARGS+=(--classifier-pose-weight 0.45 --classifier-threshold 0.475)
fi

echo "python=${PYTHON_BIN} model=${MODEL} classifiers=$(( ${#CLASSIFIER_ARGS[@]} > 0 ))" >&2
