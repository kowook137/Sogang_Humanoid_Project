#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${JETSON_PYTHON:-python3}"
ENGINE="${JETSON_POSE_MODEL:-${PROJECT_DIR}/openpose/models/yolo11n-pose.engine}"

if [[ ! -f "${ENGINE}" ]]; then
  echo "TensorRT engine not found: ${ENGINE}" >&2
  echo "Create it on this Jetson using the command documented in fall_detection/README.md." >&2
  exit 1
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/yolo_pose.py" \
  --source 0 \
  --model "${ENGINE}" \
  --device 0 \
  --image-size 416 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 15 \
  --headless \
  --no-render \
  "$@"
