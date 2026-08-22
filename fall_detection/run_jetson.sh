#!/usr/bin/env bash
# Unattended on-device fall detection: no window, status published to a file.
# For the on-screen version, use run_camera.sh instead.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=jetson_env.sh
source "${SCRIPT_DIR}/jetson_env.sh"

# Measured on an Orin Nano Super: eager-mode inference costs ~31 ms at every
# input size from 256 to 640, so it is launch-overhead bound, not compute bound.
# Shrinking the input buys no frame rate - only a TensorRT engine does.
#
# --fall-hold-seconds 0 keeps FALLEN latched until the process is restarted or
# a consumer acknowledges it. Unattended monitoring is exactly the case where a
# motionless person on the floor must not quietly go back to NORMAL.
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
  --status-file "${PROJECT_DIR}/outputs/live_fall_status.json" \
  --heartbeat-seconds 1 \
  --fall-hold-seconds 0 \
  "${CLASSIFIER_ARGS[@]}" \
  "$@"
