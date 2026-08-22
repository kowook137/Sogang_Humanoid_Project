#!/usr/bin/env bash
# Live fall detection on the attached camera, drawn in a window on this desktop.
#
# Q or ESC closes it, R acknowledges a raised alarm. For unattended operation
# with no display, use run_jetson.sh instead.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=jetson_env.sh
source "${SCRIPT_DIR}/jetson_env.sh"

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "No display found (DISPLAY and WAYLAND_DISPLAY are both unset)." >&2
  echo "Run this from the Jetson desktop session, or over SSH with:" >&2
  echo "  DISPLAY=:0 fall_detection/run_camera.sh" >&2
  echo "For a headless machine use run_jetson.sh instead." >&2
  exit 1
fi

# 30 FPS from the camera against ~35 ms of inference: frames are dropped rather
# than queued, which keeps what is on screen current instead of falling behind.
#
# --auto-clear-seconds 5 suits a demo: the alarm releases once the person is
# back on their feet, so a fall can be shown repeatedly without a restart.
# --fall-hold-seconds 0 leaves it latched for as long as they are still down.
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/yolo_pose.py" \
  --source "${JETSON_CAMERA:-0}" \
  --model "${MODEL}" \
  --device 0 \
  --image-size 640 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --heartbeat-seconds 0 \
  --fall-hold-seconds 0 \
  --auto-clear-seconds 5 \
  --status-file "${PROJECT_DIR}/outputs/live_fall_status.json" \
  "${CLASSIFIER_ARGS[@]}" \
  "$@"
