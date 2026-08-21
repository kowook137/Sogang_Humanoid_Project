#!/usr/bin/env bash
# Provision a Jetson for on-device fall detection. Safe to re-run.
#
# JetPack already ships the only aarch64 PyTorch with CUDA, plus OpenCV and
# NumPy compiled against each other. The whole point of this script is to add
# what is missing WITHOUT letting pip replace any of them - installing
# fall_detection/requirements.txt here would do exactly that and silently drop
# GPU support.
#
#   fall_detection/setup_jetson.sh          # user-level install only
#   fall_detection/setup_jetson.sh --apt    # also run the sudo apt steps
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WEIGHTS="${PROJECT_DIR}/openpose/models/yolo11n-pose.pt"
WEIGHTS_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
# JetPack's NumPy. Pinning it stops pip from pulling NumPy 2.x as a transitive
# dependency, which would break the ABI that JetPack OpenCV was built against.
JETPACK_NUMPY="1.26.4"

RUN_APT=0
[[ "${1:-}" == "--apt" ]] && RUN_APT=1

step() { printf '\n== %s\n' "$1"; }

step "Checking the board"
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This is not an ARM64 Jetson. Run it on the device." >&2
  exit 1
fi
[[ -f /etc/nv_tegra_release ]] && head -1 /etc/nv_tegra_release
[[ -f /proc/device-tree/model ]] && tr -d '\0' < /proc/device-tree/model && echo

step "Locating the JetPack interpreter"
# Deliberately not `python3`: a conda or venv shim early in PATH shadows the
# JetPack interpreter, and only the JetPack one owns the CUDA build of PyTorch.
PYTHON_BIN=""
for candidate in "${JETSON_PYTHON:-}" /usr/bin/python3.10 /usr/bin/python3; do
  [[ -n "${candidate}" && -x "${candidate}" ]] || continue
  if "${candidate}" -c 'import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    PYTHON_BIN="${candidate}"
    break
  fi
done
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No interpreter with CUDA-enabled PyTorch was found." >&2
  echo "JetPack normally provides it at /usr/bin/python3.10. Verify with:" >&2
  echo "  /usr/bin/python3.10 -c 'import torch; print(torch.cuda.is_available())'" >&2
  echo "If that is False, reinstall the NVIDIA aarch64 PyTorch for your JetPack." >&2
  exit 1
fi
echo "using ${PYTHON_BIN}"
if [[ "$(command -v python3 || true)" != "${PYTHON_BIN}" ]]; then
  echo "note: \`python3\` on PATH is $(command -v python3 || echo none), which is NOT this one."
  echo "      Always launch through run_jetson.sh, or call ${PYTHON_BIN} explicitly."
fi

step "Installing Python packages (user-level, JetPack packages untouched)"
"${PYTHON_BIN}" -m pip install --user --no-deps --upgrade \
  ultralytics ultralytics-thop py-cpuinfo
# onnx/onnxslim are only needed to build a TensorRT engine, but pull NumPy 2.x
# unless constrained.
"${PYTHON_BIN}" -m pip install --user --upgrade \
  --constraint <(echo "numpy==${JETPACK_NUMPY}") onnx onnxslim

step "Fetching the pose model"
if [[ -f "${WEIGHTS}" ]]; then
  echo "already present: ${WEIGHTS}"
else
  mkdir -p "$(dirname "${WEIGHTS}")"
  curl -fL --retry 3 -o "${WEIGHTS}" "${WEIGHTS_URL}"
  echo "downloaded: ${WEIGHTS}"
fi

step "System packages"
APT_NEEDED=()
"${PYTHON_BIN}" -c 'import tensorrt' 2>/dev/null || APT_NEEDED+=("tensorrt")
command -v ffmpeg >/dev/null || APT_NEEDED+=("ffmpeg")
command -v v4l2-ctl >/dev/null || APT_NEEDED+=("v4l-utils")
if [[ ${#APT_NEEDED[@]} -eq 0 ]]; then
  echo "nothing missing"
elif [[ ${RUN_APT} -eq 1 ]]; then
  sudo apt update
  sudo apt install -y "${APT_NEEDED[@]}"
else
  echo "missing (need root): ${APT_NEEDED[*]}"
  echo "  sudo apt install -y ${APT_NEEDED[*]}"
  echo "  ...or re-run this script with --apt"
fi

step "Verifying the runtime stack"
"${PYTHON_BIN}" - <<'PY'
import numpy, cv2, torch, ultralytics
print(f"  numpy       {numpy.__version__}")
print(f"  opencv      {cv2.__version__}")
print(f"  torch       {torch.__version__}  cuda_available={torch.cuda.is_available()}")
print(f"  ultralytics {ultralytics.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch lost CUDA access; a pip install replaced the JetPack build.")
# Catch a NumPy ABI mismatch here rather than mid-run.
cv2.cvtColor(numpy.zeros((4, 4, 3), numpy.uint8), cv2.COLOR_BGR2GRAY)
PY

step "Preflight"
"${PYTHON_BIN}" "${SCRIPT_DIR}/jetson_preflight.py" --json-only >/dev/null || true
"${PYTHON_BIN}" "${SCRIPT_DIR}/jetson_preflight.py" 2>&1 >/dev/null || true

cat <<EOF

Setup finished. Next:
  fall_detection/run_jetson.sh          run the detector on camera 0
  ${PYTHON_BIN} -m unittest discover -s fall_detection/tests

A TensorRT engine is optional but is the only way to raise the frame rate on
this board; build it here, never on another machine:
  ${PYTHON_BIN} -m ultralytics export model=${WEIGHTS} format=engine imgsz=640 half=True device=0
EOF
