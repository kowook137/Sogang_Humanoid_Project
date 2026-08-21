"""Check whether this Jetson is ready to run the fall-detection pipeline.

Reports two classes of problem separately, because they need different actions:

* blockers - the pipeline cannot run at all (no CUDA PyTorch, no model, ...)
* warnings - it runs but degraded (no TensorRT, low memory headroom, ...)

Run it with the SAME interpreter you intend to launch the detector with:

    /usr/bin/python3.10 fall_detection/jetson_preflight.py

Checking with a different interpreter is the most common false result, since a
conda or venv shim early in PATH shadows the JetPack Python that owns the
aarch64 CUDA build of PyTorch.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


MINIMUM_PYTHON = (3, 10)
# CUDA context plus the pose model need roughly this much unified memory. On an
# 8 GB Orin Nano a desktop session can leave less than this free, and CUDA then
# fails at cuBLAS init rather than at allocation time.
MINIMUM_FREE_MEMORY_MB = 1800


def command_output(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    output = (result.stdout or result.stderr).strip()
    return output if result.returncode == 0 else None


def inspect_camera(device: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(device), "exists": device.exists()}
    if device.exists():
        result["formats"] = command_output(
            ["v4l2-ctl", "--device", str(device), "--list-formats-ext"]
        )
    else:
        result["available_video_devices"] = sorted(
            str(path) for path in Path("/dev").glob("video*")
        )
    return result


def memory_status() -> dict[str, object]:
    """Total/available RAM in MB. On Jetson this is also the GPU's memory."""
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = int(rest.split()[0]) // 1024
    except (OSError, ValueError, IndexError):
        return {"total_mb": None, "available_mb": None}
    return {"total_mb": values.get("MemTotal"), "available_mb": values.get("MemAvailable")}


def tegra_release() -> str | None:
    release = Path("/etc/nv_tegra_release")
    if release.is_file():
        return release.read_text(encoding="utf-8", errors="replace").strip()
    return None


def board_model() -> str | None:
    model = Path("/proc/device-tree/model")
    if model.is_file():
        return model.read_text(encoding="utf-8", errors="replace").strip("\x00").strip()
    return None


def python_report() -> dict[str, object]:
    return {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "resolved_python3_on_path": shutil.which("python3"),
        "note": (
            "Launch the detector with this same executable; `python3` on PATH "
            "may resolve elsewhere."
        ),
    }


def import_report() -> dict[str, object]:
    report: dict[str, object] = {}
    try:
        import cv2

        report["opencv"] = cv2.__version__
    except ImportError:
        report["opencv"] = None
    try:
        import numpy

        report["numpy"] = numpy.__version__
    except ImportError:
        report["numpy"] = None
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "built_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            # An "a0+nv" local version marks the NVIDIA aarch64 build. A plain
            # upstream wheel on aarch64 has no CUDA support at all.
            "is_jetpack_build": "nv" in torch.__version__,
        }
    except ImportError:
        report["torch"] = None
    try:
        import ultralytics

        report["ultralytics"] = ultralytics.__version__
    except ImportError:
        report["ultralytics"] = None
    try:
        import tensorrt

        report["tensorrt"] = tensorrt.__version__
    except ImportError:
        report["tensorrt"] = None
    try:
        import rclpy  # noqa: F401

        report["rclpy"] = True
    except ImportError:
        report["rclpy"] = False
    return report


def collect(camera: Path, model: Path) -> dict[str, object]:
    weights = model.with_suffix(".pt")
    report: dict[str, object] = {
        "board": board_model(),
        "machine": platform.machine(),
        "jetson_release": tegra_release(),
        "jetpack_package": command_output(["dpkg-query", "-W", "nvidia-jetpack"]),
        "cuda_compiler": command_output(["nvcc", "--version"]),
        "power_mode": command_output(["nvpmodel", "-q"]),
        "tegrastats_available": shutil.which("tegrastats") is not None,
        "trtexec_available": Path("/usr/src/tensorrt/bin/trtexec").is_file(),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "memory": memory_status(),
        "python": python_report(),
        "camera": inspect_camera(camera),
        "model": {
            "engine": {"path": str(model), "exists": model.is_file()},
            "weights": {"path": str(weights), "exists": weights.is_file()},
        },
    }
    report.update(import_report())
    return report


def problems(report: dict[str, object]) -> tuple[list[str], list[str]]:
    """Split findings into blockers and warnings."""
    blockers: list[str] = []
    warnings: list[str] = []

    python_info = report["python"]
    assert isinstance(python_info, dict)
    if sys.version_info < MINIMUM_PYTHON:
        blockers.append(
            f"Python {python_info['version']} is too old; this code needs "
            f"{MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]}+ (try /usr/bin/python3.10)"
        )
    resolved = python_info["resolved_python3_on_path"]
    if resolved and Path(resolved).resolve() != Path(sys.executable).resolve():
        warnings.append(
            f"`python3` on PATH is {resolved}, not {sys.executable}. "
            "Launch scripts that call plain `python3` may pick the wrong one."
        )

    if report["machine"] not in {"aarch64", "arm64"}:
        blockers.append("This is not an ARM64 Jetson environment")
    if report["jetson_release"] is None:
        warnings.append("/etc/nv_tegra_release is missing; cannot confirm the L4T version")

    torch_info = report["torch"]
    if not isinstance(torch_info, dict):
        blockers.append("PyTorch is not installed for this interpreter")
    elif not torch_info["cuda_available"]:
        blockers.append(
            f"PyTorch {torch_info['version']} cannot reach the GPU "
            f"(built_cuda={torch_info['built_cuda']}). On JetPack, install the "
            "NVIDIA aarch64 wheel; a generic PyPI wheel has no CUDA on ARM."
        )
    elif not torch_info["is_jetpack_build"]:
        warnings.append(
            f"PyTorch {torch_info['version']} does not look like a JetPack build; "
            "a pip upgrade may have replaced it"
        )

    if report["opencv"] is None:
        blockers.append("OpenCV is unavailable")
    if report["ultralytics"] is None:
        blockers.append(
            "Ultralytics is unavailable. Install it without touching the JetPack "
            "torch: pip install --user --no-deps ultralytics ultralytics-thop py-cpuinfo"
        )

    model_info = report["model"]
    assert isinstance(model_info, dict)
    if not model_info["engine"]["exists"] and not model_info["weights"]["exists"]:
        blockers.append(
            f"No pose model found at {model_info['engine']['path']} "
            f"or {model_info['weights']['path']}"
        )
    elif not model_info["engine"]["exists"]:
        warnings.append(
            "No TensorRT engine; running from PyTorch weights. Build one on this "
            "device with: yolo export format=engine imgsz=640 half=True device=0"
        )

    if report["tensorrt"] is None:
        warnings.append(
            "TensorRT python bindings are unavailable, so a FP16 engine cannot be "
            "built here (sudo apt install tensorrt)"
        )

    camera_info = report["camera"]
    assert isinstance(camera_info, dict)
    if not camera_info["exists"]:
        others = camera_info.get("available_video_devices") or []
        detail = f"; devices present: {', '.join(others)}" if others else "; no /dev/video* at all"
        blockers.append(f"Camera device {camera_info['path']} is missing{detail}")

    memory = report["memory"]
    assert isinstance(memory, dict)
    available = memory.get("available_mb")
    if isinstance(available, int) and available < MINIMUM_FREE_MEMORY_MB:
        warnings.append(
            f"Only {available} MB RAM available; CUDA init needs roughly "
            f"{MINIMUM_FREE_MEMORY_MB} MB on this unified-memory board. "
            "Close the desktop session or run headless."
        )

    if not report["rclpy"]:
        warnings.append(
            "rclpy is not importable; source /opt/ros/<distro>/setup.bash before "
            "publishing fall events over ROS 2"
        )
    return blockers, warnings


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=Path, default=Path("/dev/video0"))
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "openpose/models/yolo11n-pose.engine",
    )
    parser.add_argument(
        "--ignore-camera",
        action="store_true",
        help="Do not treat a missing camera as a blocker (bench testing on files)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the JSON report, without the human-readable summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect(args.camera, args.model)
    blockers, warnings = problems(report)
    if args.ignore_camera:
        blockers = [item for item in blockers if not item.startswith("Camera device")]
    report["ready"] = not blockers
    report["blockers"] = blockers
    report["warnings"] = warnings
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not args.json_only:
        print(file=sys.stderr)
        print(f"READY: {'yes' if not blockers else 'no'}", file=sys.stderr)
        for item in blockers:
            print(f"  BLOCKER  {item}", file=sys.stderr)
        for item in warnings:
            print(f"  WARNING  {item}", file=sys.stderr)
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
