"""Check whether a Jetson is ready to run the fall-detection pipeline."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command_output(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    result = subprocess.run(
        [executable, *command[1:]],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    output = (result.stdout or result.stderr).strip()
    return output if result.returncode == 0 else None


def inspect_camera(device: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": str(device), "exists": device.exists()}
    if device.exists():
        result["formats"] = command_output(
            ["v4l2-ctl", "--device", str(device), "--list-formats-ext"]
        )
    return result


def collect(camera: Path, model: Path) -> dict[str, object]:
    report: dict[str, object] = {
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "jetson_release": None,
        "jetpack": command_output(["dpkg-query", "-W", "nvidia-jetpack"]),
        "cuda_compiler": command_output(["nvcc", "--version"]),
        "tegrastats_available": shutil.which("tegrastats") is not None,
        "camera": inspect_camera(camera),
        "model": {"path": str(model), "exists": model.is_file(), "suffix": model.suffix},
    }
    release = Path("/etc/nv_tegra_release")
    if release.is_file():
        report["jetson_release"] = release.read_text(encoding="utf-8", errors="replace").strip()
    try:
        import cv2

        report["opencv"] = cv2.__version__
    except ImportError:
        report["opencv"] = None
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "built_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        report["torch"] = None
    try:
        import ultralytics

        report["ultralytics"] = ultralytics.__version__
    except ImportError:
        report["ultralytics"] = None
    return report


def failures(report: dict[str, object]) -> list[str]:
    issues: list[str] = []
    if report["machine"] not in {"aarch64", "arm64"}:
        issues.append("This is not an ARM64 Jetson environment")
    if report["jetson_release"] is None:
        issues.append("/etc/nv_tegra_release is missing")
    torch_info = report["torch"]
    if not isinstance(torch_info, dict) or not torch_info["cuda_available"]:
        issues.append("CUDA-enabled PyTorch is unavailable")
    if report["opencv"] is None:
        issues.append("OpenCV is unavailable")
    if report["ultralytics"] is None:
        issues.append("Ultralytics is unavailable")
    if not report["camera"]["exists"]:  # type: ignore[index]
        issues.append("Camera device is missing")
    if not report["model"]["exists"]:  # type: ignore[index]
        issues.append("Pose model is missing")
    return issues


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=Path, default=Path("/dev/video0"))
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "openpose/models/yolo11n-pose.engine",
    )
    parser.add_argument("--allow-non-jetson", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect(args.camera, args.model)
    issues = failures(report)
    report["ready"] = not issues
    report["issues"] = issues
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.allow_non_jetson:
        return 0
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
