"""Run the rule-based fall detector on a Windows OpenPose camera stream."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detector import FallState, detect_fall
from features import extract_features
from process_video import select_person
from openpose_runtime import find_openpose


STATE_COLORS = {
    FallState.NORMAL: (0, 200, 0),
    FallState.FALLING: (0, 200, 255),
    FallState.FALL_CANDIDATE: (0, 120, 255),
    FallState.FALLEN: (0, 0, 255),
    FallState.UNKNOWN: (160, 160, 160),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Resolution must look like 640x480") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution dimensions must be positive")
    return width, height


def read_frame(path: Path) -> np.ndarray | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return select_person(payload.get("people", []))


def estimated_fps(timestamps: deque[float], fallback: float) -> float:
    if len(timestamps) < 5:
        return fallback
    intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
    intervals = intervals[(intervals > 0.001) & (intervals < 1.0)]
    if not intervals.size:
        return fallback
    return float(np.clip(1.0 / np.median(intervals), 2.0, 60.0))


def dashboard(
    state: FallState,
    fps: float,
    frame_count: int,
    detected: bool,
    message: str,
) -> np.ndarray:
    image = np.zeros((300, 720, 3), dtype=np.uint8)
    color = STATE_COLORS[state]
    cv2.putText(image, "LIVE FALL DETECTION", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(image, state.name, (24, 125), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
    cv2.putText(image, f"pose={'YES' if detected else 'NO'}  processing_fps={fps:.1f}  frames={frame_count}", (24, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
    cv2.putText(image, message, (24, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 1)
    cv2.putText(image, "Press Q or ESC to stop", (24, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live OpenPose rule-based fall detection.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-resolution", type=parse_resolution, default=(640, 480))
    parser.add_argument("--fallback-fps", type=float, default=10.0)
    parser.add_argument("--net-resolution", default="-1x256")
    parser.add_argument(
        "--openpose-root",
        type=Path,
        help="OpenPose source/build or portable root (default: auto-detect)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = find_openpose(project_root(), args.openpose_root)

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_dir = project_root() / "tmp" / "live_detection" / session
    json_dir.mkdir(parents=True, exist_ok=False)
    width, height = args.camera_resolution
    command = [
        str(runtime.executable),
        "--camera", str(args.camera),
        "--camera_resolution", f"{width}x{height}",
        "--model_folder", str(runtime.model_dir),
        "--model_pose", "BODY_25",
        "--number_people_max", "1",
        "--net_resolution", args.net_resolution,
        "--write_json", str(json_dir),
        "--display", "1",
    ]

    process = subprocess.Popen(command, cwd=runtime.working_dir)
    keypoints: list[np.ndarray] = []
    timestamps: deque[float] = deque(maxlen=60)
    consumed: set[Path] = set()
    current = FallState.UNKNOWN
    started = time.monotonic()
    cv2.namedWindow("Fall Detection Status", cv2.WINDOW_NORMAL)

    try:
        while process.poll() is None:
            for path in sorted(json_dir.glob("*_keypoints.json")):
                if path in consumed:
                    continue
                frame = read_frame(path)
                if frame is None:
                    continue
                consumed.add(path)
                keypoints.append(frame)
                timestamps.append(time.monotonic())

            fps = estimated_fps(timestamps, args.fallback_fps)
            detected = bool(keypoints and np.any(keypoints[-1][:, 2] >= 0.2))
            message = "Stand still for 2 seconds to calibrate"
            if len(keypoints) >= max(5, round(fps * 2.0)):
                features = extract_features(np.stack(keypoints), fps, width, height)
                result = detect_fall(features)
                current = FallState(int(result.states[-1]))
                message = "FALL ALERT" if current == FallState.FALLEN else "Monitoring"
            elif time.monotonic() - started < 2.0:
                current = FallState.UNKNOWN

            cv2.imshow("Fall Detection Status", dashboard(current, fps, len(keypoints), detected, message))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                process.terminate()
                break
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        cv2.destroyAllWindows()

    print(f"Live session JSON: {json_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
