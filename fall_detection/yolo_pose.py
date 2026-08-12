"""Run YOLO pose on a video or webcam and feed the existing fall detector."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from collections import deque
from pathlib import Path

_TEMP_DIR = Path(tempfile.gettempdir())
os.environ.setdefault("YOLO_CONFIG_DIR", str(_TEMP_DIR / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(_TEMP_DIR / "matplotlib"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from detector import FallState, detect_fall
from features import extract_features


COCO_TO_BODY25 = {
    0: 0,   # nose
    6: 2,   # right shoulder
    8: 3,
    10: 4,
    5: 5,   # left shoulder
    7: 6,
    9: 7,
    12: 9,  # right hip
    14: 10,
    16: 11,
    11: 12, # left hip
    13: 13,
    15: 14,
    2: 15,  # right eye
    1: 16,
    4: 17,  # right ear
    3: 18,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_source(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def resolve_device(requested: str) -> str:
    """Resolve auto/CPU/CUDA selection and fail early for an unavailable GPU."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda:"):
        normalized = normalized.split(":", 1)[1]
    if normalized.isdecimal():
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device was requested but PyTorch cannot access a GPU. "
                f"torch={torch.__version__}, built_cuda={torch.version.cuda}. "
                "In WSL, confirm that /dev/dxg exists and nvidia-smi works; "
                "otherwise run with --device cpu."
            )
        if int(normalized) >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {normalized} does not exist; "
                f"available device count={torch.cuda.device_count()}"
            )
        return normalized
    raise ValueError("Device must be auto, cpu, or a CUDA index such as 0")


def midpoint(points: np.ndarray, left: int, right: int) -> np.ndarray:
    pair = points[[left, right]]
    valid = np.isfinite(pair[:, :2]).all(axis=1) & (pair[:, 2] > 0)
    if not np.any(valid):
        return np.asarray([np.nan, np.nan, 0.0], dtype=np.float32)
    confidence = pair[valid, 2]
    xy = np.average(pair[valid, :2], axis=0, weights=confidence)
    return np.asarray([xy[0], xy[1], float(np.mean(confidence))], dtype=np.float32)


def coco_to_body25(xy: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    result = np.full((25, 3), np.nan, dtype=np.float32)
    result[:, 2] = 0.0
    coco = np.column_stack((xy, confidence)).astype(np.float32)
    for source, destination in COCO_TO_BODY25.items():
        result[destination] = coco[source]
    result[1] = midpoint(coco, 5, 6)   # neck
    result[8] = midpoint(coco, 11, 12) # mid hip
    result[result[:, 2] <= 0, :2] = np.nan
    return result


def select_pose(result: object) -> np.ndarray:
    missing = np.full((25, 3), np.nan, dtype=np.float32)
    missing[:, 2] = 0.0
    if result.keypoints is None or len(result.keypoints) == 0:
        return missing
    xy = result.keypoints.xy.cpu().numpy()
    confidence = result.keypoints.conf.cpu().numpy()
    selected = int(np.argmax(np.mean(confidence, axis=1)))
    return coco_to_body25(xy[selected], confidence[selected])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO pose and rule-based fall detection.")
    parser.add_argument("--source", default="0", help="Camera number or video/image path")
    parser.add_argument("--model", type=Path, default=project_root() / "openpose/models/yolo11n-pose.pt")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 1, ... (default: auto)")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=15.0)
    parser.add_argument("--output", type=Path, help="Optional annotated MP4 output")
    parser.add_argument("--headless", action="store_true", help="Do not open a GUI window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means unlimited")
    parser.add_argument("--loop", action="store_true", help="Loop a video file like a virtual webcam")
    parser.add_argument("--realtime", action="store_true", help="Pace video-file inference at its recorded FPS")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = parse_source(args.source)
    device = resolve_device(args.device)
    if isinstance(source, str) and not Path(source).expanduser().is_file():
        raise FileNotFoundError(f"Input not found: {source}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Pose model not found: {args.model}")

    capture = cv2.VideoCapture(source)
    if isinstance(source, int):
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fallback_fps = source_fps if source_fps > 1 else 10.0
    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fallback_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {args.output}")

    model = YOLO(str(args.model))
    poses: deque[np.ndarray] = deque(maxlen=max(60, round(fallback_fps * 15)))
    timestamps: deque[float] = deque(maxlen=60)
    state = FallState.UNKNOWN
    processed = 0
    started = time.perf_counter()
    next_frame_at = started
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if args.loop and isinstance(source, str):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    poses.clear()
                    timestamps.clear()
                    state = FallState.UNKNOWN
                    next_frame_at = time.perf_counter()
                    continue
                break
            if args.realtime and isinstance(source, str):
                delay = next_frame_at - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                next_frame_at += 1.0 / fallback_fps
            prediction = model.predict(frame, imgsz=args.image_size, conf=args.confidence, device=device, verbose=False)[0]
            pose = select_pose(prediction)
            poses.append(pose)
            timestamps.append(time.perf_counter())
            fps = fallback_fps
            if isinstance(source, int) and len(timestamps) >= 5:
                fps = float(np.clip(1.0 / np.median(np.diff(timestamps)), 1.0, 60.0))
            if len(poses) >= max(5, round(fps * 2)):
                features = extract_features(np.stack(poses), fps, width, height)
                state = FallState(int(detect_fall(features).states[-1]))

            annotated = prediction.plot()
            color = (0, 0, 255) if state == FallState.FALLEN else (0, 220, 0)
            cv2.rectangle(annotated, (10, 10), (390, 64), (0, 0, 0), -1)
            cv2.putText(annotated, f"STATE: {state.name}  FPS: {fps:.1f}", (20, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if writer:
                writer.write(annotated)
            if not args.headless:
                cv2.imshow("YOLO Pose Fall Detection", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        capture.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print(
        f"source={source} device={device} torch={torch.__version__} "
        f"cuda={torch.version.cuda} frames={processed} elapsed={elapsed:.2f}s "
        f"average_fps={processed / elapsed:.2f} final_state={state.name}"
    )
    if args.output:
        print(f"output={args.output.resolve()}")
    if processed == 0:
        raise RuntimeError(f"No frames read from source: {source}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)
