"""Run YOLO pose on a video or webcam and feed the existing fall detector."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

from detector import DetectorConfig, FallState
from streaming import StreamingFallDetector


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


# Frames used to measure the real processing rate before the feature windows
# are sized. A camera's advertised FPS is rarely what the pipeline achieves.
FPS_PROBE_FRAMES = 15


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def measured_fps(timestamps: deque[float], fallback: float) -> float:
    if len(timestamps) < 5:
        return fallback
    intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
    intervals = intervals[(intervals > 0.001) & (intervals < 2.0)]
    if not intervals.size:
        return fallback
    return float(np.clip(1.0 / np.median(intervals), 1.0, 60.0))


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
    spans = np.ptp(xy, axis=1)
    areas = np.maximum(spans[:, 0], 1.0) * np.maximum(spans[:, 1], 1.0)
    scores = areas * np.mean(confidence, axis=1) * np.mean(confidence > 0.2, axis=1)
    selected = int(np.argmax(scores))
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
    parser.add_argument(
        "--camera-fourcc",
        default="MJPG",
        help="FourCC requested for camera input, for example MJPG or YUYV",
    )
    parser.add_argument(
        "--preserve-camera-settings",
        action="store_true",
        help="Use the V4L2 format already configured on the camera without resetting it",
    )
    parser.add_argument(
        "--ffmpeg-camera",
        action="store_true",
        help="Read a Linux camera through FFmpeg (useful for damaged MJPEG over WSL USB/IP)",
    )
    parser.add_argument("--output", type=Path, help="Optional annotated MP4 output")
    parser.add_argument("--headless", action="store_true", help="Do not open a GUI window")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means unlimited")
    parser.add_argument("--loop", action="store_true", help="Loop a video file like a virtual webcam")
    parser.add_argument("--realtime", action="store_true", help="Pace video-file inference at its recorded FPS")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip skeleton drawing (recommended for headless Jetson deployment)",
    )
    parser.add_argument(
        "--auto-clear-seconds",
        type=float,
        default=0.0,
        help="Clear a FALLEN latch after the person is upright this long; 0 keeps it latched",
    )
    parser.add_argument(
        "--fall-hold-seconds",
        type=float,
        default=30.0,
        help="Clear a FALLEN latch this long after the last confirming frame; 0 latches until restart",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        help="Atomically update this JSON file with the current fall-detection status",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=1.0,
        help="Write/print status at least this often; 0 only reports state changes",
    )
    defaults = DetectorConfig()
    parser.add_argument("--min-downward-speed", type=float, default=defaults.min_downward_speed)
    parser.add_argument("--min-drop", type=float, default=defaults.min_drop)
    parser.add_argument("--min-torso-angle", type=float, default=defaults.min_torso_angle)
    parser.add_argument("--min-bbox-aspect", type=float, default=defaults.min_bbox_aspect)
    parser.add_argument("--min-low-hip", type=float, default=defaults.min_low_hip)
    parser.add_argument("--classifier", type=Path, help="Optional engineered-feature GRU checkpoint")
    parser.add_argument("--pose-classifier", type=Path, help="Optional raw-pose GRU checkpoint")
    parser.add_argument("--classifier-device", default="cpu", help="PyTorch device for the small GRU")
    parser.add_argument("--classifier-threshold", type=float, default=0.80)
    parser.add_argument("--classifier-pose-weight", type=float, default=0.45)
    parser.add_argument("--classifier-window-seconds", type=float, default=4.0)
    parser.add_argument("--classifier-confirm-seconds", type=float, default=0.5)
    parser.add_argument("--classifier-interval", type=int, default=3, help="Run GRU every N pose frames")
    return parser.parse_args()


def publish_status(
    path: Path | None,
    state: FallState,
    frame: int,
    fps: float,
    inference_ms: float,
    reason: str,
    fall_probability: float | None = None,
    latched: bool = False,
    calibrating: bool = False,
) -> None:
    payload = {
        "schema": 1,
        "timestamp": time.time(),
        "state": state.name,
        "fall_detected": state == FallState.FALLEN,
        "fall_latched": latched,
        "calibrating": calibrating,
        "frame": frame,
        "fps": round(fps, 3),
        "inference_ms": round(inference_ms, 3),
        "reason": reason,
        "fall_probability": round(fall_probability, 4) if fall_probability is not None else None,
    }
    print("FALL_STATUS " + json.dumps(payload, separators=(",", ":")), flush=True)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.heartbeat_seconds < 0 or args.fall_hold_seconds < 0:
        raise ValueError("--heartbeat-seconds and --fall-hold-seconds cannot be negative")
    if not 0 < args.classifier_threshold < 1:
        raise ValueError("--classifier-threshold must be between 0 and 1")
    if not 0 <= args.classifier_pose_weight <= 1:
        raise ValueError("--classifier-pose-weight must be between 0 and 1")
    if args.classifier_window_seconds <= 0 or args.classifier_confirm_seconds <= 0:
        raise ValueError("classifier window and confirmation seconds must be positive")
    if args.classifier_interval < 1:
        raise ValueError("--classifier-interval must be at least 1")
    detector_config = DetectorConfig(
        min_downward_speed=args.min_downward_speed,
        min_drop=args.min_drop,
        min_torso_angle=args.min_torso_angle,
        min_bbox_aspect=args.min_bbox_aspect,
        min_low_hip=args.min_low_hip,
    )
    classifier = None
    if args.classifier:
        if not args.classifier.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {args.classifier}")
        from realtime_classifier import RealtimeFallClassifier

        classifier = RealtimeFallClassifier(args.classifier, args.classifier_device)
    pose_classifier = None
    if args.pose_classifier:
        if not args.pose_classifier.is_file():
            raise FileNotFoundError(f"Pose classifier checkpoint not found: {args.pose_classifier}")
        from realtime_classifier import RealtimeFallClassifier

        pose_classifier = RealtimeFallClassifier(args.pose_classifier, args.classifier_device)
    classifier_enabled = classifier is not None or pose_classifier is not None
    source = parse_source(args.source)
    device = resolve_device(args.device)
    if isinstance(source, str) and not Path(source).expanduser().is_file():
        raise FileNotFoundError(f"Input not found: {source}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Pose model not found: {args.model}")

    ffmpeg_process: subprocess.Popen[bytes] | None = None
    capture: cv2.VideoCapture | None = None
    if isinstance(source, int) and args.ffmpeg_camera:
        input_format = "mjpeg" if args.camera_fourcc.upper() == "MJPG" else "yuyv422"
        ffmpeg_process = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "v4l2", "-input_format", input_format,
                "-video_size", f"{args.camera_width}x{args.camera_height}",
                "-framerate", str(args.camera_fps),
                "-i", f"/dev/video{source}",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    else:
        capture = cv2.VideoCapture(source)
    if capture is not None and isinstance(source, int) and not args.preserve_camera_settings:
        if len(args.camera_fourcc) != 4:
            raise ValueError("--camera-fourcc must contain exactly four characters")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.camera_fourcc.upper()))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if capture is not None and not capture.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {source}")
    width = args.camera_width if ffmpeg_process else int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = args.camera_height if ffmpeg_process else int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    source_fps = args.camera_fps if ffmpeg_process else float(capture.get(cv2.CAP_PROP_FPS))
    fallback_fps = source_fps if source_fps > 1 else 10.0
    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fallback_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {args.output}")

    model = YOLO(str(args.model))
    timestamps: deque[float] = deque(maxlen=60)
    # A video file reports its own rate; a camera has to be measured, so the
    # first frames are buffered and replayed once the real rate is known.
    detector: StreamingFallDetector | None = None
    if isinstance(source, str):
        detector = StreamingFallDetector(
            fps=fallback_fps,
            frame_height=height,
            config=detector_config,
            auto_clear_seconds=args.auto_clear_seconds,
            fall_hold_seconds=args.fall_hold_seconds,
        )
    pending: list[np.ndarray] = []
    state = FallState.UNKNOWN
    latched = False
    calibrating = True
    fps = fallback_fps
    published_state: FallState | None = None
    last_published_at = float("-inf")
    fall_probability: float | None = None
    classifier_candidate_count = 0
    # The rule layer runs incrementally and keeps no history, but the GRU reads
    # a raw pose window, so only pay for the buffer when a checkpoint is loaded.
    poses: deque[np.ndarray] = deque(
        maxlen=max(60, round(fallback_fps * 15)) if classifier_enabled else 0
    )
    processed = 0
    inference_times: list[float] = []
    started = time.perf_counter()
    next_frame_at = started
    try:
        while True:
            if ffmpeg_process:
                assert ffmpeg_process.stdout is not None
                expected_bytes = width * height * 3
                data = bytearray()
                while len(data) < expected_bytes:
                    chunk = ffmpeg_process.stdout.read(expected_bytes - len(data))
                    if not chunk:
                        break
                    data.extend(chunk)
                ok = len(data) == expected_bytes
                frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3) if ok else None
            else:
                assert capture is not None
                ok, frame = capture.read()
            if not ok:
                if capture is not None and args.loop and isinstance(source, str):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    timestamps.clear()
                    detector = StreamingFallDetector(
                        fps=fallback_fps,
                        frame_height=height,
                        config=detector_config,
                        auto_clear_seconds=args.auto_clear_seconds,
                        fall_hold_seconds=args.fall_hold_seconds,
                    )
                    state = FallState.UNKNOWN
                    latched = False
                    calibrating = True
                    classifier_candidate_count = 0
                    next_frame_at = time.perf_counter()
                    continue
                break
            if args.realtime and isinstance(source, str):
                delay = next_frame_at - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                next_frame_at += 1.0 / fallback_fps
            inference_started = time.perf_counter()
            prediction = model.predict(
                frame,
                imgsz=args.image_size,
                conf=args.confidence,
                device=device,
                verbose=False,
            )[0]
            inference_times.append(time.perf_counter() - inference_started)
            pose = select_pose(prediction)
            timestamps.append(time.perf_counter())
            poses.append(pose)
            if detector is None:
                pending.append(pose)
                if len(pending) >= FPS_PROBE_FRAMES:
                    fps = measured_fps(timestamps, fallback_fps)
                    detector = StreamingFallDetector(
                        fps=fps,
                        frame_height=height,
                        config=detector_config,
                        auto_clear_seconds=args.auto_clear_seconds,
                        fall_hold_seconds=args.fall_hold_seconds,
                    )
                    for buffered in pending:
                        update = detector.update(buffered)
                    pending.clear()
                    state, latched = update.state, update.fall_latched
                    calibrating = update.calibrating
            else:
                update = detector.update(pose)
                state, latched = update.state, update.fall_latched
                calibrating = update.calibrating
                fps = detector.fps

            # A learned second opinion on top of the rule layer. It only ever
            # raises the alarm, and only when the geometry agrees, so a GRU that
            # fires on an unusual-looking but upright pose cannot trip it alone.
            if (
                detector is not None
                and classifier_enabled
                and not calibrating
                and processed % args.classifier_interval == 0
            ):
                classifier_frames = max(15, round(fps * args.classifier_window_seconds))
                # Slice before stacking: the buffer holds far more than the window.
                classifier_poses = np.stack(list(poses)[-classifier_frames:])
                engineered_probability = (
                    classifier.probability(classifier_poses) if classifier is not None else None
                )
                pose_probability = (
                    pose_classifier.probability(classifier_poses)
                    if pose_classifier is not None
                    else None
                )
                if engineered_probability is not None and pose_probability is not None:
                    fall_probability = (
                        args.classifier_pose_weight * pose_probability
                        + (1.0 - args.classifier_pose_weight) * engineered_probability
                    )
                else:
                    fall_probability = (
                        engineered_probability
                        if engineered_probability is not None
                        else pose_probability
                    )
                posture_evidence = update.torso_angle >= 40.0 or update.bbox_aspect >= 0.70
                if fall_probability >= args.classifier_threshold and posture_evidence:
                    classifier_candidate_count += args.classifier_interval
                else:
                    classifier_candidate_count = max(
                        0, classifier_candidate_count - args.classifier_interval
                    )
                if classifier_candidate_count >= max(1, round(fps * args.classifier_confirm_seconds)):
                    # Go through the detector so its latch, and everything read
                    # back from it, stays the single source of truth.
                    detector.force_fallen()
                    state, latched = detector.state, detector.fall_latched

            now = time.perf_counter()
            state_changed = state != published_state
            heartbeat_due = args.heartbeat_seconds > 0 and now - last_published_at >= args.heartbeat_seconds
            if state_changed or heartbeat_due:
                publish_status(
                    args.status_file,
                    state,
                    processed,
                    fps,
                    inference_times[-1] * 1000.0,
                    "state_change" if state_changed else "heartbeat",
                    fall_probability,
                    latched=latched,
                    calibrating=calibrating,
                )
                published_state = state
                last_published_at = now

            should_render = not args.no_render and (writer is not None or not args.headless)
            if should_render:
                annotated = prediction.plot()
                color = (0, 0, 255) if latched else (0, 220, 0)
                label = "CALIBRATING" if calibrating else state.name
                cv2.rectangle(annotated, (10, 10), (430, 64), (0, 0, 0), -1)
                cv2.putText(
                    annotated,
                    f"{label}{'  [FALL]' if latched else ''}  FPS: {fps:.1f}",
                    (20, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                if writer:
                    writer.write(annotated)
            if not args.headless:
                display_frame = annotated if should_render else frame
                cv2.imshow("YOLO Pose Fall Detection", display_frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        if capture is not None:
            capture.release()
        if ffmpeg_process is not None:
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                ffmpeg_process.kill()
                ffmpeg_process.wait()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    inference_ms = np.asarray(inference_times, dtype=np.float64) * 1000.0
    warmup_ms = float(inference_ms[0]) if len(inference_ms) else float("nan")
    steady_ms = inference_ms[1:] if len(inference_ms) > 1 else inference_ms
    median_ms = float(np.median(steady_ms)) if len(steady_ms) else float("nan")
    p95_ms = float(np.percentile(steady_ms, 95)) if len(steady_ms) else float("nan")
    print(
        f"source={source} device={device} torch={torch.__version__} "
        f"cuda={torch.version.cuda} frames={processed} elapsed={elapsed:.2f}s "
        f"average_fps={processed / elapsed:.2f} warmup_ms={warmup_ms:.2f} "
        f"inference_median_ms={median_ms:.2f} "
        f"inference_p95_ms={p95_ms:.2f} final_state={state.name} "
        f"fall_latched={latched} "
        f"first_fallen_frame={detector.first_fallen_frame if detector else None}"
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
