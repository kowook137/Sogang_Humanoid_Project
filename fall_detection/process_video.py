"""Run OpenPose on a video and convert BODY_25 JSON output to NPZ.

The script intentionally keeps OpenPose execution separate from fall detection.
It produces a rendered video, the original per-frame OpenPose JSON files, a
time-aligned NumPy array, and processing metadata.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from openpose_runtime import OpenPoseRuntime, find_openpose


BODY_25_COUNT = 25
BODY_25_NAMES = (
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def relative_display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orientation_degrees = int(capture.get(cv2.CAP_PROP_ORIENTATION_META))
    capture.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "orientation_degrees": orientation_degrees,
        "duration_seconds": frame_count / fps if fps > 0 else None,
    }


def normalize_video_orientation(
    source: Path,
    destination: Path,
    metadata: dict[str, Any],
) -> tuple[Path, bool]:
    """Bake display rotation into pixels because OpenPose ignores MP4 rotation metadata."""
    if metadata["orientation_degrees"] % 360 == 0:
        return source, False

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video for orientation normalization: {source}")

    # OpenCV applies CAP_PROP_ORIENTATION_META automatically while decoding.
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata["fps"],
        (metadata["width"], metadata["height"]),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create normalized input video: {destination}")

    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    capture.release()
    writer.release()
    if written == 0:
        raise RuntimeError(f"Input video contained no readable frames: {source}")
    return destination, True


def ensure_empty_output(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    json_dir = output_dir / "keypoints_json"
    rendered_avi = output_dir / "rendered.avi"
    rendered_video = output_dir / "rendered.mp4"
    npz_path = output_dir / "keypoints.npz"

    generated_paths = (
        rendered_avi,
        rendered_video,
        npz_path,
        output_dir / "metadata.json",
    )
    if any(path.exists() for path in generated_paths) or (
        json_dir.exists() and any(json_dir.iterdir())
    ):
        raise FileExistsError(
            f"Output already exists in {output_dir}. "
            "Choose another --output-dir to preserve previous results."
        )

    json_dir.mkdir(parents=True, exist_ok=True)
    return json_dir, rendered_avi, rendered_video, npz_path


def run_openpose(
    input_path: Path,
    runtime: OpenPoseRuntime,
    json_dir: Path,
    rendered_video: Path,
    net_resolution: str,
) -> tuple[list[str], float]:
    command = [
        str(runtime.executable),
        "--video",
        str(input_path),
        "--model_folder",
        str(runtime.model_dir),
        "--model_pose",
        "BODY_25",
        "--number_people_max",
        "1",
        "--net_resolution",
        net_resolution,
        "--write_json",
        str(json_dir),
        "--write_video",
        str(rendered_video),
        "--display",
        "0",
    ]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=runtime.working_dir,
        check=False,
        text=True,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"OpenPose failed with exit code {completed.returncode}")

    return command, elapsed


def convert_avi_to_mp4(source: Path, destination: Path, fallback_fps: float) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open rendered AVI: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = fallback_fps

    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create rendered MP4: {destination}")

    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    capture.release()
    writer.release()
    if written == 0:
        raise RuntimeError(f"Rendered AVI contained no readable frames: {source}")

    source.unlink()


def select_person(people: list[dict[str, Any]]) -> np.ndarray:
    missing = np.full((BODY_25_COUNT, 3), np.nan, dtype=np.float32)
    missing[:, 2] = 0.0
    if not people:
        return missing

    candidates: list[np.ndarray] = []
    for person in people:
        values = person.get("pose_keypoints_2d", [])
        if len(values) != BODY_25_COUNT * 3:
            continue
        candidates.append(np.asarray(values, dtype=np.float32).reshape(BODY_25_COUNT, 3))

    if not candidates:
        return missing

    # This also behaves sensibly if number_people_max is changed in the future.
    selected = max(candidates, key=lambda item: float(np.mean(item[:, 2])))
    invalid = selected[:, 2] <= 0
    selected[invalid, :2] = np.nan
    return selected


def convert_json(json_dir: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    json_files = sorted(json_dir.glob("*_keypoints.json"))
    if not json_files:
        raise RuntimeError(f"OpenPose generated no JSON files in {json_dir}")

    frames: list[np.ndarray] = []
    detected: list[bool] = []
    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        frame = select_person(payload.get("people", []))
        frames.append(frame)
        detected.append(bool(np.any(frame[:, 2] > 0)))

    keypoints = np.stack(frames, axis=0)
    confidence = keypoints[:, :, 2]
    stats = {
        "json_frame_count": len(json_files),
        "detected_frame_count": int(np.count_nonzero(detected)),
        "missing_frame_count": int(len(detected) - np.count_nonzero(detected)),
        "detected_frame_ratio": float(np.mean(detected)),
        "mean_joint_confidence": {
            name: float(np.mean(confidence[:, index]))
            for index, name in enumerate(BODY_25_NAMES)
        },
        "joint_valid_ratio_at_0_3": {
            name: float(np.mean(confidence[:, index] >= 0.3))
            for index, name in enumerate(BODY_25_NAMES)
        },
    }
    return keypoints, [path.name for path in json_files], stats


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Generate rendered OpenPose video, BODY_25 JSON, and NPZ output."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input MP4/AVI path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: outputs/<input-stem>)",
    )
    parser.add_argument(
        "--openpose-root",
        type=Path,
        help="OpenPose source/build or portable root (default: auto-detect)",
    )
    parser.add_argument(
        "--net-resolution",
        default="-1x256",
        help="OpenPose network resolution; height must be a multiple of 16",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    input_path = args.input.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "outputs" / input_path.stem
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    runtime = find_openpose(root, args.openpose_root)

    source_metadata = video_metadata(input_path)
    json_dir, rendered_avi, rendered_video, npz_path = ensure_empty_output(output_dir)
    normalized_input = output_dir / "_normalized_input.mp4"
    processing_input, remove_processing_input = normalize_video_orientation(
        input_path,
        normalized_input,
        source_metadata,
    )
    try:
        command, processing_seconds = run_openpose(
            input_path=processing_input,
            runtime=runtime,
            json_dir=json_dir,
            rendered_video=rendered_avi,
            net_resolution=args.net_resolution,
        )
        convert_avi_to_mp4(rendered_avi, rendered_video, source_metadata["fps"])
    finally:
        if remove_processing_input and normalized_input.exists():
            normalized_input.unlink()
    keypoints, frame_files, keypoint_stats = convert_json(json_dir)

    np.savez_compressed(
        npz_path,
        keypoints=keypoints,
        frame_files=np.asarray(frame_files),
        joint_names=np.asarray(BODY_25_NAMES),
        fps=np.float32(source_metadata["fps"]),
        frame_width=np.int32(source_metadata["width"]),
        frame_height=np.int32(source_metadata["height"]),
    )

    metadata = {
        "format_version": 1,
        "input": relative_display(input_path, root),
        "output_directory": relative_display(output_dir, root),
        "rendered_video": "rendered.mp4",
        "keypoints_json_directory": "keypoints_json",
        "keypoints_npz": "keypoints.npz",
        "pose_model": "BODY_25",
        "keypoint_shape": list(keypoints.shape),
        "coordinate_order": ["x", "y", "confidence"],
        "coordinate_system": "image pixels; origin top-left; +x right; +y down",
        "missing_value_policy": "x/y=NaN and confidence=0",
        "net_resolution": args.net_resolution,
        "orientation_normalized": remove_processing_input,
        "source_video": source_metadata,
        "keypoint_statistics": keypoint_stats,
        "processing_seconds": processing_seconds,
        "openpose_command": command,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    if source_metadata["frame_count"] != keypoints.shape[0]:
        print(
            "Warning: input frame count and JSON frame count differ: "
            f"{source_metadata['frame_count']} != {keypoints.shape[0]}",
            file=sys.stderr,
        )

    print(f"Processed: {relative_display(input_path, root)}")
    print(f"Output: {relative_display(output_dir, root)}")
    print(f"Keypoints shape: {keypoints.shape}")
    print(
        "Detected frames: "
        f"{keypoint_stats['detected_frame_count']}/{keypoint_stats['json_frame_count']} "
        f"({keypoint_stats['detected_frame_ratio']:.1%})"
    )
    print(f"Processing time: {processing_seconds:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
