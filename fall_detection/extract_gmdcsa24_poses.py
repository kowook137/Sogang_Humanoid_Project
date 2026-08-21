"""Extract BODY_25-compatible pose sequences from every GMDCSA-24 video."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from yolo_pose import project_root, resolve_device, select_pose


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def extract_video(
    model: YOLO,
    video_path: Path,
    device: str,
    image_size: int,
    confidence: float,
    batch_size: int,
) -> tuple[np.ndarray, float, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    poses: list[np.ndarray] = []

    def infer_batch() -> None:
        if not frames:
            return
        results = model.predict(
            frames,
            imgsz=image_size,
            conf=confidence,
            device=device,
            verbose=False,
        )
        poses.extend(select_pose(result) for result in results)
        frames.clear()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            if len(frames) >= batch_size:
                infer_batch()
        infer_batch()
    finally:
        capture.release()
    if not poses:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    return np.stack(poses).astype(np.float32), fps, width, height


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "data/datasets/gmdcsa24/manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "data/datasets/gmdcsa24/poses")
    parser.add_argument("--model", type=Path, default=root / "openpose/models/yolo11n-pose.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    rows = load_manifest(args.manifest.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model.resolve()))
    completed = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        destination = args.output_dir / f"{row['video_id']}.npz"
        if destination.is_file() and not args.overwrite:
            skipped += 1
            print(f"[{index:03d}/{len(rows)}] skip {row['video_id']}", flush=True)
            continue
        poses, fps, width, height = extract_video(
            model,
            Path(row["pipeline_path"]),
            device,
            args.image_size,
            args.confidence,
            args.batch_size,
        )
        np.savez_compressed(
            destination,
            keypoints=poses,
            fps=np.float32(fps),
            frame_width=np.int32(width),
            frame_height=np.int32(height),
            video_id=row["video_id"],
            subject=row["subject"],
            label=row["label"],
            split=row["split"],
        )
        completed += 1
        detected = float(np.mean(np.any(poses[:, :, 2] > 0, axis=1)))
        print(
            f"[{index:03d}/{len(rows)}] {row['video_id']} frames={len(poses)} "
            f"person_frames={detected:.1%}",
            flush=True,
        )
    print(f"done device={device} extracted={completed} skipped={skipped} total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
