"""Convert FallVision COCO keypoint CSV files to BODY_25-compatible NPZ sequences."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from yolo_pose import midpoint, project_root


NAME_TO_BODY25 = {
    "Nose": 0,
    "Right Shoulder": 2,
    "Right Elbow": 3,
    "Right Wrist": 4,
    "Left Shoulder": 5,
    "Left Elbow": 6,
    "Left Wrist": 7,
    "Right Hip": 9,
    "Right Knee": 10,
    "Right Ankle": 11,
    "Left Hip": 12,
    "Left Knee": 13,
    "Left Ankle": 14,
    "Right Eye": 15,
    "Left Eye": 16,
    "Right Ear": 17,
    "Left Ear": 18,
}
ARCHIVE_PATTERN = re.compile(r"^(nf|f)_mask_([bcs])_(\d)_(?:keypoints|ketpoints)_csv$")
CATEGORY_NAMES = {"b": "bed", "c": "chair", "s": "stand"}


def convert_csv(source: Path) -> np.ndarray:
    values: list[tuple[int, str, float, float, float]] = []
    max_frame = 0
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            frame = int(row["Frame"])
            max_frame = max(max_frame, frame)
            values.append(
                (frame - 1, row["Keypoint"].strip(), float(row["X"]), float(row["Y"]), float(row["Confidence"]))
            )
    keypoints = np.full((max_frame, 25, 3), np.nan, dtype=np.float32)
    keypoints[:, :, 2] = 0.0
    for frame, name, x, y, confidence in values:
        destination = NAME_TO_BODY25.get(name)
        if destination is not None:
            keypoints[frame, destination] = (x, y, confidence)
    for frame in range(max_frame):
        coco_like = keypoints[frame]
        keypoints[frame, 1] = midpoint(coco_like, 5, 2)
        keypoints[frame, 8] = midpoint(coco_like, 12, 9)
    keypoints[keypoints[:, :, 2] <= 0, :2] = np.nan
    return keypoints


def main() -> int:
    root = project_root() / "data/datasets/fallvision"
    source_root = root / "keypoints"
    output_dir = root / "poses"
    manifest_path = root / "manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    csv_files = sorted(source_root.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(
            f"No FallVision CSV files found under {source_root}; "
            "run download_fallvision_keypoints.py first"
        )
    for index, source in enumerate(csv_files, start=1):
        match = ARCHIVE_PATTERN.match(source.parent.name)
        if not match:
            raise RuntimeError(f"Unexpected FallVision directory: {source.parent.name}")
        label_code, category_code, archive = match.groups()
        label = "normal" if label_code == "nf" else "fall"
        category = CATEGORY_NAMES[category_code]
        video_id = f"fv_{label}_{category}_a{archive}_{source.stem.lower()}"
        destination = output_dir / f"{video_id}.npz"
        if not destination.is_file():
            keypoints = convert_csv(source)
            np.savez_compressed(
                destination,
                keypoints=keypoints,
                frame_width=np.int32(0),
                frame_height=np.int32(0),
                video_id=video_id,
                label=label,
                dataset="fallvision",
            )
        manifest.append(
            {
                "video_id": video_id,
                "dataset": "fallvision",
                "label": label,
                "category": category,
                "archive": archive,
                "split": "train",
                "pose_path": str(destination.resolve()),
                "source_path": str(source.resolve()),
            }
        )
        if index % 250 == 0 or index == len(csv_files):
            print(f"converted={index}/{len(csv_files)}", flush=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"manifest={manifest_path} videos={len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
