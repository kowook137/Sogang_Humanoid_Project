"""Evaluate the real-time rule detector on a pose-sequence manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from detector import DetectorConfig, detect_fall
from features import extract_features
from train_gmdcsa24 import metrics


def evaluate_row(row: dict[str, str], fps: float, config: DetectorConfig) -> tuple[int, int]:
    with np.load(row["pose_path"]) as data:
        keypoints = data["keypoints"].astype(np.float32)
        width = int(data.get("frame_width", 0))
        height = int(data.get("frame_height", 0))
    valid_x = keypoints[:, :, 0][np.isfinite(keypoints[:, :, 0])]
    valid_y = keypoints[:, :, 1][np.isfinite(keypoints[:, :, 1])]
    width = width or (max(1, int(np.max(valid_x)) + 1) if valid_x.size else 640)
    height = height or (max(1, int(np.max(valid_y)) + 1) if valid_y.size else 480)
    prediction = int(detect_fall(extract_features(keypoints, fps, width, height), config).fall_detected)
    label = int(row["label"] == "fall")
    return label, prediction


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=root / "data/datasets/fallvision/manifest.csv"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    defaults = DetectorConfig()
    parser.add_argument("--min-downward-speed", type=float, default=defaults.min_downward_speed)
    parser.add_argument("--min-drop", type=float, default=defaults.min_drop)
    parser.add_argument("--min-torso-angle", type=float, default=defaults.min_torso_angle)
    parser.add_argument("--min-bbox-aspect", type=float, default=defaults.min_bbox_aspect)
    parser.add_argument("--min-low-hip", type=float, default=defaults.min_low_hip)
    parser.add_argument(
        "--output", type=Path, default=root / "outputs/fallvision_rule_evaluation.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    config = DetectorConfig(
        min_downward_speed=args.min_downward_speed,
        min_drop=args.min_drop,
        min_torso_angle=args.min_torso_angle,
        min_bbox_aspect=args.min_bbox_aspect,
        min_low_hip=args.min_low_hip,
    )
    labels: list[int] = []
    predictions: list[int] = []
    groups: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    errors: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        label, prediction = evaluate_row(row, args.fps, config)
        labels.append(label)
        predictions.append(prediction)
        group_labels, group_predictions = groups[row.get("category", "unknown")]
        group_labels.append(label)
        group_predictions.append(prediction)
        if label != prediction and len(errors) < 100:
            errors.append(
                {"video_id": row["video_id"], "label": row["label"], "category": row.get("category", "")}
            )
        if index % 250 == 0 or index == len(rows):
            print(f"evaluated={index}/{len(rows)}", flush=True)
    report = {
        "manifest": str(args.manifest.resolve()),
        "assumed_fps": args.fps,
        "config": config.__dict__,
        "overall": metrics(labels, predictions),
        "by_category": {
            name: metrics(group_labels, group_predictions)
            for name, (group_labels, group_predictions) in sorted(groups.items())
        },
        "first_errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "first_errors"}, indent=2))
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
