"""Select a two-model FallVision ensemble on validation and evaluate test once."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_fall_ensemble import load_model, probabilities
from train_gmdcsa24 import metrics


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=root / "data/datasets/fallvision/manifest_split.csv"
    )
    parser.add_argument("--pose-dir", type=Path, default=root / "data/datasets/fallvision/poses")
    parser.add_argument(
        "--pose-checkpoint", type=Path,
        default=root / "outputs/fallvision_pose_training/best_model.pt",
    )
    parser.add_argument(
        "--engineered-checkpoint", type=Path,
        default=root / "outputs/fallvision_engineered_training/best_model.pt",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "outputs/fallvision_ensemble_evaluation.json"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-validation-recall", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    usable_rows = []
    for row in rows:
        with np.load(args.pose_dir / f"{row['video_id']}.npz") as data:
            if len(data["keypoints"]) >= 15:
                usable_rows.append(row)
    splits = {
        name: [row for row in usable_rows if row["split"] == name]
        for name in ("validation", "test")
    }
    scores: dict[str, dict[str, np.ndarray]] = {name: {} for name in splits}
    for kind, checkpoint_path in (
        ("pose", args.pose_checkpoint), ("engineered", args.engineered_checkpoint)
    ):
        model, checkpoint = load_model(checkpoint_path, device)
        for split, split_rows in splits.items():
            scores[split][kind] = probabilities(
                split_rows, args.pose_dir, model, checkpoint, kind, device
            )
    validation_labels = [int(row["label"] == "fall") for row in splits["validation"]]
    best = None
    for pose_weight in np.arange(0.0, 1.01, 0.05):
        combined = pose_weight * scores["validation"]["pose"] + (
            1.0 - pose_weight
        ) * scores["validation"]["engineered"]
        for threshold in np.arange(0.25, 0.751, 0.025):
            result = metrics(validation_labels, (combined >= threshold).astype(int).tolist())
            recall = float(result["recall"])
            rank = (
                int(recall >= args.min_validation_recall),
                float(result["precision"]) if recall >= args.min_validation_recall else recall,
                float(result["f1"]),
            )
            if best is None or rank > best[0]:
                best = (rank, float(pose_weight), float(threshold), result)
    assert best is not None
    _, pose_weight, threshold, validation_result = best
    test_scores = pose_weight * scores["test"]["pose"] + (
        1.0 - pose_weight
    ) * scores["test"]["engineered"]
    test_labels = [int(row["label"] == "fall") for row in splits["test"]]
    report = {
        "selection_split": "validation",
        "pose_weight": pose_weight,
        "engineered_weight": 1.0 - pose_weight,
        "threshold": threshold,
        "selection_target_min_recall": args.min_validation_recall,
        "validation": validation_result,
        "test": metrics(test_labels, (test_scores >= threshold).astype(int).tolist()),
        "limitation": "clip-level split; participant IDs unavailable",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
