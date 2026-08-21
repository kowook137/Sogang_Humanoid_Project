"""Select an ensemble on validation subjects and evaluate once on the held-out subject."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_gmdcsa24 import FallGRU, engineered_features, metrics, pose_features, resample


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def legacy_features(path: Path) -> np.ndarray:
    with np.load(path) as data:
        keypoints = data["keypoints"].astype(np.float32)
        width = max(float(data["frame_width"]), 1.0)
        height = max(float(data["frame_height"]), 1.0)
    confidence = keypoints[:, :, 2]
    valid = np.isfinite(keypoints[:, :, :2]).all(axis=2) & (confidence > 0)
    x = np.where(valid, keypoints[:, :, 0] / width, 0.0)
    y = np.where(valid, keypoints[:, :, 1] / height, 0.0)
    confidence = np.where(valid, confidence, 0.0)
    return np.concatenate((x, y, confidence), axis=1).astype(np.float32)


def load_model(path: Path, device: torch.device) -> tuple[FallGRU, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = FallGRU(input_size=int(checkpoint["input_size"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def probabilities(
    rows: list[dict[str, str]],
    pose_dir: Path,
    model: FallGRU,
    checkpoint: dict[str, object],
    feature_kind: str,
    device: torch.device,
) -> np.ndarray:
    feature_function = {
        "legacy": legacy_features,
        "pose": pose_features,
        "engineered": engineered_features,
    }[feature_kind]
    result = []
    for row in rows:
        sequence = feature_function(pose_dir / f"{row['video_id']}.npz")
        inputs = torch.from_numpy(resample(sequence, int(checkpoint["frames"]))).unsqueeze(0).to(device)
        result.append(float(torch.softmax(model(inputs), dim=1)[0, 1]))
    return np.asarray(result)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "data/datasets/gmdcsa24/manifest.csv")
    parser.add_argument("--pose-dir", type=Path, default=root / "data/datasets/gmdcsa24/poses")
    parser.add_argument("--output", type=Path, default=root / "outputs/ensemble_evaluation.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but no CUDA GPU is available")
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    splits = {name: [row for row in rows if row["split"] == name] for name in ("validation", "test")}
    specifications = [
        ("legacy", project_root() / "outputs/gmdcsa24_training/best_model.pt"),
        ("pose", project_root() / "outputs/combined_training/best_model.pt"),
        ("engineered", project_root() / "outputs/engineered_combined_training/best_model.pt"),
    ]
    missing_checkpoints = [str(path) for _, path in specifications if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(f"Missing ensemble checkpoint: {missing_checkpoints[0]}")
    scores: dict[str, dict[str, np.ndarray]] = {split: {} for split in splits}
    for feature_kind, checkpoint_path in specifications:
        model, checkpoint = load_model(checkpoint_path, device)
        for split, split_rows in splits.items():
            scores[split][feature_kind] = probabilities(
                split_rows, args.pose_dir, model, checkpoint, feature_kind, device
            )
    validation_labels = [1 if row["label"] == "fall" else 0 for row in splits["validation"]]
    best: tuple[tuple[float, float, float], float, dict[str, float | int]] | None = None
    for legacy_weight_int in range(11):
        for pose_weight_int in range(11 - legacy_weight_int):
            engineered_weight_int = 10 - legacy_weight_int - pose_weight_int
            weights = np.asarray((legacy_weight_int, pose_weight_int, engineered_weight_int)) / 10.0
            combined = sum(
                weights[index] * scores["validation"][name]
                for index, name in enumerate(("legacy", "pose", "engineered"))
            )
            for threshold in np.arange(0.25, 0.751, 0.025):
                validation_predictions = (combined >= threshold).astype(int).tolist()
                result = metrics(validation_labels, validation_predictions)
                rank = (float(result["f1"]), float(result["accuracy"]), float(result["recall"]))
                if best is None or rank > (
                    float(best[2]["f1"]), float(best[2]["accuracy"]), float(best[2]["recall"])
                ):
                    best = (tuple(float(value) for value in weights), float(threshold), result)
    assert best is not None
    weights, threshold, validation_result = best
    test_combined = sum(
        weights[index] * scores["test"][name]
        for index, name in enumerate(("legacy", "pose", "engineered"))
    )
    test_labels = [1 if row["label"] == "fall" else 0 for row in splits["test"]]
    test_predictions = (test_combined >= threshold).astype(int).tolist()
    report = {
        "selection_split": "validation",
        "weights": dict(zip(("legacy", "pose", "engineered"), weights)),
        "threshold": threshold,
        "validation": validation_result,
        "test": metrics(test_labels, test_predictions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
