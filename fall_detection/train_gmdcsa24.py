"""Train and evaluate a pose-sequence GRU on GMDCSA-24."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABELS = {"normal": 0, "fall": 1}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resample(sequence: np.ndarray, frames: int) -> np.ndarray:
    if frames < 1:
        raise ValueError("frames must be at least 1")
    if len(sequence) < 1:
        raise ValueError("cannot resample an empty sequence")
    if len(sequence) == frames:
        return sequence.astype(np.float32, copy=False)
    source = np.linspace(0.0, 1.0, len(sequence))
    target = np.linspace(0.0, 1.0, frames)
    output = np.empty((frames, sequence.shape[1]), dtype=np.float32)
    for feature in range(sequence.shape[1]):
        output[:, feature] = np.interp(target, source, sequence[:, feature])
    return output


def pose_features_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    keypoints = keypoints.astype(np.float32, copy=False)
    confidence = keypoints[:, :, 2]
    valid = np.isfinite(keypoints[:, :, :2]).all(axis=2) & (confidence > 0)
    xy = keypoints[:, :, :2].copy()
    timeline = np.arange(len(keypoints))
    for joint in range(25):
        for coordinate in range(2):
            known = valid[:, joint] & np.isfinite(xy[:, joint, coordinate])
            if np.any(known):
                xy[:, joint, coordinate] = np.interp(
                    timeline, timeline[known], xy[known, joint, coordinate]
                )
            else:
                xy[:, joint, coordinate] = 0.0
    torso_valid = valid[:, 1] & valid[:, 8]
    torso_lengths = np.linalg.norm(xy[:, 1] - xy[:, 8], axis=1)
    scale_values = torso_lengths[torso_valid & (torso_lengths > 1e-3)]
    if len(scale_values):
        scale = float(np.median(scale_values))
    else:
        spans = np.nanmax(np.where(valid[:, :, None], xy, np.nan), axis=1) - np.nanmin(
            np.where(valid[:, :, None], xy, np.nan), axis=1
        )
        finite_spans = spans[np.isfinite(spans).all(axis=1)]
        scale = float(np.median(np.max(finite_spans, axis=1))) if len(finite_spans) else 1.0
    early = max(1, round(len(keypoints) * 0.2))
    root_valid = valid[:early, 8]
    if np.any(root_valid):
        origin = np.median(xy[:early, 8][root_valid], axis=0)
    elif np.any(valid):
        origin = np.median(xy[valid], axis=0)
    else:
        origin = np.zeros(2, dtype=np.float32)
    normalized = np.clip((xy - origin) / max(scale, 1e-3), -8.0, 8.0)
    x = np.where(valid, normalized[:, :, 0], 0.0)
    y = np.where(valid, normalized[:, :, 1], 0.0)
    confidence = np.where(valid, confidence, 0.0)
    # Sequence-relative coordinates work across unknown resolutions while retaining
    # displacement relative to the person's initial pelvis position.
    return np.concatenate((x, y, confidence), axis=1).astype(np.float32)


def pose_features(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return pose_features_from_keypoints(data["keypoints"])


def engineered_features_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    pose = pose_features_from_keypoints(keypoints)
    if len(pose) < 2:
        return np.zeros((len(pose), 13), dtype=np.float32)
    x, y, confidence = pose[:, :25], pose[:, 25:50], pose[:, 50:75]
    valid = confidence > 0
    any_valid = np.any(valid, axis=1)
    bbox_width = np.max(np.where(valid, x, -np.inf), axis=1) - np.min(
        np.where(valid, x, np.inf), axis=1
    )
    bbox_height = np.max(np.where(valid, y, -np.inf), axis=1) - np.min(
        np.where(valid, y, np.inf), axis=1
    )
    bbox_width = np.where(any_valid, bbox_width, 0.0)
    bbox_height = np.where(any_valid, bbox_height, 0.0)
    aspect = bbox_width / np.maximum(bbox_height, 1e-3)
    coverage = np.mean(valid, axis=1)
    hip_x, hip_y = x[:, 8], y[:, 8]
    head_x, head_y = x[:, 0], y[:, 0]
    torso_x, torso_y = x[:, 1] - hip_x, y[:, 1] - hip_y
    hip_vx = np.gradient(hip_x)
    hip_vy = np.gradient(hip_y)
    head_vy = np.gradient(head_y)
    features = np.column_stack(
        (
            hip_x, hip_y, head_x, head_y, torso_x, torso_y,
            bbox_width, bbox_height, aspect, coverage,
            hip_vx, hip_vy, head_vy,
        )
    )
    return np.nan_to_num(features, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)


def engineered_features(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return engineered_features_from_keypoints(data["keypoints"])


class PoseDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        frames: int,
        augment: bool,
        feature_mode: str,
    ) -> None:
        self.samples: list[tuple[np.ndarray, int]] = []
        for row in rows:
            sequence = (
                engineered_features(Path(row["pose_path"]))
                if feature_mode == "engineered"
                else pose_features(Path(row["pose_path"]))
            )
            if len(sequence) < 15:
                continue
            self.samples.append((sequence, LABELS[row["label"]]))
        self.frames = frames
        self.augment = augment
        self.feature_mode = feature_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, label = self.samples[index]
        if self.augment:
            keep = random.uniform(0.85, 1.0)
            length = max(2, round(len(sequence) * keep))
            start = random.randint(0, len(sequence) - length)
            sequence = sequence[start : start + length].copy()
            if random.random() < 0.5:
                if self.feature_mode == "engineered":
                    sequence[:, [0, 2, 4, 10]] *= -1.0
                else:
                    sequence[:, :25] *= -1.0
            coordinate_count = 10 if self.feature_mode == "engineered" else 50
            sequence[:, :coordinate_count] += np.random.normal(
                0, 0.01, sequence[:, :coordinate_count].shape
            )
        sequence = resample(sequence, self.frames)
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)


class FallGRU(nn.Module):
    def __init__(self, input_size: int = 75, hidden_size: int = 96) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.25,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(inputs)
        return self.classifier(sequence.mean(dim=1))


def metrics(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "samples": len(labels), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float | int]]:
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += float(loss_function(logits, targets)) * len(targets)
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / len(labels), metrics(labels, predictions)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "data/datasets/gmdcsa24/manifest.csv")
    parser.add_argument("--pose-dir", type=Path, default=root / "data/datasets/gmdcsa24/poses")
    parser.add_argument(
        "--fallvision-manifest",
        type=Path,
        default=root / "data/datasets/fallvision/manifest.csv",
        help="Optional FallVision manifest; all entries are added to training only",
    )
    parser.add_argument("--no-fallvision", action="store_true", help="Train only on GMDCSA-24")
    parser.add_argument("--initial-checkpoint", type=Path, help="Initialize model weights from a prior checkpoint")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/gmdcsa24_training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-mode", choices=("pose", "engineered"), default="pose")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no CUDA GPU is available")
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["pose_path"] = str((args.pose_dir / f"{row['video_id']}.npz").resolve())
    missing = [row["video_id"] for row in rows if not (args.pose_dir / f"{row['video_id']}.npz").is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} pose files; first missing: {missing[0]}")
    grouped = {split: [row for row in rows if row["split"] == split] for split in ("train", "validation", "test")}
    if not args.no_fallvision and args.fallvision_manifest.is_file():
        with args.fallvision_manifest.open(encoding="utf-8-sig", newline="") as stream:
            fallvision_rows = list(csv.DictReader(stream))
        missing_fallvision = [row["video_id"] for row in fallvision_rows if not Path(row["pose_path"]).is_file()]
        if missing_fallvision:
            raise RuntimeError(
                f"Missing {len(missing_fallvision)} FallVision pose files; first missing: {missing_fallvision[0]}"
            )
        grouped["train"].extend(fallvision_rows)
    loaders = {
        split: DataLoader(
            PoseDataset(
                grouped[split], args.frames, augment=split == "train", feature_mode=args.feature_mode
            ),
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
        for split in grouped
    }
    empty_splits = [split for split, loader in loaders.items() if not loader.dataset]
    if empty_splits:
        raise RuntimeError(f"No usable pose sequences in split(s): {', '.join(empty_splits)}")
    input_size = 13 if args.feature_mode == "engineered" else 75
    model = FallGRU(input_size=input_size).to(device)
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(initial["model_state"])
        print(f"initialized_from={args.initial_checkpoint.resolve()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_model.pt"
    best_f1 = -1.0
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in loaders["train"]:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss) * len(targets)
        validation_loss, validation_metrics = evaluate(model, loaders["validation"], device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(loaders["train"].dataset),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_f1": validation_metrics["f1"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_loss={validation_loss:.4f} val_acc={validation_metrics['accuracy']:.3f} "
            f"val_f1={validation_metrics['f1']:.3f}",
            flush=True,
        )
        if float(validation_metrics["f1"]) > best_f1:
            best_f1 = float(validation_metrics["f1"])
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "frames": args.frames,
                    "input_size": input_size,
                    "feature_mode": args.feature_mode,
                    "labels": LABELS,
                    "seed": args.seed,
                    "best_epoch": epoch,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early_stop epoch={epoch} best_f1={best_f1:.3f}")
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    validation_loss, validation_metrics = evaluate(model, loaders["validation"], device)
    test_loss, test_metrics = evaluate(model, loaders["test"], device)
    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "best_epoch": saved["best_epoch"],
        "split_sizes": {key: len(value) for key, value in grouped.items()},
        "validation": {"loss": validation_loss, **validation_metrics},
        "test": {"loss": test_loss, **test_metrics},
        "history": history,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))
    print(f"checkpoint={checkpoint.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
