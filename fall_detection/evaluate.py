"""Evaluate the rule-based fall detector on processed OpenPose outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from detector import DetectorConfig, FallState, detect_fall
from features import PoseFeatures, extract_features


STATE_COLORS = {
    FallState.NORMAL: (0, 200, 0),
    FallState.FALLING: (0, 200, 255),
    FallState.FALL_CANDIDATE: (0, 120, 255),
    FallState.FALLEN: (0, 0, 255),
    FallState.UNKNOWN: (160, 160, 160),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_features(output_dir: Path) -> PoseFeatures:
    with np.load(output_dir / "keypoints.npz") as data:
        return extract_features(
            data["keypoints"],
            float(data["fps"]),
            int(data["frame_width"]),
            int(data["frame_height"]),
        )


def render_result(source: Path, destination: Path, features: PoseFeatures, states: np.ndarray, scores: np.ndarray) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open result video: {source}")
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        features.fps,
        (features.frame_width, features.frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create result video: {destination}")

    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        state = FallState(int(states[index]))
        color = STATE_COLORS[state]
        cv2.rectangle(frame, (12, 12), (465, 118), (0, 0, 0), -1)
        cv2.putText(frame, f"STATE: {state.name}", (25, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.putText(frame, f"score={scores[index]:.2f}", (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        angle = features.torso_angle[index]
        speed_values = np.asarray([features.hip_speed[index], features.head_speed[index]])
        speed = float(np.nanmax(speed_values)) if np.any(np.isfinite(speed_values)) else 0.0
        cv2.putText(
            frame,
            f"down_speed={speed:.2f}/s  torso={angle:.1f} deg",
            (25, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        writer.write(frame)
        index += 1

    capture.release()
    writer.release()
    if index != len(states):
        raise RuntimeError(f"Rendered {index} frames but expected {len(states)}")


def expected_label(name: str) -> str:
    if "_fall_" in name:
        return "fall"
    if "_normal_" in name:
        return "normal"
    raise ValueError(f"Cannot infer label from directory name: {name}")


def evaluate(outputs_root: Path, dataset_dir: Path, write_videos: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dataset_stems = {path.stem for path in dataset_dir.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}}
    for output_dir in sorted(path for path in outputs_root.iterdir() if path.name in dataset_stems and (path / "keypoints.npz").is_file()):
        try:
            label = expected_label(output_dir.name)
        except ValueError:
            continue
        features = load_features(output_dir)
        result = detect_fall(features, DetectorConfig())
        prediction = "fall" if result.fall_detected else "normal"
        row = {
            "video": output_dir.name,
            "expected": label,
            "predicted": prediction,
            "correct": label == prediction,
            "first_falling_frame": result.first_falling_frame,
            "first_fallen_frame": result.first_fallen_frame,
            "first_fallen_seconds": (
                round(result.first_fallen_frame / features.fps, 3)
                if result.first_fallen_frame is not None
                else None
            ),
            "max_score": round(float(np.nanmax(result.fall_score)), 4),
            "max_downward_speed": round(
                float(np.nanmax(np.fmax(features.hip_speed, features.head_speed))), 4
                if np.any(np.isfinite(features.hip_speed) | np.isfinite(features.head_speed))
                else 0.0,
            ),
            "max_torso_angle": round(float(np.nanmax(features.torso_angle)), 3),
            "max_bbox_aspect": round(float(np.nanmax(features.bbox_aspect)), 3),
        }
        rows.append(row)
        if write_videos:
            render_result(
                output_dir / "rendered.mp4",
                output_dir / "fall_detection.mp4",
                features,
                result.states,
                result.fall_score,
            )
        with (output_dir / "fall_detection.json").open("w", encoding="utf-8") as stream:
            json.dump(row, stream, ensure_ascii=False, indent=2)
        np.savez_compressed(
            output_dir / "fall_detection.npz",
            states=result.states,
            state_names=np.asarray([state.name for state in FallState]),
            fall_score=result.fall_score,
            head_y=features.head_y,
            hip_y=features.hip_y,
            torso_angle=features.torso_angle,
            bbox_aspect=features.bbox_aspect,
            hip_speed=features.hip_speed,
            head_speed=features.head_speed,
        )
    return rows


def write_summary(rows: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(rows: list[dict[str, object]]) -> None:
    tp = sum(row["expected"] == "fall" and row["predicted"] == "fall" for row in rows)
    tn = sum(row["expected"] == "normal" and row["predicted"] == "normal" for row in rows)
    fp = sum(row["expected"] == "normal" and row["predicted"] == "fall" for row in rows)
    fn = sum(row["expected"] == "fall" and row["predicted"] == "normal" for row in rows)
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"videos={len(rows)} TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"accuracy={accuracy:.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    for row in rows:
        marker = "OK" if row["correct"] else "MISS"
        print(f"{marker:4} {row['video']}: expected={row['expected']} predicted={row['predicted']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule-based fall detection on processed videos.")
    parser.add_argument("--outputs-dir", type=Path, default=project_root() / "outputs")
    parser.add_argument("--dataset-dir", type=Path, default=project_root() / "data" / "raw_videos")
    parser.add_argument("--summary", type=Path, default=project_root() / "outputs" / "fall_evaluation.csv")
    parser.add_argument("--write-videos", action="store_true", help="Write annotated fall_detection.mp4 per video")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = evaluate(args.outputs_dir.resolve(), args.dataset_dir.resolve(), args.write_videos)
    if not rows:
        raise RuntimeError(f"No processed fall/normal outputs found in {args.outputs_dir}")
    write_summary(rows, args.summary.resolve())
    print_metrics(rows)
    print(f"summary={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
