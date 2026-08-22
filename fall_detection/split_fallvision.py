"""Create a deterministic stratified FallVision train/validation/test manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/datasets/fallvision/manifest.csv")
    parser.add_argument(
        "--output", type=Path, default=root / "data/datasets/fallvision/manifest_split.csv"
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    return parser.parse_args()


def stable_key(video_id: str) -> bytes:
    return hashlib.sha256(("sogang-fallvision-v1:" + video_id).encode()).digest()


def main() -> int:
    args = parse_args()
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")
    if not 0 < args.validation_fraction < 1 - args.train_fraction:
        raise ValueError("--validation-fraction must leave room for test data")
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["label"], row["category"])].append(row)
    counts = defaultdict(int)
    for group in groups.values():
        group.sort(key=lambda row: stable_key(row["video_id"]))
        train_end = round(len(group) * args.train_fraction)
        validation_end = train_end + round(len(group) * args.validation_fraction)
        for index, row in enumerate(group):
            row["split"] = (
                "train" if index < train_end else "validation" if index < validation_end else "test"
            )
            counts[row["split"]] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["video_id"]))
    print(f"output={args.output.resolve()} counts={dict(counts)}")
    print("warning=FallVision does not expose participant IDs in this manifest; this is a clip-level split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
