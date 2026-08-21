"""Prepare the downloaded GMDCSA-24 videos for the local fall pipeline."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
SPLITS = {1: "train", 2: "train", 3: "validation", 4: "test"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_dataset_root(extracted_dir: Path) -> Path:
    candidates = [
        path
        for path in extracted_dir.iterdir()
        if path.is_dir() and any(path.glob("Subject */Fall"))
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one GMDCSA-24 root under {extracted_dir}, found {len(candidates)}"
        )
    return candidates[0]


def read_descriptions(csv_path: Path) -> dict[str, tuple[str, str]]:
    descriptions: dict[str, tuple[str, str]] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            # Some source descriptions contain unquoted commas. Rejoin them.
            descriptions[row[0].strip()] = (
                row[1].strip(),
                ",".join(row[2:]).strip(),
            )
    return descriptions


def build_manifest(dataset_root: Path, link_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    link_dir.mkdir(parents=True, exist_ok=True)
    for subject_dir in sorted(dataset_root.glob("Subject *")):
        try:
            subject_number = int(subject_dir.name.rsplit(" ", 1)[1])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"Unexpected subject directory: {subject_dir}") from error
        if subject_number not in SPLITS:
            raise RuntimeError(f"No split assigned for Subject {subject_number}")
        subject = f"s{subject_number:02d}"
        for source_class, label in (("Fall", "fall"), ("ADL", "normal")):
            descriptions = read_descriptions(subject_dir / f"{source_class}.csv")
            videos = sorted(
                path
                for path in (subject_dir / source_class).iterdir()
                if path.suffix.lower() in VIDEO_SUFFIXES
            )
            for video in videos:
                video_id = f"{subject}_{label}_{int(video.stem):03d}"
                link = link_dir / f"{video_id}{video.suffix.lower()}"
                target = Path(os.path.relpath(video.resolve(), start=link.parent.resolve()))
                if link.is_symlink():
                    if Path(os.readlink(link)) != target:
                        link.unlink()
                elif link.exists():
                    raise FileExistsError(f"Refusing to replace non-symlink: {link}")
                if not link.exists():
                    link.symlink_to(target)
                duration, description = descriptions.get(video.name, ("", ""))
                rows.append(
                    {
                        "video_id": video_id,
                        "subject": subject,
                        "label": label,
                        "split": SPLITS[subject_number],
                        "duration": duration,
                        "description": description,
                        "source_path": str(video.resolve()),
                        "pipeline_path": str(link.absolute()),
                    }
                )
    return rows


def write_manifest(rows: list[dict[str, str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=root / "data/datasets/gmdcsa24/extracted",
    )
    parser.add_argument(
        "--link-dir",
        type=Path,
        default=root / "data/raw_videos",
        help="Flat symlink view compatible with the existing evaluation pipeline",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "data/datasets/gmdcsa24/manifest.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = find_dataset_root(args.extracted_dir.resolve())
    rows = build_manifest(dataset_root, args.link_dir.resolve())
    if not rows:
        raise RuntimeError(f"No videos found under {dataset_root}")
    write_manifest(rows, args.manifest.resolve())
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["split"], row["label"])
        counts[key] = counts.get(key, 0) + 1
    print(f"dataset_root={dataset_root}")
    print(f"manifest={args.manifest.resolve()}")
    print(f"link_dir={args.link_dir.resolve()}")
    print(f"videos={len(rows)}")
    for split in ("train", "validation", "test"):
        print(
            f"{split}: fall={counts.get((split, 'fall'), 0)} "
            f"normal={counts.get((split, 'normal'), 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
