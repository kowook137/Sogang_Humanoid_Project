#!/usr/bin/env python3
"""Convert verified recording_manifest.csv rows to GPT-SoVITS .list format."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root = args.manifest.parent
    lines: list[str] = []
    errors: list[str] = []
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            if row["status"].strip().lower() != "ready":
                continue
            if row["transcript_verified"].strip().lower() != "true":
                errors.append(f"{row_number}: ready인데 transcript_verified가 true가 아님")
                continue
            audio = (root / row["audio_path"]).resolve()
            if not audio.is_file() or audio.stat().st_size == 0:
                errors.append(f"{row_number}: WAV 없음: {audio}")
                continue
            text = row["text"].strip()
            if not text or any(mark in text for mark in "|\r\n"):
                errors.append(f"{row_number}: 대본이 비었거나 금지 문자를 포함함")
                continue
            speaker = row["speaker"].strip() or "busan_user"
            language = row["language"].strip() or "ko"
            lines.append(f"{audio}|{speaker}|{language}|{text}")
    if errors:
        raise SystemExit("\n".join(errors))
    if not lines:
        raise SystemExit("검수 완료 녹음이 없습니다. WAV를 넣고 status=ready, transcript_verified=true로 바꾸세요.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"완료: {args.output} ({len(lines)}개)")

if __name__ == "__main__":
    main()
