"""Analyze AI-Hub Gyeongsang ZIPs and extract conservative Busan chat pairs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


YOUTH_AGES = {"10대", "20대", "30대"}
POLITE_ENDING = re.compile(
    r"(요|니다|니까|세요|셔요|시죠|실래요|드릴게요|예)[.!?~]*$"
)
PLACEHOLDER = re.compile(r"(&[^&]+&|@[가-힣A-Za-z_-]+|\(\(.*?\)\))")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-zip", type=Path, required=True)
    parser.add_argument("--validation-zip", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "processed" / "gyeongsang",
    )
    return parser.parse_args()


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("~", "")).strip()


def is_complete(value: str) -> bool:
    return 8 <= len(value) <= 200 and value.endswith((".", "?", "!"))


def analyze_zip(path: Path, split: str) -> tuple[dict, list[dict]]:
    stats: Counter[str] = Counter()
    ages: Counter[str] = Counter()
    residences: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()

    with ZipFile(path) as archive:
        names = archive.namelist()
        stats["archive_entries"] = len(names)
        for name in names:
            if not name.lower().endswith(".json"):
                stats["non_json_files"] += 1
                continue

            document = json.loads(archive.read(name))
            stats["documents"] += 1
            topic = document.get("metadata", {}).get("topic", "unknown")
            topics[topic] += 1
            speakers = {
                str(speaker["id"]): speaker for speaker in document.get("speaker", [])
            }
            for speaker in speakers.values():
                ages[speaker.get("age", "unknown")] += 1
                residences[speaker.get("principal_residence", "unknown")] += 1

            utterances = document.get("utterance", [])
            stats["utterances"] += len(utterances)
            for utterance in utterances:
                dialect = clean_text(utterance.get("dialect_form"))
                standard = clean_text(utterance.get("standard_form"))
                if dialect and dialect != standard:
                    stats["dialect_utterances"] += 1

            for previous, current in zip(utterances, utterances[1:]):
                if previous.get("speaker_id") == current.get("speaker_id"):
                    continue
                speaker = speakers.get(str(current.get("speaker_id")), {})
                if speaker.get("age") not in YOUTH_AGES:
                    continue
                if speaker.get("principal_residence") != "부산":
                    continue

                user_text = clean_text(previous.get("standard_form"))
                assistant_text = clean_text(current.get("dialect_form"))
                assistant_standard = clean_text(current.get("standard_form"))
                if not is_complete(user_text) or not is_complete(assistant_text):
                    continue
                if assistant_text == assistant_standard:
                    continue
                if not POLITE_ENDING.search(assistant_text):
                    continue
                stats["strict_pairs_before_cleanup"] += 1
                if PLACEHOLDER.search(user_text) or PLACEHOLDER.search(assistant_text):
                    stats["rejected_placeholders"] += 1
                    continue
                pair = (user_text, assistant_text)
                if pair in seen:
                    stats["rejected_duplicates"] += 1
                    continue
                seen.add(pair)
                candidates.append(
                    {
                        "id": f"{split}_{len(candidates) + 1:06d}",
                        "dialect": "gyeongsang_busan",
                        "status": "candidate",
                        "source": {
                            "dataset": "AI-Hub 한국어 방언 발화(경상도)",
                            "document_id": document.get("id"),
                            "utterance_id": current.get("id"),
                            "speaker_age": speaker.get("age"),
                            "speaker_principal_residence": speaker.get(
                                "principal_residence"
                            ),
                            "topic": topic,
                        },
                        "messages": [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": assistant_text},
                        ],
                        "assistant_standard": assistant_standard,
                    }
                )

    stats["candidates"] = len(candidates)
    report = {
        "split": split,
        "source_zip": str(path),
        "stats": dict(stats),
        "speaker_ages": dict(ages.most_common()),
        "principal_residences": dict(residences.most_common()),
        "top_topics": dict(topics.most_common(20)),
    }
    return report, candidates


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for split, path in (
        ("train", args.training_zip),
        ("validation", args.validation_zip),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        report, candidates = analyze_zip(path, split)
        reports.append(report)
        write_jsonl(args.output_dir / f"{split}_candidates.jsonl", candidates)
        print(f"{split}: {len(candidates)} candidates")

    (args.output_dir / "analysis.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
