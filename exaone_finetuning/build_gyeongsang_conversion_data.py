"""Build standard-Korean to Busan-dialect conversion pairs from AI-Hub ZIPs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from zipfile import ZipFile


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = MODULE_DIR / "data" / "processed" / "gyeongsang" / "conversion"
YOUTH_AGES = {"10대", "20대", "30대"}
BUSAN_LOCATION_FIELDS = ("birthplace", "principal_residence", "current_residence")
POLITE_ENDING = re.compile(
    r"(요|습니다|습니까|입니다|입니까|합니다|합니까|세요|셔요|시죠|실래요|"
    r"드릴게요|지예|네예|입니더|습니더|는교|은교)[.!?~]*$"
)
STRONG_DIALECT = re.compile(
    r"(겠지예|네예|지예|는교|은교|아입니|하이소|보이소|"
    r"주이소|가이소|드이소|데이(?=[.!?~\s]|$)|아이가|우짜|쪼매|단디|가꼬)"
)
TARGET_VOICE_EXCLUDED = re.compile(
    r"(?:습니더|입니더|심더|심니더)[.!?~]*$|(?:노|나)[.?~]*$"
)
VERIFIED_POLITE_MARKER = re.compile(
    r"(?:네예|지예|겠지예|는교|은교|능교|"
    r"하이소|보이소|주이소|가이소|드이소|마이소)[.!?~]*$"
)
PLACEHOLDER = re.compile(
    r"(&[^&]+&|@[가-힣A-Za-z0-9_-]+|\(\(.*?\)\)|\{[^}]+\}|/\([^)]*\))"
)
TRANSCRIPTION_REPAIR = re.compile(r"-[가-힣]+-")
SYSTEM_PROMPT = (
    "다음 표준어 문장의 의미와 높임 수준을 유지하면서 "
    "현대 부산 사투리로 자연스럽게 바꾸세요."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-zip", type=Path, required=True)
    parser.add_argument("--validation-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").replace("~", "")).strip()


def is_strict_busan_speaker(speaker: dict) -> bool:
    """Require consistent Busan metadata instead of a broad Gyeongsang label."""
    return speaker.get("age") in YOUTH_AGES and all(
        speaker.get(field) == "부산" for field in BUSAN_LOCATION_FIELDS
    )


def robot_exclusion_reasons(standard: str, dialect: str, register: str) -> list[str]:
    """Conservative text-policy gate; TTS supplies uncertain regional prosody later."""
    reasons = []
    if register != "polite":
        reasons.append("non_polite")
    if TARGET_VOICE_EXCLUDED.search(dialect):
        reasons.append("excluded_ending")
    if not standard.endswith((".", "?", "!")) or not dialect.endswith((".", "?", "!")):
        reasons.append("incomplete_sentence")
    return reasons


def changed_dialect_eojeols(utterance: dict) -> list[dict]:
    """Return AI-Hub-labelled dialect tokens with their paired standard forms."""
    return [
        {
            "dialect": str(item.get("eojeol", "")).strip(),
            "standard": str(item.get("standard", "")).strip(),
        }
        for item in utterance.get("eojeolList", [])
        if item.get("isDialect")
        and str(item.get("eojeol", "")).strip()
        and str(item.get("eojeol", "")).strip()
        != str(item.get("standard", "")).strip()
    ]


def has_verified_polite_marker(changes: list[dict]) -> bool:
    """Require an explicit AI-Hub dialect label on a target-approved ending."""
    return any(VERIFIED_POLITE_MARKER.search(item["dialect"]) for item in changes)


def classify_strength(
    standard: str, dialect: str, dialect_eojeol_count: int
) -> tuple[str, float, int]:
    strong_markers = len(STRONG_DIALECT.findall(dialect))
    change_ratio = 1.0 - SequenceMatcher(None, standard, dialect).ratio()
    if strong_markers > 0:
        strength = "strong"
    elif dialect_eojeol_count >= 2 or change_ratio >= 0.08:
        strength = "medium"
    else:
        strength = "weak"
    return strength, change_ratio, strong_markers


def process_zip(path: Path, split: str) -> tuple[list[dict], dict]:
    records: list[dict] = []
    stats: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()

    with ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".json"):
                continue
            document = json.loads(archive.read(name))
            speakers = {
                str(speaker["id"]): speaker for speaker in document.get("speaker", [])
            }
            topic = document.get("metadata", {}).get("topic", "unknown")
            for utterance in document.get("utterance", []):
                stats["utterances_seen"] += 1
                speaker = speakers.get(str(utterance.get("speaker_id")), {})
                if not is_strict_busan_speaker(speaker):
                    stats["rejected_speaker_metadata"] += 1
                    continue

                standard = clean_text(utterance.get("standard_form"))
                dialect = clean_text(utterance.get("dialect_form"))
                if not standard or not dialect or standard == dialect:
                    stats["rejected_no_dialect_change"] += 1
                    continue
                if not (4 <= len(standard) <= 200 and 4 <= len(dialect) <= 200):
                    stats["rejected_length"] += 1
                    continue
                combined = standard + " " + dialect
                if PLACEHOLDER.search(combined) or TRANSCRIPTION_REPAIR.search(combined):
                    stats["rejected_annotation_noise"] += 1
                    continue
                pair = (standard, dialect)
                if pair in seen:
                    stats["rejected_duplicate"] += 1
                    continue
                seen.add(pair)

                dialect_eojeol_count = sum(
                    bool(item.get("isDialect"))
                    for item in utterance.get("eojeolList", [])
                )
                strength, change_ratio, strong_markers = classify_strength(
                    standard, dialect, dialect_eojeol_count
                )
                register = "polite" if POLITE_ENDING.search(dialect) else "non_polite"
                dialect_changes = changed_dialect_eojeols(utterance)
                exclusion_reasons = robot_exclusion_reasons(
                    standard, dialect, register
                )
                safe_for_robot = not exclusion_reasons
                verified_for_robot = safe_for_robot and has_verified_polite_marker(
                    dialect_changes
                )
                stats[f"strength_{strength}"] += 1
                stats[f"register_{register}"] += 1
                stats[f"{register}_{strength}"] += 1
                stats[
                    "robot_conservative" if safe_for_robot else "robot_excluded"
                ] += 1
                if verified_for_robot:
                    stats["robot_verified_marker"] += 1

                records.append(
                    {
                        "id": f"{split}_conversion_{len(records) + 1:07d}",
                        "dialect": "gyeongsang_busan",
                        "task": "standard_to_dialect",
                        "status": "candidate",
                        "strength": strength,
                        "register": register,
                        "safe_for_robot": safe_for_robot,
                        "verified_for_robot": verified_for_robot,
                        "robot_exclusion_reasons": exclusion_reasons,
                        "aihub_dialect_changes": dialect_changes,
                        "source_standard": standard,
                        "target_dialect": dialect,
                        "metrics": {
                            "change_ratio": round(change_ratio, 4),
                            "dialect_eojeol_count": dialect_eojeol_count,
                            "strong_marker_count": strong_markers,
                        },
                        "source": {
                            "dataset": "AI-Hub 한국어 방언 발화(경상도)",
                            "document_id": document.get("id"),
                            "utterance_id": utterance.get("id"),
                            "speaker_age": speaker.get("age"),
                            "speaker_principal_residence": speaker.get(
                                "principal_residence"
                            ),
                            "speaker_birthplace": speaker.get("birthplace"),
                            "speaker_current_residence": speaker.get(
                                "current_residence"
                            ),
                            "topic": topic,
                        },
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": standard},
                            {"role": "assistant", "content": dialect},
                        ],
                    }
                )
                stats["accepted"] += 1

    return records, {"split": split, "source_zip": str(path), "stats": dict(stats)}


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
        records, report = process_zip(path, split)
        reports.append(report)
        write_jsonl(args.output_dir / f"{split}_all.jsonl", records)
        for register in ("polite", "non_polite"):
            write_jsonl(
                args.output_dir / f"{split}_{register}.jsonl",
                [record for record in records if record["register"] == register],
            )
        for strength in ("strong", "medium", "weak"):
            write_jsonl(
                args.output_dir / f"{split}_{strength}.jsonl",
                [record for record in records if record["strength"] == strength],
            )
        write_jsonl(
            args.output_dir / f"{split}_robot_conservative.jsonl",
            [record for record in records if record["safe_for_robot"]],
        )
        write_jsonl(
            args.output_dir / f"{split}_robot_verified.jsonl",
            [record for record in records if record["verified_for_robot"]],
        )
        print(split, report["stats"])

    (args.output_dir / "summary.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
