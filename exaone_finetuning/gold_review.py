"""Prepare, validate, and export human-reviewed Gyeongsang dialogue data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


FIELDS = [
    "id",
    "topic",
    "user",
    "standard_answer",
    "candidate_1",
    "candidate_2",
    "candidate_3",
    "decision",
    "edited_answer",
    "notes",
]
DECISIONS = {"", "accept_1", "accept_2", "accept_3", "edit", "reject"}
SYSTEM_PROMPT = (
    "어르신과 대화하는 친절한 AI입니다. 정확하고 안전하게 답하면서 "
    "현대 부산·경남의 자연스러운 존댓말을 사용하세요."
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
    return rows


def normalize_candidate(record: dict) -> dict:
    candidates = record.get("candidates") or []
    if isinstance(candidates, str):
        candidates = [candidates]
    legacy = record.get("dialect_polite")
    if legacy and legacy not in candidates:
        candidates.insert(0, legacy)
    candidates = [str(value).strip() for value in candidates if str(value).strip()][:3]
    candidates.extend([""] * (3 - len(candidates)))
    return {
        "id": str(record.get("id", "")).strip(),
        "topic": str(record.get("topic", "general")).strip(),
        "user": str(record.get("user", record.get("prompt", ""))).strip(),
        "standard_answer": str(
            record.get("standard_answer", record.get("dialect_non_polite", ""))
        ).strip(),
        "candidate_1": candidates[0],
        "candidate_2": candidates[1],
        "candidate_3": candidates[2],
        "decision": "",
        "edited_answer": "",
        "notes": "",
    }


def prepare(input_path: Path, output_path: Path, limit: int) -> None:
    records = [normalize_candidate(row) for row in read_jsonl(input_path)]
    records = [row for row in records if row["id"]][:limit]
    if not records:
        raise ValueError("no records with a non-empty id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"review CSV: {output_path} ({len(records)} rows)")


def read_review_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
        return [{key: (row.get(key) or "").strip() for key in FIELDS} for row in reader]


def selected_answer(row: dict) -> str | None:
    decision = row["decision"].lower()
    if decision.startswith("accept_"):
        return row[f"candidate_{decision[-1]}"]
    if decision == "edit":
        return row["edited_answer"]
    return None


def validate(rows: list[dict], require_complete: bool = False) -> list[str]:
    errors = []
    ids = set()
    for number, row in enumerate(rows, 2):
        decision = row["decision"].lower()
        if row["id"] in ids:
            errors.append(f"row {number}: duplicate id {row['id']}")
        ids.add(row["id"])
        if decision not in DECISIONS:
            errors.append(f"row {number}: invalid decision {decision!r}")
            continue
        if require_complete and not decision:
            errors.append(f"row {number}: decision is empty")
        if decision.startswith("accept_") and not selected_answer(row):
            errors.append(f"row {number}: selected candidate is empty")
        if decision == "edit" and not row["edited_answer"]:
            errors.append(f"row {number}: edited_answer is empty")
        if selected_answer(row) and not row["user"]:
            errors.append(f"row {number}: user prompt is empty")
    return errors


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def validation_split(identifier: str, percent: int) -> bool:
    digest = hashlib.sha256(identifier.encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100 < percent


def export(rows: list[dict], output_dir: Path, validation_percent: int) -> None:
    errors = validate(rows, require_complete=True)
    if errors:
        raise ValueError("review validation failed:\n" + "\n".join(errors[:30]))

    train, validation, preferences = [], [], []
    for row in rows:
        chosen = selected_answer(row)
        if not chosen:
            continue
        sample = {
            "id": row["id"],
            "topic": row["topic"],
            "reviewed": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["user"]},
                {"role": "assistant", "content": chosen},
            ],
        }
        target = validation if validation_split(row["id"], validation_percent) else train
        target.append(sample)
        for rejected in (row["candidate_1"], row["candidate_2"], row["candidate_3"]):
            if rejected and rejected != chosen:
                preferences.append(
                    {
                        "id": row["id"],
                        "prompt": sample["messages"][:-1],
                        "chosen": [{"role": "assistant", "content": chosen}],
                        "rejected": [{"role": "assistant", "content": rejected}],
                    }
                )

    if not train or not validation:
        raise ValueError("approved rows must produce non-empty train and validation splits")
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "validation.jsonl", validation)
    write_jsonl(output_dir / "preferences.jsonl", preferences)
    summary = {
        "reviewed": len(rows),
        "approved": len(train) + len(validation),
        "rejected": sum(row["decision"].lower() == "reject" for row in rows),
        "train": len(train),
        "validation": len(validation),
        "preference_pairs": len(preferences),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("prepare", help="create an Excel-compatible review CSV")
    make.add_argument("--input", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--limit", type=int, default=100)
    check = subparsers.add_parser("validate", help="validate a completed review CSV")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("--require-complete", action="store_true")
    build = subparsers.add_parser("export", help="export approved SFT and DPO data")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--validation-percent", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    if options.command == "prepare":
        prepare(options.input, options.output, options.limit)
    elif options.command == "validate":
        errors = validate(read_review_csv(options.input), options.require_complete)
        if errors:
            raise SystemExit("\n".join(errors))
        print("review CSV is valid")
    else:
        export(read_review_csv(options.input), options.output_dir, options.validation_percent)


if __name__ == "__main__":
    main()
