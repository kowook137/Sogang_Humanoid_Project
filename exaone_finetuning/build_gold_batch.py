"""Build a deduplicated 100-question gold review batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCES = [
    MODULE_DIR / "data/processed/gyeongsang/lora_v3/train.jsonl",
    MODULE_DIR / "data/processed/gyeongsang/lora_v3/validation.jsonl",
]
DEFAULT_SUPPLEMENT = MODULE_DIR / "data/gold/gyeongsang_batch_001_supplement.jsonl"
DEFAULT_OUTPUT = MODULE_DIR / "data/gold/gyeongsang_batch_001_questions.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_questions(paths: list[Path]) -> list[dict]:
    result, seen = [], set()
    for path in paths:
        for record in read_jsonl(path):
            users = [m["content"].strip() for m in record.get("messages", []) if m.get("role") == "user"]
            if not users:
                continue
            user = users[-1]
            if user in seen:
                continue
            seen.add(user)
            result.append({"id": record["id"], "topic": record.get("topic", "general"), "user": user})
    return result


def build(sources: list[Path], supplement: Path, output: Path, count: int = 100) -> list[dict]:
    rows = extract_questions(sources)
    seen = {row["user"] for row in rows}
    for row in read_jsonl(supplement):
        if row["user"] not in seen:
            rows.append(row)
            seen.add(row["user"])
    rows = rows[:count]
    if len(rows) != count:
        raise ValueError(f"expected {count} unique questions, got {len(rows)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=100)
    options = parser.parse_args()
    rows = build(DEFAULT_SOURCES, DEFAULT_SUPPLEMENT, options.output, options.count)
    print(f"questions: {len(rows)}")
    print(f"output: {options.output}")


if __name__ == "__main__":
    main()
