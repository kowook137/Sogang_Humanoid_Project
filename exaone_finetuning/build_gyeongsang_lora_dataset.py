"""Build a conservative first LoRA dataset from reviewed drafts and AI-Hub pairs."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONVERSION_DIR = MODULE_DIR / "data" / "processed" / "gyeongsang" / "conversion"
DEFAULT_DRAFTS = MODULE_DIR / "data" / "drafts" / "gyeongsang_pilot.jsonl"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "data" / "processed" / "gyeongsang" / "lora_v1"
CHAT_SYSTEM_PROMPT = (MODULE_DIR / "prompts" / "gyeongsang.txt").read_text(encoding="utf-8").strip()

ADJACENT_REPEAT = re.compile(r"(?:^|\s)([가-힣A-Za-z]+)\s+\1(?:\s|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-dir", type=Path, default=DEFAULT_CONVERSION_DIR)
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-train-medium", type=int, default=1500)
    parser.add_argument("--max-validation-medium", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_clean_conversion(record: dict) -> bool:
    source = record["source_standard"]
    target = record["target_dialect"]
    metrics = record["metrics"]
    if record["register"] != "polite" or record["strength"] == "weak":
        return False
    if not (8 <= len(source) <= 160 and 8 <= len(target) <= 160):
        return False
    if not (0.015 <= metrics["change_ratio"] <= 0.35):
        return False
    combined = source + " " + target
    if ADJACENT_REPEAT.search(combined):
        return False
    return True


def conversion_rank(record: dict) -> tuple:
    metrics = record["metrics"]
    return (
        record["strength"] == "strong",
        min(metrics["dialect_eojeol_count"], 5),
        -abs(metrics["change_ratio"] - 0.10),
        -len(record["target_dialect"]),
        record["id"],
    )


def select_conversions(records: list[dict], medium_limit: int) -> list[dict]:
    clean = [record for record in records if is_clean_conversion(record)]
    strong = [record for record in clean if record["strength"] == "strong"]
    medium = [record for record in clean if record["strength"] == "medium"]
    strong.sort(key=conversion_rank, reverse=True)
    medium.sort(key=conversion_rank, reverse=True)
    return strong + medium[:medium_limit]


def normalize_record(record: dict, source_kind: str) -> dict:
    messages = record["messages"]
    if source_kind == "curated_chat":
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, *messages]
    return {
        "id": record["id"],
        "task": "dialect_chat" if source_kind == "curated_chat" else "standard_to_dialect",
        "source_kind": source_kind,
        "messages": messages,
    }


def summarize(records: list[dict]) -> dict:
    counts = Counter(record["source_kind"] for record in records)
    tasks = Counter(record["task"] for record in records)
    return {"total": len(records), "source_kind": dict(counts), "task": dict(tasks)}


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_source = load_jsonl(args.conversion_dir / "train_polite.jsonl")
    validation_source = load_jsonl(args.conversion_dir / "validation_polite.jsonl")
    drafts = load_jsonl(args.drafts)

    train = [
        normalize_record(record, "aihub_conversion")
        for record in select_conversions(train_source, args.max_train_medium)
    ]
    train.extend(normalize_record(record, "curated_chat") for record in drafts)
    validation = [
        normalize_record(record, "aihub_conversion")
        for record in select_conversions(validation_source, args.max_validation_medium)
    ]
    rng.shuffle(train)

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    summary = {
        "version": "gyeongsang_lora_v1",
        "seed": args.seed,
        "filters": {
            "register": "polite",
            "strength": ["strong", "medium"],
            "length": [8, 160],
            "change_ratio": [0.015, 0.35],
            "reject_adjacent_repetition": True,
            "weak_examples_included": False,
        },
        "train": summarize(train),
        "validation": summarize(validation),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
