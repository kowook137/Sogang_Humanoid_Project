"""Build the dialogue-first Gyeongsang QLoRA v2 dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = MODULE_DIR / "data" / "drafts" / "gyeongsang_chat_v2.jsonl"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "data" / "processed" / "gyeongsang" / "lora_v2"
SYSTEM_PROMPT = (MODULE_DIR / "prompts" / "gyeongsang.txt").read_text(
    encoding="utf-8"
).strip()
DIALECT_MARKER = re.compile(
    r"(네예|지예|입니더|습니더|는교|은교|아입니|하이소|보이소|"
    r"주이소|가이소|드이소|쉬이소|누우이소|그라모|맞나|데이)"
)
KNOWN_BAD_PHRASES = ("습당", "반갑습당", "하게요!", "hurt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_and_normalize(records: list[dict]) -> list[dict]:
    normalized = []
    seen_ids = set()
    seen_conversations = set()
    for line_number, record in enumerate(records, start=1):
        record_id = record.get("id")
        split = record.get("split")
        topic = record.get("topic")
        messages = record.get("messages")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"line {line_number}: id is required")
        if record_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate id {record_id}")
        if split not in {"train", "validation"}:
            raise ValueError(f"line {line_number}: split must be train or validation")
        if not isinstance(topic, str) or not topic:
            raise ValueError(f"line {line_number}: topic is required")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"line {line_number}: at least one dialogue turn is required")
        if len(messages) % 2 != 0:
            raise ValueError(f"line {line_number}: dialogue must end with assistant")

        clean_messages = []
        for index, message in enumerate(messages):
            expected_role = "user" if index % 2 == 0 else "assistant"
            if not isinstance(message, dict) or message.get("role") != expected_role:
                raise ValueError(
                    f"line {line_number}, message {index + 1}: expected {expected_role}"
                )
            content = message.get("content")
            if not isinstance(content, str) or len(content.strip()) < 2:
                raise ValueError(
                    f"line {line_number}, message {index + 1}: content is too short"
                )
            content = re.sub(r"\s+", " ", content).strip()
            if any(phrase in content for phrase in KNOWN_BAD_PHRASES):
                raise ValueError(
                    f"line {line_number}, message {index + 1}: known bad phrase found"
                )
            clean_messages.append({"role": expected_role, "content": content})

        fingerprint = tuple((item["role"], item["content"]) for item in clean_messages)
        if fingerprint in seen_conversations:
            raise ValueError(f"line {line_number}: duplicate conversation")
        seen_ids.add(record_id)
        seen_conversations.add(fingerprint)
        normalized.append(
            {
                "id": record_id,
                "task": "dialect_chat",
                "source_kind": "curated_chat_v2",
                "topic": topic,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *clean_messages,
                ],
                "_split": split,
            }
        )
    return normalized


def conversation_fingerprint(record: dict) -> tuple[str, ...]:
    return tuple(
        message["content"]
        for message in record["messages"]
        if message["role"] in {"user", "assistant"}
    )


def summarize(records: list[dict]) -> dict:
    assistant_messages = [
        message["content"]
        for record in records
        for message in record["messages"]
        if message["role"] == "assistant"
    ]
    return {
        "records": len(records),
        "assistant_turns": len(assistant_messages),
        "multi_turn_records": sum(
            sum(message["role"] == "assistant" for message in record["messages"]) > 1
            for record in records
        ),
        "topics": dict(sorted(Counter(record["topic"] for record in records).items())),
        "average_assistant_chars": round(
            sum(map(len, assistant_messages)) / len(assistant_messages), 1
        ),
        "dialect_marker_rate": round(
            sum(bool(DIALECT_MARKER.search(text)) for text in assistant_messages)
            / len(assistant_messages),
            3,
        ),
    }


def build_dataset(records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    normalized = validate_and_normalize(records)
    train = [
        {key: value for key, value in record.items() if key != "_split"}
        for record in normalized
        if record["_split"] == "train"
    ]
    validation = [
        {key: value for key, value in record.items() if key != "_split"}
        for record in normalized
        if record["_split"] == "validation"
    ]
    if not train or not validation:
        raise ValueError("both train and validation splits must be non-empty")

    train_fingerprints = {conversation_fingerprint(record) for record in train}
    validation_fingerprints = {conversation_fingerprint(record) for record in validation}
    overlap = train_fingerprints & validation_fingerprints
    if overlap:
        raise ValueError(f"train/validation leakage detected: {len(overlap)} conversations")

    summary = {
        "version": "gyeongsang_lora_v2_pilot",
        "policy": {
            "task": "dialect_chat_only",
            "conversion_examples_included": False,
            "assistant_only_loss_expected": True,
            "manual_review_required_before_scale_up": True,
        },
        "train": summarize(train),
        "validation": summarize(validation),
    }
    return train, validation, summary


def main() -> int:
    args = parse_args()
    train, validation, summary = build_dataset(load_jsonl(args.source))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
