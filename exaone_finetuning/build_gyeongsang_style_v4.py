"""Build a balanced, real-speaker Busan dialect rewriting dataset."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = MODULE_DIR / "data" / "processed" / "gyeongsang" / "conversion"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "data" / "processed" / "gyeongsang" / "style_v4"
SYSTEM_PROMPT = (
    "다음 답변의 의미, 사실, 존댓말 수준을 그대로 유지하면서 현대 부산·경남의 "
    "자연스러운 존댓말로 바꾸세요. 새로운 정보나 조언을 추가하지 말고, "
    "같은 사투리 어미를 반복하거나 과장된 방송식 사투리를 쓰지 마세요."
)
TOKEN = re.compile(r"[가-힣A-Za-z0-9]+")
ANNOTATION_NOISE = re.compile(
    r"(&[^&]+&|@[가-힣A-Za-z0-9_-]+|\(\(.*?\)\)|\{[^}]+\}|/\([^)]*\)|-[가-힣]+-)"
)
POLITE_ENDING = re.compile(
    r"(요|습니다|습니까|입니다|입니까|합니다|합니까|세요|셔요|시죠|실래요|"
    r"드릴게요|지예|네예|입니더|습니더|는교|은교|예)[.!?~]*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-medium", type=int, default=800)
    parser.add_argument("--train-weak", type=int, default=1400)
    parser.add_argument("--validation-medium", type=int, default=150)
    parser.add_argument("--validation-weak", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def repetition_count(text: str) -> int:
    counts = Counter(TOKEN.findall(text))
    return sum(count - 2 for count in counts.values() if count > 2)


def is_clean(record: dict) -> bool:
    source = record.get("source_standard", "").strip()
    target = record.get("target_dialect", "").strip()
    metrics = record.get("metrics", {})
    ratio = metrics.get("change_ratio")
    if record.get("register") != "polite":
        return False
    if record.get("strength") not in {"weak", "medium", "strong"}:
        return False
    if not (12 <= len(source) <= 180 and 12 <= len(target) <= 180):
        return False
    if not isinstance(ratio, (int, float)) or not (0.02 <= ratio <= 0.30):
        return False
    if source == target or ANNOTATION_NOISE.search(source + " " + target):
        return False
    if not POLITE_ENDING.search(target):
        return False
    if repetition_count(source) > 3 or repetition_count(target) > 3:
        return False
    return True


def quality_rank(record: dict) -> tuple:
    metrics = record["metrics"]
    ratio = metrics["change_ratio"]
    target_ratio = 0.06 if record["strength"] == "weak" else 0.11
    return (
        min(metrics.get("dialect_eojeol_count", 0), 6),
        -abs(ratio - target_ratio),
        -repetition_count(record["target_dialect"]),
        -abs(len(record["target_dialect"]) - 60),
        record["id"],
    )


def _round_robin_by_topic(records: list[dict], limit: int) -> list[dict]:
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_topic[record["source"].get("topic", "unknown")].append(record)
    for candidates in by_topic.values():
        candidates.sort(key=quality_rank, reverse=True)

    selected: list[dict] = []
    topics = sorted(by_topic)
    while topics and len(selected) < limit:
        remaining = []
        for topic in topics:
            if by_topic[topic] and len(selected) < limit:
                selected.append(by_topic[topic].pop(0))
            if by_topic[topic]:
                remaining.append(topic)
        topics = remaining
    return selected


def balanced_select(records: list[dict], strength: str, limit: int) -> list[dict]:
    candidates = [
        record
        for record in records
        if record["strength"] == strength and is_clean(record)
    ]
    # `쫌` is extremely frequent in the source corpus. Prefer every other real
    # transformation first, then use `쫌` examples only to fill the requested size.
    diverse = [record for record in candidates if "쫌" not in record["target_dialect"]]
    jjom = [record for record in candidates if "쫌" in record["target_dialect"]]
    selected = _round_robin_by_topic(diverse, limit)
    if len(selected) < limit:
        selected.extend(_round_robin_by_topic(jjom, limit - len(selected)))
    return selected


def normalize(record: dict) -> dict:
    return {
        "id": record["id"],
        "task": "dialect_rewrite",
        "source_kind": "aihub_busan_real_speech",
        "strength": record["strength"],
        "topic": record["source"].get("topic", "unknown"),
        "source": record["source"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["source_standard"]},
            {"role": "assistant", "content": record["target_dialect"]},
        ],
    }


def build_split(
    records: list[dict], medium_limit: int, weak_limit: int, rng: random.Random
) -> list[dict]:
    strong = sorted(
        (record for record in records if record["strength"] == "strong" and is_clean(record)),
        key=quality_rank,
        reverse=True,
    )
    selected = [
        *strong,
        *balanced_select(records, "medium", medium_limit),
        *balanced_select(records, "weak", weak_limit),
    ]
    normalized = [normalize(record) for record in selected]
    rng.shuffle(normalized)
    return normalized


def fingerprint(record: dict) -> tuple[str, str]:
    return record["messages"][1]["content"], record["messages"][2]["content"]


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("dataset split is empty after quality filtering")
    return {
        "records": len(records),
        "strength": dict(sorted(Counter(r["strength"] for r in records).items())),
        "topics": dict(sorted(Counter(r["topic"] for r in records).items())),
        "average_target_chars": round(
            sum(len(r["messages"][2]["content"]) for r in records) / len(records), 1
        ),
    }


def build_dataset(
    train_source: list[dict],
    validation_source: list[dict],
    *,
    train_medium: int = 800,
    train_weak: int = 1400,
    validation_medium: int = 150,
    validation_weak: int = 350,
    seed: int = 42,
) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    train = build_split(train_source, train_medium, train_weak, rng)
    validation = build_split(
        validation_source, validation_medium, validation_weak, rng
    )
    overlap = {fingerprint(r) for r in train} & {fingerprint(r) for r in validation}
    if overlap:
        raise ValueError(f"train/validation leakage: {len(overlap)} duplicate pairs")
    summary = {
        "version": "gyeongsang_style_v4",
        "task": "meaning_preserving_dialect_rewrite",
        "source": "AI-Hub 한국어 방언 발화(경상도), 부산 주 성장지 10~30대",
        "seed": seed,
        "policy": {
            "real_speaker_pairs_only": True,
            "polite_only": True,
            "safety_and_memory_training": False,
            "runtime_usage": "second_pass_rewriter",
            "human_review_required": True,
        },
        "train": summarize(train),
        "validation": summarize(validation),
    }
    return train, validation, summary


def main() -> int:
    args = parse_args()
    train_source = load_jsonl(args.source_dir / "train_polite.jsonl")
    validation_source = load_jsonl(args.source_dir / "validation_polite.jsonl")
    train, validation, summary = build_dataset(
        train_source,
        validation_source,
        train_medium=args.train_medium,
        train_weak=args.train_weak,
        validation_medium=args.validation_medium,
        validation_weak=args.validation_weak,
        seed=args.seed,
    )
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
