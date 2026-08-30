"""Build a high-contrast, register-conditioned Busan rewrite dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from build_gyeongsang_style_v4 import (
    ANNOTATION_NOISE,
    _round_robin_by_topic,
    fingerprint,
    load_jsonl,
    repetition_count,
    write_jsonl,
)

MODULE_DIR = Path(__file__).resolve().parent
SOURCE = MODULE_DIR / "data/processed/gyeongsang/conversion"
OUTPUT = MODULE_DIR / "data/processed/gyeongsang/style_v5"
PROMPTS = {
    "polite": (
        "표준어 문장의 뜻과 존댓말 수준을 그대로 유지하면서 현대 부산·경남의 "
        "자연스러운 존댓말로 바꾸세요. 정보를 추가하거나 삭제하지 마세요."
    ),
    "non_polite": (
        "표준어 문장의 뜻과 반말 수준을 그대로 유지하면서 현대 부산·경남의 "
        "자연스러운 반말로 바꾸세요. 정보를 추가하거나 삭제하지 마세요."
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=52)
    return parser.parse_args()


def clean(record):
    source = record["source_standard"].strip()
    target = record["target_dialect"].strip()
    ratio = record["metrics"]["change_ratio"]
    return (
        record["strength"] in {"medium", "strong"}
        and 10 <= len(source) <= 180
        and 10 <= len(target) <= 180
        and 0.08 <= ratio <= 0.40
        and source != target
        and not ANNOTATION_NOISE.search(source + " " + target)
        and repetition_count(source) <= 3
        and repetition_count(target) <= 3
    )


def select(records, register, strength, limit):
    candidates = [
        record for record in records
        if clean(record) and record["register"] == register and record["strength"] == strength
    ]
    return _round_robin_by_topic(candidates, min(limit, len(candidates)))


def normalize(record):
    register = record["register"]
    return {
        "id": f"v5_{record['id']}",
        "task": "register_conditioned_dialect_rewrite",
        "register": register,
        "strength": record["strength"],
        "topic": record["source"].get("topic", "unknown"),
        "metrics": record["metrics"],
        "messages": [
            {"role": "system", "content": PROMPTS[register]},
            {"role": "user", "content": record["source_standard"]},
            {"role": "assistant", "content": record["target_dialect"]},
        ],
    }


def build_split(records, limits, seed, polite_weight=1):
    selected = []
    for (register, strength), limit in limits.items():
        selected.extend(select(records, register, strength, limit))
    output = [normalize(record) for record in selected]
    polite = [record for record in output if record["register"] == "polite"]
    for repeat in range(2, polite_weight + 1):
        for record in polite:
            weighted = {**record, "id": f"{record['id']}_weight{repeat}"}
            output.append(weighted)
    random.Random(seed).shuffle(output)
    return output


def summary(records):
    return {
        "records": len(records),
        "register": dict(Counter(r["register"] for r in records)),
        "strength": dict(Counter(r["strength"] for r in records)),
        "groups": dict(Counter(f"{r['register']}_{r['strength']}" for r in records)),
    }


def main():
    args = parse_args()
    train_source = load_jsonl(args.source_dir / "train_all.jsonl")
    validation_source = load_jsonl(args.source_dir / "validation_all.jsonl")
    train = build_split(train_source, {
        ("polite", "strong"): 1000,
        ("polite", "medium"): 5000,
        ("non_polite", "strong"): 4000,
        ("non_polite", "medium"): 1000,
    }, args.seed, polite_weight=4)
    validation = build_split(validation_source, {
        ("polite", "strong"): 200,
        ("polite", "medium"): 500,
        ("non_polite", "strong"): 300,
        ("non_polite", "medium"): 300,
    }, args.seed + 1)
    overlap = {fingerprint(r) for r in train} & {fingerprint(r) for r in validation}
    if overlap:
        raise ValueError(f"train/validation leakage: {len(overlap)}")
    report = {
        "version": "gyeongsang_style_v5_high_contrast",
        "weak_examples": 0,
        "train": summary(train),
        "validation": summary(validation),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
