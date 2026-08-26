"""Validate the v4 dialect rewriting data before any GPU training starts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = MODULE_DIR / "data" / "processed" / "gyeongsang" / "style_v4"
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
POLITE = re.compile(
    r"(요|습니다|습니까|입니다|입니까|합니다|합니까|세요|셔요|시죠|실래요|"
    r"지예|네예|입니더|습니더|는교|은교|예)[.!?~]*$"
)
DIALECT = re.compile(
    r"(몬|마이|쫌|함|이케|가주고|가주구|그라모|카이|캐도|지예|네예|"
    r"입니더|습니더|는교|은교|하이소|보이소|주이소|가이소|드이소|예)"
)
TOKEN = re.compile(r"[가-힣A-Za-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def repeated_tokens(text: str) -> int:
    counts = Counter(TOKEN.findall(text))
    return sum(count - 2 for count in counts.values() if count > 2)


def analyze(records: list[dict]) -> dict:
    similarities = []
    changed = 0
    polite = 0
    dialect = 0
    number_mismatches = 0
    repetitions = 0
    ids = set()
    pairs = set()
    duplicate_ids = 0
    duplicate_pairs = 0
    marker_counts: Counter[str] = Counter()
    marker_record_counts: Counter[str] = Counter()
    for record in records:
        record_id = record["id"]
        source = record["messages"][1]["content"]
        target = record["messages"][2]["content"]
        pair = (source, target)
        duplicate_ids += record_id in ids
        duplicate_pairs += pair in pairs
        ids.add(record_id)
        pairs.add(pair)
        changed += source != target
        polite += bool(POLITE.search(target))
        markers = DIALECT.findall(target)
        dialect += bool(markers)
        marker_counts.update(markers)
        marker_record_counts.update(set(markers))
        number_mismatches += NUMBER.findall(source) != NUMBER.findall(target)
        repetitions += repeated_tokens(target) > 3
        similarities.append(SequenceMatcher(None, source, target).ratio())
    total = len(records)
    return {
        "records": total,
        "changed_rate": round(changed / total, 4),
        "polite_rate": round(polite / total, 4),
        "dialect_marker_rate": round(dialect / total, 4),
        "number_mismatch_rate": round(number_mismatches / total, 4),
        "excessive_repetition_rate": round(repetitions / total, 4),
        "similarity_median": round(statistics.median(similarities), 4),
        "similarity_p10": round(sorted(similarities)[max(0, total // 10 - 1)], 4),
        "duplicate_ids": duplicate_ids,
        "duplicate_pairs": duplicate_pairs,
        "top_markers": dict(marker_counts.most_common(20)),
        "top_marker_record_rates": {
            marker: round(count / total, 4)
            for marker, count in marker_record_counts.most_common(20)
        },
    }


def validate(train: list[dict], validation: list[dict]) -> dict:
    report = {"train": analyze(train), "validation": analyze(validation)}
    train_pairs = {
        (r["messages"][1]["content"], r["messages"][2]["content"])
        for r in train
    }
    validation_pairs = {
        (r["messages"][1]["content"], r["messages"][2]["content"])
        for r in validation
    }
    report["cross_split_pair_overlap"] = len(train_pairs & validation_pairs)

    failures = []
    if report["train"]["records"] < 2000:
        failures.append("train records must be at least 2000")
    if report["validation"]["records"] < 400:
        failures.append("validation records must be at least 400")
    for split in ("train", "validation"):
        metrics = report[split]
        if metrics["changed_rate"] != 1.0:
            failures.append(f"{split}: unchanged targets found")
        if metrics["polite_rate"] < 0.98:
            failures.append(f"{split}: polite rate below 98%")
        if metrics["dialect_marker_rate"] < 0.15:
            failures.append(f"{split}: dialect marker rate below 15%")
        if metrics["top_marker_record_rates"] and max(
            metrics["top_marker_record_rates"].values()
        ) > 0.35:
            failures.append(f"{split}: one dialect marker appears in over 35% of records")
        if metrics["number_mismatch_rate"] > 0.005:
            failures.append(f"{split}: number mismatch rate above 0.5%")
        if metrics["excessive_repetition_rate"] > 0.01:
            failures.append(f"{split}: repetition rate above 1%")
        if not (0.72 <= metrics["similarity_median"] <= 0.99):
            failures.append(f"{split}: median similarity outside safe range")
        if metrics["duplicate_ids"] or metrics["duplicate_pairs"]:
            failures.append(f"{split}: duplicates found")
    if report["cross_split_pair_overlap"]:
        failures.append("train/validation pair leakage found")
    report["passed"] = not failures
    report["failures"] = failures
    return report


def main() -> int:
    args = parse_args()
    report = validate(
        load_jsonl(args.data_dir / "train.jsonl"),
        load_jsonl(args.data_dir / "validation.jsonl"),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    report_path = args.report or args.data_dir / "quality_report.json"
    report_path.write_text(rendered, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("style v4 quality gate failed")
    print(f"quality gate passed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
