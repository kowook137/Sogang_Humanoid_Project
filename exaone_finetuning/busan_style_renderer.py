"""Render conservative Busan text using transformations grounded in AI-Hub speech."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    replacement: str
    evidence: str


# Rules are deliberately sentence-final and change only the ending. 2020 AI-Hub
# speech is treated as representative for this project's elderly-facing persona.
RULES = (
    Rule(
        "neyo_to_neye",
        re.compile(r"네요(?=[.!?](?:\s|$))"),
        "네예",
        "DKCI20000122.1.1.86",
    ),
    Rule(
        "jiyo_to_jiye",
        re.compile(r"지요(?=[.!?](?:\s|$))"),
        "지예",
        "DKSR20003393.1.1.159",
    ),
    Rule(
        "jyo_to_jiye",
        re.compile(r"죠(?=[.!?](?:\s|$))"),
        "지예",
        "DKSR20002654.1.1.1",
    ),
    Rule(
        "request_juseyo_to_juiso",
        re.compile(r"해 주세요(?=[.!?](?:\s|$))"),
        "해 주이소",
        "DKSR20004012.1.1.95",
    ),
    Rule(
        "suggest_boseyo_to_boiso",
        re.compile(r"해 보세요(?=[.!?](?:\s|$))"),
        "해 보이소",
        "DKSR20004059.1.1.209",
    ),
    Rule(
        "prohibit_maseyo_to_maiso",
        re.compile(r"지 마세요(?=[.!?](?:\s|$))"),
        "지 마이소",
        "DKSR20005045.1.1.17",
    ),
    Rule(
        "request_haseyo_to_hyiso",
        re.compile(r"하세요(?=[.!?](?:\s|$))"),
        "하이소",
        "DKSR20001357.1.1.248",
    ),
)


def render(text: str, max_changes: int = 1) -> tuple[str, list[dict[str, str]]]:
    """Apply at most ``max_changes`` exact, traceable ending substitutions."""
    rendered = text.strip()
    changes: list[dict[str, str]] = []
    if max_changes < 1:
        return rendered, changes
    for rule in RULES:
        match = rule.pattern.search(rendered)
        if not match:
            continue
        before = match.group(0)
        rendered = rendered[: match.start()] + rule.replacement + rendered[match.end() :]
        changes.append(
            {
                "rule": rule.name,
                "before": before,
                "after": rule.replacement,
                "evidence": rule.evidence,
            }
        )
        if len(changes) >= max_changes:
            break
    return rendered, changes


def convert_review_csv(input_path: Path, output_path: Path) -> dict[str, int]:
    """Create review-only pairs from standard answers without free-form rewriting."""
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            standard = (row.get("standard_answer") or "").strip()
            styled, changes = render(standard)
            changed += bool(changes)
            record = {
                "id": row.get("id", ""),
                "category": row.get("category", row.get("topic", "general")),
                "user": row.get("user", ""),
                "standard_answer": standard,
                "busan_answer": styled,
                "changed": bool(changes),
                "applied_changes": changes,
                "status": "review_required",
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "changed": changed, "unchanged": len(rows) - changed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    print(json.dumps(convert_review_csv(options.input_csv, options.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
