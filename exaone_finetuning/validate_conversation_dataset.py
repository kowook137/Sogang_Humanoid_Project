"""Validate reviewed conversation JSONL and detect train/evaluation leakage."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


SPACE = re.compile(r"\s+")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return rows


def normalized_text(value: str) -> str:
    return SPACE.sub(" ", value).strip().casefold()


def conversation_text(record: dict, role: str | None = None) -> str:
    values = []
    for message in record.get("messages") or []:
        if role is None or message.get("role") == role:
            values.append(str(message.get("content", "")))
    return normalized_text("\n".join(values))


def validate_plan(plan: dict) -> list[str]:
    errors = []
    targets = plan.get("targets") or {}
    expected = targets.get("approved_training_records")
    training = plan.get("training_1000") or {}
    if sum(training.values()) != expected:
        errors.append("training category counts do not match approved_training_records")
    pilot = plan.get("pilot_100") or {}
    if sum(pilot.values()) != 100:
        errors.append("pilot_100 category counts must sum to 100")
    if set(pilot) != set(training):
        errors.append("pilot and training categories differ")
    if sum((plan.get("input_style_percent") or {}).values()) != 100:
        errors.append("input_style_percent must sum to 100")
    return errors


def validate_records(
    rows: list[dict], plan: dict, *, training: bool, strict: bool
) -> tuple[list[str], dict]:
    errors = []
    ids = set()
    categories = Counter()
    input_styles = Counter()
    assistant_turns = 0
    requirements = plan.get("record_requirements") or {}
    required_fields = requirements.get("required_fields") or []
    allowed_roles = set(requirements.get("allowed_roles") or [])
    multi_minimum = requirements.get("multi_turn_minimum_assistant_turns", 2)

    for index, row in enumerate(rows, 1):
        label = f"row {index}"
        identifier = str(row.get("id", "")).strip()
        if not identifier:
            errors.append(f"{label}: missing id")
        elif identifier in ids:
            errors.append(f"{label}: duplicate id {identifier}")
        ids.add(identifier)

        if strict:
            for field in required_fields:
                if field not in row:
                    errors.append(f"{label}: missing field {field}")

        category = str(row.get("category", row.get("topic", ""))).strip()
        if category:
            categories[category] += 1
        elif strict:
            errors.append(f"{label}: missing category")
        input_style = str(row.get("input_style", "")).strip()
        if input_style:
            input_styles[input_style] += 1
        elif strict:
            errors.append(f"{label}: missing input_style")

        if training and row.get("reviewed") is not True:
            errors.append(f"{label}: training record is not human reviewed")

        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            errors.append(f"{label}: messages must be a non-empty list")
            continue
        roles = []
        for message_index, message in enumerate(messages, 1):
            role = message.get("role") if isinstance(message, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if role not in allowed_roles:
                errors.append(f"{label} message {message_index}: invalid role {role!r}")
            if not isinstance(content, str) or not content.strip():
                errors.append(f"{label} message {message_index}: empty content")
            roles.append(role)
        user_count = roles.count("user")
        assistant_count = roles.count("assistant")
        assistant_turns += assistant_count
        if not user_count or not assistant_count:
            errors.append(f"{label}: requires user and assistant turns")
        if category.startswith("multi_turn") and assistant_count < multi_minimum:
            errors.append(
                f"{label}: multi-turn record has {assistant_count} assistant turns; "
                f"requires at least {multi_minimum}"
            )

    summary = {
        "records": len(rows),
        "assistant_turns": assistant_turns,
        "categories": dict(sorted(categories.items())),
        "input_styles": dict(sorted(input_styles.items())),
    }
    return errors, summary


def leakage_errors(train: list[dict], evaluation: list[dict]) -> list[str]:
    errors = []
    train_ids = {str(row.get("id", "")).strip() for row in train}
    eval_ids = {str(row.get("id", "")).strip() for row in evaluation}
    for identifier in sorted((train_ids & eval_ids) - {""}):
        errors.append(f"train/evaluation duplicate id: {identifier}")

    train_user = {conversation_text(row, "user") for row in train} - {""}
    train_full = {conversation_text(row) for row in train} - {""}
    for row in evaluation:
        identifier = str(row.get("id", "")).strip() or "<missing-id>"
        if conversation_text(row, "user") in train_user:
            errors.append(f"evaluation user text leaked from training: {identifier}")
        if conversation_text(row) in train_full:
            errors.append(f"evaluation conversation leaked from training: {identifier}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("dataset_plan.json"))
    parser.add_argument("--train", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    plan = read_json(options.plan)
    errors = validate_plan(plan)
    report = {"plan": str(options.plan)}
    train_rows = read_jsonl(options.train) if options.train else []
    eval_rows = read_jsonl(options.evaluation) if options.evaluation else []
    if options.train:
        train_errors, report["training"] = validate_records(
            train_rows, plan, training=True, strict=options.strict
        )
        errors.extend(train_errors)
    if options.evaluation:
        eval_errors, report["evaluation"] = validate_records(
            eval_rows, plan, training=False, strict=options.strict
        )
        errors.extend(eval_errors)
    if options.train and options.evaluation:
        errors.extend(leakage_errors(train_rows, eval_rows))
    report["errors"] = errors
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

