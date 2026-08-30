"""Generate a 10-item Gyeongsang teacher pilot with the OpenAI Responses API."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

DEFAULT_MODEL = "gpt-5.5"
DIALECT = re.compile(
    r"(?:입니더|습니더|합니더|됩니더|아입니꺼|입니꺼|습니꺼|하이소|"
    r"보이소|주이소|드이소|마이소|네예|지예|나예|는교|은교|우째|묵)"
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

INSTRUCTIONS = """당신은 부산에서 성장한 한국어 대화 작가이자 데이터 검수자이다.
입력으로 사용자 질문과 정확한 표준어 존댓말 답변이 주어진다. 표준 답변의 의미, 사실,
숫자, 고유명사, 안전 조언, 문장 기능을 보존하면서 어르신에게 말하는 현대 부산·경남의
자연스러운 친근한 존댓말 후보 세 개를 작성하라.

필수 조건:
- 정보를 추가하거나 삭제하지 않는다.
- 질문을 평서문으로, 평서문을 질문으로 바꾸지 않는다.
- 사물이나 개념에 높임말을 붙이지 않는다.
- 표준어 문장 끝에 '네예'만 기계적으로 붙이지 않는다.
- '아이가', '데이'를 반복하는 과장된 방송식 사투리를 쓰지 않는다.
- 세 후보는 모두 자연스러워야 하며 표현은 서로 달라야 한다.
- candidate_1은 사투리 강도 약~중, candidate_2는 중, candidate_3은 중이되 더 구어체로 쓴다.
- 설명이나 평가 없이 후보만 반환한다.

자연스러운 방향 예시:
표준: 바로 일으키지 마세요. 먼저 의식과 호흡을 확인하세요.
방언: 바로 일으키지는 마이소. 먼저 의식하고 호흡부터 확인해 보이소.
표준: 현재 센서값이 없어 실내 온도를 알 수 없습니다.
방언: 지금 센서값이 없어가 실내 온도는 알 수 없습니더.
표준: 중요한 일 세 가지만 먼저 적어 보세요.
방언: 중요한 일 세 가지만 먼저 적어 보이소.
표준: 잘 모르겠습니다. 조금 더 자세히 말씀해 주세요.
방언: 잘 모르겠습니더. 쪼매만 더 자세히 말씀해 주이소.
표준: 많이 속상하셨겠어요. 천천히 말씀하셔도 괜찮아요.
방언: 마이 속상하셨겠네예. 천천히 말씀하셔도 괜찮아예.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_1": {"type": "string"},
        "candidate_2": {"type": "string"},
        "candidate_3": {"type": "string"},
    },
    "required": ["candidate_1", "candidate_2", "candidate_3"],
    "additionalProperties": False,
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_candidates(standard: str, candidates: list[str]) -> list[str]:
    errors = []
    candidates = [candidate.strip() for candidate in candidates]
    if len(candidates) != 3 or any(not candidate for candidate in candidates):
        errors.append("three_nonempty_candidates_required")
        return errors
    if len(set(candidates)) != 3:
        errors.append("duplicate_candidates")
    for index, candidate in enumerate(candidates, 1):
        if not DIALECT.search(candidate):
            errors.append(f"candidate_{index}_dialect_not_detected")
        if NUMBER.findall(candidate) != NUMBER.findall(standard):
            errors.append(f"candidate_{index}_number_changed")
        if LATIN.findall(candidate) != LATIN.findall(standard):
            errors.append(f"candidate_{index}_latin_changed")
    return errors


def request_candidates(client, model: str, user: str, standard: str) -> tuple[list[str], dict]:
    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=f"사용자 질문:\n{user}\n\n표준어 답변:\n{standard}",
        text={
            "format": {
                "type": "json_schema",
                "name": "gyeongsang_candidates",
                "strict": True,
                "schema": SCHEMA,
            }
        },
        store=False,
    )
    data = json.loads(response.output_text)
    candidates = [data[f"candidate_{index}"] for index in range(1, 4)]
    usage = getattr(response, "usage", None)
    usage_data = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else {}
    return candidates, usage_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY 환경변수가 없습니다. API 키를 코드나 Git에 저장하지 마세요.")
    from openai import OpenAI

    rows = read_jsonl(options.input)[: options.limit]
    done = set()
    for path in (options.output, options.rejected):
        if path.exists():
            done.update(row["id"] for row in read_jsonl(path))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.rejected.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    with options.output.open("a", encoding="utf-8") as good, options.rejected.open("a", encoding="utf-8") as bad:
        for index, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            candidates, usage = request_candidates(
                client, options.model, row["user"], row["standard_answer"]
            )
            errors = validate_candidates(row["standard_answer"], candidates)
            result = {
                "id": row["id"],
                "topic": row.get("topic", "general"),
                "user": row["user"],
                "standard_answer": row["standard_answer"],
                "candidates": candidates,
                "teacher_model": options.model,
                "usage": usage,
            }
            target = bad if errors else good
            if errors:
                result["validation_errors"] = errors
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            print(f"{index}/{len(rows)} {'REJECT ' + ','.join(errors) if errors else 'PASS'}", flush=True)


if __name__ == "__main__":
    main()
