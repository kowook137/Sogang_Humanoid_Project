"""Generate review-only Gyeongsang candidates with Vertex AI Gemini."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_PROJECT = "project-a965291e-1224-4ea7-bf1"
DIALECT = re.compile(
    r"(?:입니더|습니더|합니더|됩니더|아입니꺼|입니꺼|습니꺼|하이소|"
    r"보이소|주이소|드이소|마이소|네예|지예|나예|는교|은교|우째|묵)"
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

SYSTEM_INSTRUCTION = """당신은 부산에서 성장한 한국어 대화 작가이자 데이터 검수자이다.
사용자 질문에 대한 정확하고 친절한 표준어 존댓말 답변 하나를 먼저 작성하고, 그 답변의
의미와 안전 정보를 유지한 현대 부산·경남 존댓말 후보 세 개를 작성하라.

필수 조건:
- 모르는 현재 정보, 날씨, 센서값을 추측하지 않는다.
- 응급 상황에서는 사투리보다 정확한 안전 안내를 우선한다.
- 원래 답변의 사실, 숫자, 고유명사, 조언과 질문 여부를 바꾸지 않는다.
- 새로운 정보를 추가하거나 기존 정보를 삭제하지 않는다.
- 사물이나 개념에 높임말을 붙이지 않는다.
- 표준어 문장 끝에 '네예'만 기계적으로 붙이지 않는다.
- '아이가', '데이'를 반복하는 과장된 방송식 사투리를 쓰지 않는다.
- 세 후보의 의미는 같아야 하지만 표현은 서로 달라야 한다.
- candidate_1은 약~중, candidate_2는 중, candidate_3은 중 정도의 자연스러운 구어체로 쓴다.

예시:
표준: 바로 일으키지 마세요. 먼저 의식과 호흡을 확인하세요.
방언: 바로 일으키지는 마이소. 먼저 의식하고 호흡부터 확인해 보이소.
표준: 현재 센서값이 없어 실내 온도를 알 수 없습니다.
방언: 지금 센서값이 없어가 실내 온도는 알 수 없습니더.
표준: 중요한 일 세 가지만 먼저 적어 보세요.
방언: 중요한 일 세 가지만 먼저 적어 보이소.
표준: 잘 모르겠습니다. 조금 더 자세히 말씀해 주세요.
방언: 잘 모르겠습니더. 쪼매만 더 자세히 말씀해 주이소.
"""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "standard_answer": {"type": "STRING"},
        "candidate_1": {"type": "STRING"},
        "candidate_2": {"type": "STRING"},
        "candidate_3": {"type": "STRING"},
    },
    "required": ["standard_answer", "candidate_1", "candidate_2", "candidate_3"],
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_candidates(standard: str, candidates: list[str]) -> list[str]:
    errors = []
    values = [candidate.strip() for candidate in candidates]
    if len(values) != 3 or any(not candidate for candidate in values):
        return ["three_nonempty_candidates_required"]
    if len(set(values)) != 3:
        errors.append("duplicate_candidates")
    for index, candidate in enumerate(values, 1):
        if not DIALECT.search(candidate):
            errors.append(f"candidate_{index}_dialect_not_detected")
        if NUMBER.findall(candidate) != NUMBER.findall(standard):
            errors.append(f"candidate_{index}_number_changed")
        if LATIN.findall(candidate) != LATIN.findall(standard):
            errors.append(f"candidate_{index}_latin_changed")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    from google import genai
    from google.genai.types import GenerateContentConfig, HttpOptions

    rows = read_jsonl(options.input)[: options.limit]
    done = set()
    for path in (options.output, options.rejected):
        if path.exists():
            done.update(row["id"] for row in read_jsonl(path))
    client = genai.Client(
        vertexai=True,
        project=options.project,
        location=options.location,
        http_options=HttpOptions(api_version="v1"),
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.rejected.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("a", encoding="utf-8") as good, options.rejected.open("a", encoding="utf-8") as bad:
        for index, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            response = client.models.generate_content(
                model=options.model,
                contents=f"사용자 질문:\n{row['user']}",
                config=GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=SCHEMA,
                ),
            )
            data = json.loads(response.text)
            standard = data["standard_answer"].strip()
            candidates = [data[f"candidate_{number}"].strip() for number in range(1, 4)]
            errors = validate_candidates(standard, candidates)
            usage = getattr(response, "usage_metadata", None)
            usage_data = usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else {}
            result = {
                "id": row["id"],
                "topic": row.get("topic", "general"),
                "user": row["user"],
                "standard_answer": standard,
                "candidates": candidates,
                "teacher_model": options.model,
                "usage": usage_data,
            }
            target = bad if errors else good
            if errors:
                result["validation_errors"] = errors
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            print(f"{index}/{len(rows)} {'REJECT ' + ','.join(errors) if errors else 'PASS'}", flush=True)


if __name__ == "__main__":
    main()
