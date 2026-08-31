"""Generate review-only Gyeongsang candidates with Vertex AI Gemini."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_PROJECT = "project-a965291e-1224-4ea7-bf1"
DIALECT = re.compile(
    r"(?:입니더|습니더|합니더|됩니더|심더|낍니더|니데이|입니꺼|습니꺼|"
    r"하이소|보이소|주이소|드이소|마이소|네예|지예|나예|는교|은교|능교|"
    r"우째|묵|아니모|해가|라믄|카모)"
)
OVERDONE = re.compile(
    r"(?:안녕하십니꺼|참말로|쪼매쪼매|아입니꺼|말입니더|"
    r"라카믄|할라카믄|끼라예|가꼬|우예예|어떠세요예|어떠예|"
    r"깁니더|심니더|다카이)"
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

ANSWER_INSTRUCTION = """당신은 어르신과 대화하는 정확하고 친절한 한국어 AI이다.
사용자 질문에 실질적으로 도움이 되는 완성된 표준어 존댓말 답변을 작성하라.

필수 조건:
- 모르는 현재 정보, 날씨, 센서값을 추측하지 않는다.
- 위치나 취향을 몰라도 일반적으로 가능한 추천은 제공한다.
- 응급 상황에서는 즉시 할 행동과 신고 기준을 정확히 안내한다.
- 감정 질문에는 공감만 하고 끝내지 말고 부담 없는 구체적인 행동을 하나 이상 제안한다.
- 질문을 회피하거나 단순히 사용자의 말을 반복하지 않는다.
- 안전상 상세 설명이 필요한 경우가 아니면 1~4문장으로 간결하게 쓴다.
"""

REWRITE_INSTRUCTION = """당신은 부산에서 성장한 한국어 대화 작가이자 데이터 검수자이다.
제공된 표준어 답변을 현대 부산·경남 존댓말 후보 세 개로 변환하라. 새 답변을 다시
작성하는 작업이 아니라 말투만 변환하는 작업이다. 실제 부산의 젊은 성인이 어르신에게
편안하게 말하는 정도로 쓰고, 모든 문장을 사투리로 꾸미지 마라.

필수 조건:
- 표준어 답변의 사실, 숫자, 고유명사, 조언, 문장 기능을 그대로 유지한다.
- 정보를 추가하거나 삭제하지 않고, 조언의 강도와 우선순위를 바꾸지 않는다.
- 원문에 없는 '제일', '엄청', '확', '무조건' 같은 강도를 추가하지 않는다.
- 사물이나 개념에 높임말을 붙이지 않는다.
- 표준어 문장 끝에 '네예'만 기계적으로 붙이지 않는다.
- '아이가', '데이'를 반복하는 과장된 방송식 사투리를 쓰지 않는다.
- '안녕하십니꺼', '참말로', '쪼매쪼매', '아입니꺼', '말입니더', '라카믄',
  '할라카믄', '끼라예', '가꼬', '우예예', '어떠세요예', '깁니더', '심니더',
  '다카이' 같은 연출되거나 어색한 표현은 쓰지 않는다.
- 한 문단에서 눈에 띄는 방언 어미는 두세 번 이내로 제한하고 나머지는 자연스러운
  한국어 존댓말을 사용한다.
- 세 후보의 의미는 같아야 하지만 표현은 서로 달라야 한다.
- candidate_1은 약, candidate_2는 약~중, candidate_3은 중 정도로 쓰되 어느 후보도
  방송식 사투리나 옛날 말투처럼 들리지 않게 한다.

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

ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {"standard_answer": {"type": "STRING"}},
    "required": ["standard_answer"],
}

CANDIDATE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "candidate_1": {"type": "STRING"},
        "candidate_2": {"type": "STRING"},
        "candidate_3": {"type": "STRING"},
    },
    "required": ["candidate_1", "candidate_2", "candidate_3"],
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def input_messages(row: dict) -> list[dict]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    return [{"role": "user", "content": row["user"]}]


def final_user_message(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    raise ValueError("conversation requires at least one user message")


def conversation_prompt(messages: list[dict]) -> str:
    labels = {"system": "시스템", "user": "사용자", "assistant": "AI"}
    return "\n".join(
        f"{labels.get(message.get('role'), message.get('role'))}: "
        f"{str(message.get('content', '')).strip()}"
        for message in messages
    )


def validate_candidate(standard: str, candidate: str) -> list[str]:
    errors = []
    candidate = candidate.strip()
    if not candidate:
        return ["empty"]
    if not DIALECT.search(candidate):
        errors.append("dialect_not_detected")
    if OVERDONE.search(candidate):
        errors.append("overdone_style")
    if NUMBER.findall(candidate) != NUMBER.findall(standard):
        errors.append("number_changed")
    if LATIN.findall(candidate) != LATIN.findall(standard):
        errors.append("latin_changed")
    return errors


def validate_candidates(standard: str, candidates: list[str]) -> list[str]:
    values = [candidate.strip() for candidate in candidates]
    if len(values) != 3 or any(not candidate for candidate in values):
        return ["three_nonempty_candidates_required"]
    errors = ["duplicate_candidates"] if len(set(values)) != 3 else []
    for index, candidate in enumerate(values, 1):
        errors.extend(
            f"candidate_{index}_{error}"
            for error in validate_candidate(standard, candidate)
        )
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
            messages = input_messages(row)
            user = final_user_message(messages)
            context = conversation_prompt(messages)
            answer_response = client.models.generate_content(
                model=options.model,
                contents=(
                    "다음 대화의 마지막 사용자 발화에 답하세요. 이전 대화가 있으면 "
                    f"그 문맥과 수정 사항을 반영하세요.\n\n{context}"
                ),
                config=GenerateContentConfig(
                    system_instruction=ANSWER_INSTRUCTION,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=ANSWER_SCHEMA,
                ),
            )
            standard = json.loads(answer_response.text)["standard_answer"].strip()
            rewrite_response = client.models.generate_content(
                model=options.model,
                contents=(
                    f"대화 문맥:\n{context}\n\n"
                    f"변환할 표준어 답변:\n{standard}"
                ),
                config=GenerateContentConfig(
                    system_instruction=REWRITE_INSTRUCTION,
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=CANDIDATE_SCHEMA,
                ),
            )
            data = json.loads(rewrite_response.text)
            candidates = [data[f"candidate_{number}"].strip() for number in range(1, 4)]
            errors = validate_candidates(standard, candidates)
            candidate_validation = []
            for candidate_index, candidate in enumerate(candidates, 1):
                candidate_errors = validate_candidate(standard, candidate)
                candidate_validation.append(
                    {
                        "candidate": candidate_index,
                        "usable": not candidate_errors,
                        "errors": candidate_errors,
                    }
                )
            usable_count = sum(item["usable"] for item in candidate_validation)
            answer_usage = getattr(answer_response, "usage_metadata", None)
            rewrite_usage = getattr(rewrite_response, "usage_metadata", None)
            usage_data = {
                "answer": answer_usage.model_dump()
                if answer_usage is not None and hasattr(answer_usage, "model_dump")
                else {},
                "rewrite": rewrite_usage.model_dump()
                if rewrite_usage is not None and hasattr(rewrite_usage, "model_dump")
                else {},
            }
            result = {
                "id": row["id"],
                "topic": row.get("topic", row.get("category", "general")),
                "category": row.get("category", row.get("topic", "general")),
                "input_style": row.get("input_style", "standard_korean"),
                "user": user,
                "context_messages": messages,
                "standard_answer": standard,
                "candidates": candidates,
                "candidate_validation": candidate_validation,
                "usable_candidate_count": usable_count,
                "teacher_model": options.model,
                "usage": usage_data,
            }
            target = good if usable_count else bad
            if errors:
                result["validation_warnings"] = errors
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
            target.flush()
            status = "PASS" if usable_count else "REJECT"
            print(
                f"{index}/{len(rows)} {status} usable={usable_count}/3",
                flush=True,
            )


if __name__ == "__main__":
    main()
