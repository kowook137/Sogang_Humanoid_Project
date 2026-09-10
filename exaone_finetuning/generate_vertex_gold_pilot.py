"""Generate review-only modern Busan candidates with Vertex AI Gemini."""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

DEFAULT_PROJECT = "project-a965291e-1224-4ea7-bf1"
DIALECT = re.compile(
    r"(?:니더|심더|니데이|입니꺼|습니꺼|이소|마이소|네예|지예|나예|"
    r"는데예|고예|라꼬|다꼬|는교|은교|능교|우째|묵|아니모|해가|라믄|카모)"
)
OVERDONE = re.compile(
    r"(?:안녕하십니꺼|참말로|쪼매쪼매|아입니꺼|말입니더|"
    r"라카믄|할라카믄|끼라예|가꼬|우예예|어떠세요예|어떠예|"
    r"깁니더|낍니더|심니더|다카이|알겠심|드맀|일나(?:실|시|서)|"
    r"여이소|보세예|챙기놓|챙기두|으이소|부드러버|설채서|신교|"
    r"갑네요|이라가|올리이소|보이시면|계셔보이소|있습니다예|"
    r"골라둬|겠어예|니더|심더)"
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
DIALECT_FEATURE = re.compile(
    r"(?:니더|심더|니데이|입니꺼|습니꺼|이소|마이소|네예|지예|나예|"
    r"는데예|고예|라꼬|다꼬|는교|은교|능교|우째|묵|아니모|해가|라믄|카모)"
)
ADDED_INTENSIFIER = re.compile(r"(?:제일|엄청|무조건|진짜|딴 거보다|쪼매|확실히)")
NEGATION = re.compile(r"(?:안 |않|못 |마세요|마이소|아니|없|금지|절대)")
CRITICAL_TERMS = ("119", "112", "의식", "호흡", "가슴 압박", "신고", "대피")


def latin_terms(text: str) -> list[str]:
    """Compare Latin terms while accepting common Korean spellings of the same term."""
    normalized = text.replace("텔레비전", "TV").replace("티브이", "TV")
    return [token.upper() for token in LATIN.findall(normalized)]


def compact_similarity(source: str, candidate: str) -> float:
    source = re.sub(r"\s+", "", source)
    candidate = re.sub(r"\s+", "", candidate)
    return SequenceMatcher(None, source, candidate).ratio()

ANSWER_INSTRUCTION = """당신은 어르신과 대화하는 정확하고 친절한 한국어 AI이다.
사용자 질문에 실질적으로 도움이 되는 완성된 표준어 존댓말 답변을 작성하라.

필수 조건:
- 모르는 현재 정보, 날씨, 센서값을 추측하지 않는다.
- 위치나 취향을 몰라도 일반적으로 가능한 추천은 제공한다.
- 응급 상황에서는 즉시 할 행동과 신고 기준을 정확히 안내한다.
- 감정 질문에는 공감만 하고 끝내지 말고 부담 없는 구체적인 행동을 하나 이상 제안한다.
- 약을 먹었는지 확실하지 않은 상황에서는 임의로 추가 복용하게 하지 않는다. 약마다
  놓친 복용량의 처리법이 다르므로 처방전·복약안내서를 확인하고 의사나 약사에게
  문의하도록 안내한다.
- 질문을 회피하거나 단순히 사용자의 말을 반복하지 않는다.
- 안전상 상세 설명이 필요한 경우가 아니면 1~4문장으로 간결하게 쓴다.
"""

REWRITE_INSTRUCTION = """당신은 부산에서 성장했고 현재도 부산말을 자연스럽게 사용하는
한국어 대화 작가이자 데이터 검수자이다. 제공된 표준어 답변을 현대 부산 존댓말 후보
세 개로 변환하라. 넓은 '경상도 사투리'나 대구·경북 말투를 섞는 작업이 아니라 부산에서
실제로 쓰는 말투로 좁혀 변환하는 작업이다. 새 답변을 작성하지 말고 말투만 변환하라.
부산의 젊은 성인이 어르신에게 편안하게 말하는 정도로 쓰고, 모든 문장을 사투리로
꾸미지 마라.

생성 원칙은 보수적이다. AI-Hub 부산 화자의 실제 발화에서 문장 기능과 쓰임이 확인된
부산 표현만 사용한다. 확실하지 않은 문장은 변환하지 않고 표준어 존댓말로 둔다.
텍스트가 표준어에 가깝더라도 이후 부산 화자의 억양과 음색은 TTS 단계에서 별도로
구현하므로, 지역성을 보이려고 불확실한 어미를 만들어 내지 마라.

이번 30문장 파일럿에서 출력에 허용하는 부산 종결형은 '-네예/-지예'뿐이다. 다음은
AI-Hub 원문에서 출생지·주 거주지·현재 거주지가 모두 부산인 20~30대 화자의 발화이며,
해당 어절이 isDialect=true로 표시된 근거 예시다.
- DKSR20003393.1.1.159: 거짓말이겠지요. → 거짓말이겠지예.
- DKSR20002654.1.1.1: 그렇지요? → 그지예?
- DKCI20000122.1.1.86: 맛있겠네요. → 맛있겠네예.
- 위 문장 기능과 자연스럽게 대응하지 않으면 표준어 존댓말을 그대로 출력한다.
- '-노/-나'는 사용자 입력을 이해할 때만 참고하고 어르신 상대 AI 출력에는 쓰지 않는다.
- '-이소/-하이소/-마이소', '-데이', '-니더/-심더'는 이번 파일럿 출력에 쓰지 않는다.

필수 조건:
- 표준어 답변의 사실, 숫자, 고유명사, 조언, 문장 기능을 그대로 유지한다.
- 정보를 추가하거나 삭제하지 않고, 조언의 강도와 우선순위를 바꾸지 않는다.
- '무엇→뭐', '가장→제일', '-습니까→-으세요'처럼 부산말과 무관한 단순 구어체
  치환을 하지 않는다. 부산 표지가 필요 없으면 원문 표현을 그대로 둔다.
- '맞나요?'를 '맞지예?'로 바꾸는 것처럼 불확실 질문을 확인·동의 요구로 바꾸지 않는다.
- 추측, 확신, 가능성, 의문, 명령의 정도를 원문과 완전히 동일하게 유지한다.
- 원문에 없는 '제일', '엄청', '확', '무조건' 같은 강도를 추가하지 않는다.
- 사물이나 개념에 높임말을 붙이지 않는다.
- 표준어 문장 끝에 '네예'만 기계적으로 붙이지 않는다.
- '습니더/입니더', '보이소', '해가' 중 하나를 문단 전체에 반복해서 붙이지 않는다.
- '될낍니더', '알겠심더', '드맀습니다', '일나실', '여이소', '보세예',
  '챙기놓으면'처럼 축약이 어색하거나 철자가 무너진 표현은 쓰지 않는다.
- 후보마다 눈에 띄는 방언 표지는 전체 합계 세 번 이하로 사용한다. 약한 후보는
  한두 군데만 자연스럽게 바꾼다.
- '아이가', '데이'를 반복하는 과장된 방송식 사투리를 쓰지 않는다.
- 부산 여부가 불확실한 표현은 억지로 넣지 말고 자연스러운 표준어 존댓말을 유지한다.
- 평범한 정보 설명은 표준어 존댓말을 중심으로 쓰고 부산 표지는 한두 곳만 둔다.
- 공감·확인에는 문맥에 맞을 때만 '-네예/-지예'를 쓴다.
- '-예' 계열은 실제 공감·확인·부드러운 응답에만 쓰며 문장마다 붙이지 않는다.
- '-노/-나' 계열은 실제 부산 발화에 존재하지만 주로 비존대 의문형이다. 어르신에게
  답하는 AI의 존댓말에는 사용하지 않는다. 사용자가 부산말로 질문한 입력은 그대로
  이해하되 AI 답변에 기계적으로 따라 쓰지 않는다.
- '-보이소/-하이소/-마이소'와 '-데이'는 후속 검증 전까지 사용하지 않는다.
- 현재 목표 화자의 검수에서 '-니더/-습니더'가 현대 부산 존댓말로 자연스럽지 않다고
  판정되었으므로 사용하지 않는다.
- 평체로 들리는 절과 존댓말을 한 문장에 섞지 않는다. 예를 들어 '바람이 많이 분데이,
  따뜻하게 입으세요'처럼 높임 수준이 흔들리는 문장은 쓰지 않는다.
- '안녕하십니꺼', '참말로', '쪼매쪼매', '아입니꺼', '말입니더', '라카믄',
  '할라카믄', '끼라예', '가꼬', '우예예', '어떠세요예', '깁니더', '심니더',
  '다카이' 같은 연출되거나 어색한 표현은 쓰지 않는다.
- 한 문단에서 눈에 띄는 방언 어미는 두세 번 이내로 제한하고 나머지는 자연스러운
  한국어 존댓말을 사용한다.
- 세 후보의 의미는 같아야 하지만 표현은 서로 달라야 한다.
- candidate_1은 원문을 그대로 쓴다. candidate_2와 candidate_3은 문장 기능상
  '-네예/-지예'가 자연스러울 때만 한 번 사용하고, 그렇지 않으면 원문과 의미가 같은
  표준어 존댓말 변형을 쓴다. 후보 번호가 커진다고 사투리를 더 세게 만들지 않는다.

예시:
표준: 바로 일으키지 마세요. 먼저 의식과 호흡을 확인하세요.
방언: 바로 일으키지 마세요. 먼저 의식과 호흡을 확인하세요.
표준: 현재 센서값이 없어 실내 온도를 알 수 없습니다.
방언: 현재 센서값이 없어 실내 온도를 알 수 없습니다.
표준: 중요한 일 세 가지만 먼저 적어 보세요.
방언: 중요한 일 세 가지만 먼저 적어 보세요.
표준: 잘 모르겠습니다. 조금 더 자세히 말씀해 주세요.
방언: 잘 모르겠습니다. 조금 더 자세히 말씀해 주세요.
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


def validate_candidate(
    standard: str, candidate: str, *, allow_standard: bool = False
) -> list[str]:
    errors = []
    candidate = candidate.strip()
    if not candidate:
        return ["empty"]
    if not allow_standard and not DIALECT.search(candidate):
        errors.append("dialect_not_detected")
    if OVERDONE.search(candidate):
        errors.append("overdone_style")
    if NUMBER.findall(candidate) != NUMBER.findall(standard):
        errors.append("number_changed")
    if latin_terms(candidate) != latin_terms(standard):
        errors.append("latin_changed")
    source_length = len(re.sub(r"\s+", "", standard))
    candidate_length = len(re.sub(r"\s+", "", candidate))
    ratio = candidate_length / max(1, source_length)
    minimum_ratio, maximum_ratio = ((0.60, 1.60) if source_length < 30 else (0.78, 1.22))
    if not minimum_ratio <= ratio <= maximum_ratio:
        errors.append("length_changed")
    if compact_similarity(standard, candidate) < 0.45:
        errors.append("meaning_drift_risk")
    if len(DIALECT_FEATURE.findall(candidate)) > 3:
        errors.append("dialect_overused")
    if ADDED_INTENSIFIER.search(candidate) and not ADDED_INTENSIFIER.search(standard):
        errors.append("intensifier_added")
    if bool(NEGATION.search(standard)) != bool(NEGATION.search(candidate)):
        errors.append("negation_changed")
    if standard.count("?") != candidate.count("?"):
        errors.append("sentence_function_changed")
    for term in CRITICAL_TERMS:
        if (term in standard) != (term in candidate):
            errors.append(f"critical_term_changed:{term}")
    return errors


def validate_candidates(standard: str, candidates: list[str]) -> list[str]:
    values = [candidate.strip() for candidate in candidates]
    if len(values) != 3 or any(not candidate for candidate in values):
        return ["three_nonempty_candidates_required"]
    errors = ["duplicate_candidates"] if len(set(values)) != 3 else []
    for index, candidate in enumerate(values, 1):
        errors.extend(
            f"candidate_{index}_{error}"
            for error in validate_candidate(standard, candidate, allow_standard=True)
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
                "target_locale": "busan_modern_polite",
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
