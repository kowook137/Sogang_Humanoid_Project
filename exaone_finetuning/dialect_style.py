"""Conservative, deterministic dialect styling for production responses."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


GYEONGSANG_MARKERS = re.compile(
    r"(?:습니더|입니더|하이소|보이소|주이소|게예|네예|는가예|인가예)"
)
INFORMAL_PATTERNS = (
    re.compile(r"(?:^|[\s,])너(?:는|가|를|도|한테|랑|와)?(?:[\s,.!?]|$)"),
    re.compile(r"(?:해봐|말해봐|할까\?|거야[.!?]|하면 돼[.!?])"),
)

REWRITE_SYSTEM_PROMPT = """당신은 내용 생성자가 아니라 말투 변환기입니다.
입력된 한국어 답변의 뜻, 사실, 숫자, 고유명사, 목록 구조를 그대로 유지하면서
전체 답변을 현대 부산·경남의 친근한 존댓말로 바꾸세요. 어르신께 말하듯
`~입니더`, `~하이소`, `~보이소`, `~주이소`, `~네예`, `~실래예`를 문맥에 맞게
섞되 같은 어미를 연속해서 반복하지 마세요. 표준어 문장 종결을 가능한 한
그대로 남기지 말고, 방송·개그식 표현이나 반말은 쓰지 마세요. 설명, 머리말,
따옴표 없이 변환된 답변 본문만 출력하세요.

입력: 걱정이 많으셔서 힘드시겠어요. 잠들기 전에 조용한 음악을 들어 보세요.
출력: 걱정이 많으셔서 힘드시겠네예. 잠들기 전에 조용한 음악을 들어 보이소.

입력: 사진 앱을 열고 보낼 사진을 선택하세요. 전송 버튼을 누르면 됩니다.
출력: 사진 앱을 열고 보낼 사진을 선택하이소. 전송 버튼을 누르면 됩니더."""


def has_informal_speech(text: str) -> bool:
    """Detect high-risk informal forms that are inappropriate for an elder."""
    return any(pattern.search(text) for pattern in INFORMAL_PATTERNS)


def build_rewrite_messages(text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": text.strip()},
    ]


def clean_rewrite(text: str) -> str:
    text = text.strip().removeprefix("```text").removeprefix("```")
    text = text.removesuffix("```").strip()
    return re.sub(r"^(?:출력|변환(?:된 답변)?)[：:]\s*", "", text).strip()


def rewrite_is_safe(source: str, candidate: str) -> bool:
    """Reject obvious truncation or corruption before displaying a rewrite."""
    candidate = clean_rewrite(candidate)
    if not candidate:
        return False
    source_compact = re.sub(r"\s+", "", source)
    candidate_compact = re.sub(r"\s+", "", candidate)
    ratio = len(candidate_compact) / max(1, len(source_compact))
    if not 0.65 <= ratio <= 1.35:
        return False
    if re.findall(r"\d+(?:[.,]\d+)*", source) != re.findall(
        r"\d+(?:[.,]\d+)*", candidate
    ):
        return False
    source_latin = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", source)
    candidate_latin = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", candidate)
    if source_latin != candidate_latin:
        return False
    return SequenceMatcher(None, source_compact, candidate_compact).ratio() >= 0.45


def _replace_once(text: str, pattern: str, replacement: str) -> tuple[str, bool]:
    updated, count = re.subn(pattern, replacement, text, count=1)
    return updated, bool(count)


def style_gyeongsang(text: str) -> str:
    """Apply one or two reliable Busan/Gyeongnam honorific endings.

    The transformation deliberately stays conservative: standard polite Korean is
    allowed, while fabricated vocabulary and aggressive word-by-word rewriting are
    avoided. Safety facts, numbers, and names are never changed.
    """
    text = text.strip()
    if not text:
        return text

    replacements = (
        (r"말씀해\s*주세요(?=\s*[.!?]|$)", "말씀해 주이소"),
        (r"확인해\s*보세요(?=\s*[.!?]|$)", "확인해 보이소"),
        (r"해\s*보세요(?=\s*[.!?]|$)", "해 보이소"),
        (r"도와주세요(?=\s*[.!?]|$)", "도와주이소"),
        (r"하세요(?=\s*[.!?]|$)", "하이소"),
        (r"시겠어요(?=\s*\?|$)", "실래예"),
        (r"드릴게요(?=\s*[.!?]|$)", "드릴게예"),
        (r"할게요(?=\s*[.!?]|$)", "할게예"),
        (r"어떨까요(?=\s*\?|$)", "어떠신가예"),
        (r"좋겠네요(?=\s*[.!?]|$)", "좋겠네예"),
        (r"네요(?=\s*[.!?]|$)", "네예"),
        (r"거예요(?=\s*[.!?]|$)", "겁니더"),
        (r"입니다(?=\s*[.!]|$)", "입니더"),
        (r"습니다(?=\s*[.!]|$)", "습니더"),
        (r"인가요(?=\s*\?|$)", "인가예"),
        (r"나요(?=\s*\?|$)", "는가예"),
    )
    changes = 0
    for pattern, replacement in replacements:
        while changes < 3:
            text, changed = _replace_once(text, pattern, replacement)
            if not changed:
                break
            changes += 1
        if changes >= 3:
            break

    # If no safe inflection is available, preserve the generated answer and add a
    # short, respectful closing rather than corrupting its morphology.
    if changes or GYEONGSANG_MARKERS.search(text):
        return text
    return f"{text}\n필요하시면 편하게 말씀해 주이소."
