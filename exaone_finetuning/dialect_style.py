"""Conservative, deterministic dialect styling for production responses."""

from __future__ import annotations

import re


GYEONGSANG_MARKERS = re.compile(
    r"(?:습니더|입니더|하이소|보이소|주이소|게예|네예|는가예|인가예)"
)
INFORMAL_PATTERNS = (
    re.compile(r"(?:^|[\s,])너(?:는|가|를|도|한테|랑|와)?(?:[\s,.!?]|$)"),
    re.compile(r"(?:해봐|말해봐|할까\?|거야[.!?]|하면 돼[.!?])"),
)


def has_informal_speech(text: str) -> bool:
    """Detect high-risk informal forms that are inappropriate for an elder."""
    return any(pattern.search(text) for pattern in INFORMAL_PATTERNS)


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
        (r"드릴게요(?=\s*[.!?]|$)", "드릴게예"),
        (r"할게요(?=\s*[.!?]|$)", "할게예"),
        (r"어떨까요(?=\s*\?|$)", "어떠신가예"),
        (r"좋겠네요(?=\s*[.!?]|$)", "좋겠네예"),
        (r"겠어요(?=\s*[.!?]|$)", "겠네예"),
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
