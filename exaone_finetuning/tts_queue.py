"""JSONL handoff from dialogue generation to a future Busan-prosody TTS worker."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4


class TTSQueue:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def enqueue(
        self,
        text: str,
        *,
        session_id: str,
        style_changes: list[dict[str, str]] | None = None,
    ) -> dict:
        if not text.strip():
            raise ValueError("TTS text must not be empty")
        record = {
            "utterance_id": uuid4().hex,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "session_id": session_id,
            "text": text.strip(),
            "language": "ko-KR",
            "voice_style": "busan_contemporary_gentle_elder",
            "prosody": {
                "accent": "busan",
                "strength": "gentle",
                "audience": "elder",
            },
            "text_style_changes": style_changes or [],
            "status": "pending",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
