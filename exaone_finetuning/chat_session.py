"""Conversation state and durable JSONL logging for the EXAONE chat CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _now() -> datetime:
    return datetime.now().astimezone()


def create_session_id(now: datetime | None = None) -> str:
    timestamp = now or _now()
    return f"{timestamp:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"


def normalize_utf8(text: str) -> str:
    """Recover surrogate-escaped terminal bytes and guarantee UTF-8-safe text."""
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        pass

    try:
        raw = text.encode("utf-8", errors="surrogateescape")
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="replace").decode("utf-8")
    return raw.decode("utf-8", errors="replace")


@dataclass
class ChatSession:
    log_dir: Path
    system_prompt: str
    session_id: str = field(default_factory=create_session_id)
    messages: list[dict[str, str]] = field(default_factory=list)
    dialect: str = "standard"

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)
        if not self.system_prompt.strip():
            raise ValueError("System prompt must not be empty")
        if not self.messages:
            self.messages.append({"role": "system", "content": self.system_prompt.strip()})

    @property
    def log_path(self) -> Path:
        return self.log_dir / f"{self.session_id}.jsonl"

    def add_message(self, role: str, content: str, reasoning_mode: bool = False) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")
        content = normalize_utf8(content).strip()
        if not content:
            raise ValueError("Message content must not be empty")
        message = {"role": role, "content": content}
        self.messages.append(message)
        self._append_log(message, reasoning_mode)

    def _append_log(self, message: dict[str, str], reasoning_mode: bool) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "timestamp": _now().isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "role": message["role"],
            "content": message["content"],
            "reasoning_mode": reasoning_mode,
            "dialect": self.dialect,
        }
        # Serialize completely before opening the log so an encoding/serialization
        # failure cannot leave a half-written JSON object behind.
        serialized = json.dumps(record, ensure_ascii=False)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(serialized + "\n")

    def display_history(self) -> str:
        visible = self.messages[1:]
        if not visible:
            return "(아직 대화가 없습니다.)"
        labels = {"user": "User", "assistant": "AI"}
        return "\n".join(
            f"{labels.get(message['role'], message['role'])}: {message['content']}"
            for message in visible
        )

    @classmethod
    def load(
        cls, path: Path, system_prompt: str, dialect: str = "standard"
    ) -> "ChatSession":
        path = Path(path)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt.strip()}
        ]
        session_id: str | None = None
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {path}") from error
                role = record.get("role")
                content = record.get("content")
                if role not in {"user", "assistant"} or not isinstance(content, str):
                    raise ValueError(f"Invalid message at line {line_number}: {path}")
                record_session_id = record.get("session_id")
                if not isinstance(record_session_id, str) or not record_session_id:
                    raise ValueError(f"Missing session_id at line {line_number}: {path}")
                if session_id is not None and record_session_id != session_id:
                    raise ValueError(f"Mixed session IDs in log: {path}")
                record_dialect = record.get("dialect")
                if record_dialect is not None and record_dialect != dialect:
                    raise ValueError(
                        f"Session dialect is {record_dialect}, not {dialect}: {path}"
                    )
                session_id = record_session_id
                messages.append({"role": role, "content": normalize_utf8(content)})
        if session_id is None:
            raise ValueError(f"Session log is empty: {path}")
        return cls(
            path.parent,
            system_prompt,
            session_id=session_id,
            messages=messages,
            dialect=dialect,
        )


def find_session(log_dir: Path, session_name: str) -> Path:
    """Resolve an exact session ID or an unambiguous session ID prefix."""
    log_dir = Path(log_dir)
    exact = log_dir / f"{session_name.removesuffix('.jsonl')}.jsonl"
    if exact.is_file():
        return exact
    matches = sorted(log_dir.glob(f"{session_name}*.jsonl")) if log_dir.is_dir() else []
    if not matches:
        raise FileNotFoundError(f"Session not found: {session_name}")
    if len(matches) > 1:
        raise ValueError(f"Session prefix is ambiguous: {session_name}")
    return matches[0]


def list_sessions(log_dir: Path) -> list[Path]:
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        return []
    return sorted(log_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
