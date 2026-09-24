from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from chat_session import (  # noqa: E402
    ChatSession,
    create_session_id,
    find_session,
    list_sessions,
    normalize_utf8,
)


class ChatSessionTests(unittest.TestCase):
    def test_session_id_contains_timestamp(self) -> None:
        session_id = create_session_id(datetime(2026, 8, 18, 1, 2, 3, tzinfo=timezone.utc))
        self.assertRegex(session_id, r"^20260818-010203-[0-9a-f]{8}$")

    def test_messages_are_logged_and_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = ChatSession(Path(directory), "system prompt", "test-session")
            session.add_message("user", "안녕하세요", False)
            session.add_message("assistant", "반갑습니다", False)
            records = [json.loads(line) for line in session.log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["role"] for record in records], ["user", "assistant"])
            self.assertEqual(records[0]["session_id"], "test-session")
            self.assertIn("timestamp", records[0])
            self.assertFalse(records[0]["reasoning_mode"])
            loaded = ChatSession.load(session.log_path, "new system prompt")
            self.assertEqual(loaded.session_id, "test-session")
            self.assertEqual(loaded.messages[0]["content"], "new system prompt")
            self.assertEqual(loaded.messages[1:], session.messages[1:])

    def test_session_lookup_and_listing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            first = ChatSession(log_dir, "prompt", "alpha")
            second = ChatSession(log_dir, "prompt", "beta")
            first.add_message("user", "one")
            second.add_message("user", "two")
            self.assertEqual(find_session(log_dir, "alp"), first.log_path)
            self.assertEqual(set(list_sessions(log_dir)), {first.log_path, second.log_path})

    def test_invalid_role_is_rejected_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = ChatSession(Path(directory), "prompt", "test")
            with self.assertRaises(ValueError):
                session.add_message("tool", "unsupported")
            self.assertFalse(session.log_path.exists())

    def test_surrogate_escaped_korean_input_is_recovered(self) -> None:
        broken_input = "현서".encode("utf-8").decode("ascii", errors="surrogateescape")
        self.assertEqual(normalize_utf8(broken_input), "현서")

        with tempfile.TemporaryDirectory() as directory:
            session = ChatSession(Path(directory), "prompt", "surrogate-input")
            session.add_message("user", broken_input)
            record = json.loads(session.log_path.read_text(encoding="utf-8"))
            self.assertEqual(record["content"], "현서")

    def test_dialect_is_logged_and_checked_when_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = ChatSession(
                Path(directory), "경상도 prompt", "dialect-test", dialect="gyeongsang"
            )
            session.add_message("user", "반갑습니다")
            record = json.loads(session.log_path.read_text(encoding="utf-8"))
            self.assertEqual(record["dialect"], "gyeongsang")

            loaded = ChatSession.load(
                session.log_path, "경상도 prompt", dialect="gyeongsang"
            )
            self.assertEqual(loaded.dialect, "gyeongsang")
            with self.assertRaisesRegex(ValueError, "Session dialect"):
                ChatSession.load(session.log_path, "전라도 prompt", dialect="jeolla")


if __name__ == "__main__":
    unittest.main()
