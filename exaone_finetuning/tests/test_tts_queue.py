import json
import tempfile
import unittest
from pathlib import Path

from exaone_finetuning.tts_queue import TTSQueue


class TTSQueueTests(unittest.TestCase):
    def test_writes_busan_prosody_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tts.jsonl"
            result = TTSQueue(path).enqueue(
                "안녕하세요.", session_id="session-1", style_changes=[]
            )
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved, result)
            self.assertEqual(saved["text"], "안녕하세요.")
            self.assertEqual(saved["prosody"]["accent"], "busan")
            self.assertEqual(saved["prosody"]["audience"], "elder")
            self.assertEqual(saved["status"], "pending")

    def test_rejects_empty_text(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                TTSQueue(Path(directory) / "tts.jsonl").enqueue(
                    " ", session_id="session-1"
                )


if __name__ == "__main__":
    unittest.main()
