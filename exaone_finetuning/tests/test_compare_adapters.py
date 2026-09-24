import json
import tempfile
import unittest
from pathlib import Path

from exaone_finetuning.compare_adapters import load_questions, parse_adapter


class CompareAdaptersTest(unittest.TestCase):
    def test_parse_adapter(self):
        label, path = parse_adapter("v2=~/adapter")
        self.assertEqual(label, "v2")
        self.assertEqual(path, Path("~/adapter").expanduser())

    def test_rejects_invalid_adapter(self):
        with self.assertRaisesRegex(ValueError, "LABEL=PATH"):
            parse_adapter("adapter-only")

    def test_questions_must_end_with_user(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "bad",
                        "messages": [{"role": "assistant", "content": "끝"}],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "end with user"):
                load_questions(path)


if __name__ == "__main__":
    unittest.main()
