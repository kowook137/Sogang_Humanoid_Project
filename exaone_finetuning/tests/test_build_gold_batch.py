import tempfile
import unittest
from pathlib import Path

from exaone_finetuning.build_gold_batch import DEFAULT_SOURCES, DEFAULT_SUPPLEMENT, build
from exaone_finetuning.generate_gold_candidates import clean


class GoldBatchTests(unittest.TestCase):
    def test_builds_exactly_one_hundred_unique_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = build(DEFAULT_SOURCES, DEFAULT_SUPPLEMENT, Path(directory) / "questions.jsonl")
        self.assertEqual(len(rows), 100)
        self.assertEqual(len({row["user"] for row in rows}), 100)

    def test_cleans_generation_prefix(self):
        self.assertEqual(clean("변환: 알겠습니더."), "알겠습니더.")


if __name__ == "__main__":
    unittest.main()
