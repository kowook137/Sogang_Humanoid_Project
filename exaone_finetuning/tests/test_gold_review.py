import csv
import json
import tempfile
import unittest
from pathlib import Path

from exaone_finetuning.gold_review import (
    FIELDS,
    export,
    normalize_candidate,
    prepare,
    read_review_csv,
    validate,
)


class GoldReviewTests(unittest.TestCase):
    def test_normalizes_legacy_v6_record(self):
        row = normalize_candidate(
            {"id": "a", "dialect_non_polite": "밥 묵었나?", "dialect_polite": "밥 묵으셨나예?"}
        )
        self.assertEqual(row["standard_answer"], "밥 묵었나?")
        self.assertEqual(row["candidate_1"], "밥 묵으셨나예?")

    def test_prepare_writes_excel_compatible_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                json.dumps({"id": "a", "user": "안녕하세요", "standard_answer": "반갑습니다", "candidates": ["반갑습니더"]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output = root / "review.csv"
            prepare(source, output, 100)
            self.assertEqual(read_review_csv(output)[0]["candidate_1"], "반갑습니더")

    def test_validation_rejects_empty_selected_candidate(self):
        row = {field: "" for field in FIELDS}
        row.update({"id": "a", "user": "질문", "decision": "accept_2"})
        self.assertTrue(validate([row], require_complete=True))

    def test_export_creates_sft_and_preferences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for number in range(20):
                row = {field: "" for field in FIELDS}
                row.update(
                    {
                        "id": f"item-{number}",
                        "topic": "daily",
                        "user": "오늘 기분이 어때요?",
                        "standard_answer": "기분이 좋습니다.",
                        "candidate_1": "기분 좋습니더.",
                        "candidate_2": "기분 좋아예.",
                        "decision": "accept_1",
                    }
                )
                rows.append(row)
            export(rows, root / "out", 20)
            summary = json.loads((root / "out" / "summary.json").read_text())
            self.assertEqual(summary["approved"], 20)
            self.assertEqual(summary["preference_pairs"], 20)
            self.assertGreater(summary["validation"], 0)

    def test_export_preserves_multi_turn_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for number in range(20):
                row = {field: "" for field in FIELDS}
                context = [
                    {"role": "user", "content": "제 이름은 영희예요."},
                    {"role": "assistant", "content": "반갑습니더, 영희님."},
                    {"role": "user", "content": "제 이름이 뭐였죠?"},
                ]
                row.update(
                    {
                        "id": f"memory-{number}",
                        "topic": "memory_schedule",
                        "user": "제 이름이 뭐였죠?",
                        "context_json": json.dumps(context, ensure_ascii=False),
                        "candidate_1": "영희님이라고 하셨습니더.",
                        "decision": "accept_1",
                    }
                )
                rows.append(row)
            export(rows, root / "out", 20)
            exported = []
            for path in (root / "out" / "train.jsonl", root / "out" / "validation.jsonl"):
                exported.extend(json.loads(line) for line in path.read_text().splitlines())
            self.assertEqual(exported[0]["messages"][-2]["content"], "제 이름이 뭐였죠?")
            self.assertEqual(exported[0]["messages"][-1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
