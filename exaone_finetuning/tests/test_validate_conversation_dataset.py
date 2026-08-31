import unittest

from exaone_finetuning.validate_conversation_dataset import (
    leakage_errors,
    validate_plan,
    validate_records,
)


PLAN = {
    "targets": {"approved_training_records": 2},
    "pilot_100": {"single_turn_general": 100},
    "training_1000": {"single_turn_general": 2},
    "input_style_percent": {"standard_korean": 100},
    "record_requirements": {
        "required_fields": ["id", "category", "input_style", "reviewed", "messages"],
        "allowed_roles": ["system", "user", "assistant"],
        "multi_turn_minimum_assistant_turns": 2,
    },
}


def record(identifier: str, user: str, assistant: str, reviewed: bool = True) -> dict:
    return {
        "id": identifier,
        "category": "single_turn_general",
        "input_style": "standard_korean",
        "reviewed": reviewed,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


class ConversationDatasetValidationTests(unittest.TestCase):
    def test_plan_totals(self):
        self.assertEqual(validate_plan(PLAN), [])

    def test_reviewed_record_is_valid(self):
        errors, summary = validate_records(
            [record("train-1", "안녕하세요", "반갑습니더")],
            PLAN,
            training=True,
            strict=True,
        )
        self.assertEqual(errors, [])
        self.assertEqual(summary["assistant_turns"], 1)

    def test_unreviewed_training_record_fails(self):
        errors, _ = validate_records(
            [record("train-1", "안녕하세요", "반갑습니더", reviewed=False)],
            PLAN,
            training=True,
            strict=True,
        )
        self.assertTrue(any("not human reviewed" in error for error in errors))

    def test_detects_evaluation_leakage(self):
        train = [record("train-1", "오늘 뭐 먹을까요?", "국밥 어떠십니꺼?")]
        evaluation = [record("eval-1", "  오늘 뭐 먹을까요? ", "다른 답변")]
        errors = leakage_errors(train, evaluation)
        self.assertTrue(any("user text leaked" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
