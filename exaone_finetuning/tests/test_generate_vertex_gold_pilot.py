import unittest

from exaone_finetuning.generate_vertex_gold_pilot import (
    validate_candidate,
    validate_candidates,
)


class VertexGoldPilotTests(unittest.TestCase):
    def test_accepts_three_distinct_grounded_candidates(self):
        standard = "119에 전화하고 바로 일으키지 마세요."
        candidates = [
            "119에 전화하고 바로 일으키지는 마이소.",
            "119에 전화하신 뒤 바로 일으키지는 마이소.",
            "119에 전화부터 하시고 바로 일으키지는 마이소.",
        ]
        self.assertEqual(validate_candidates(standard, candidates), [])

    def test_rejects_missing_dialect_and_changed_number(self):
        errors = validate_candidates(
            "119에 전화하세요.",
            ["119에 전화하세요.", "112에 전화하이소.", "119에 전화해 주이소."],
        )
        self.assertIn("candidate_1_dialect_not_detected", errors)
        self.assertIn("candidate_2_number_changed", errors)

    def test_accepts_observed_modern_endings(self):
        errors = validate_candidates(
            "간단한 음식이 좋겠습니다.",
            [
                "간단한 음식이 좋겠심더.",
                "간단한 음식이면 괜찮을낍니더.",
                "간단한 음식은 어떻겠능교?",
            ],
        )
        self.assertEqual(errors, [])

    def test_rejects_overdone_style(self):
        errors = validate_candidates(
            "작은 목표부터 시작해 보세요.",
            [
                "작은 목표부터 시작해 보이소.",
                "작은 목표부터 시작해 보이소, 아입니꺼.",
                "작은 목표부터 시작하면 좋겠심더.",
            ],
        )
        self.assertIn("candidate_2_overdone_style", errors)

    def test_candidate_level_validation_keeps_clean_alternative(self):
        standard = "작은 목표부터 시작해 보세요."
        self.assertEqual(
            validate_candidate(standard, "작은 목표부터 시작해 보이소."), []
        )
        self.assertIn(
            "dialect_not_detected",
            validate_candidate(standard, "작은 목표부터 시작해 보세요."),
        )


if __name__ == "__main__":
    unittest.main()
