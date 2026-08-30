import unittest

from exaone_finetuning.generate_vertex_gold_pilot import validate_candidates


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


if __name__ == "__main__":
    unittest.main()
