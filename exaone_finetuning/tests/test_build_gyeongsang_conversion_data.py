import unittest

from exaone_finetuning.build_gyeongsang_conversion_data import (
    changed_dialect_eojeols,
    has_verified_polite_marker,
    is_strict_busan_speaker,
    robot_exclusion_reasons,
)


class BusanConversionTests(unittest.TestCase):
    def test_requires_all_three_locations_to_be_busan(self):
        speaker = {
            "age": "20대",
            "birthplace": "부산",
            "principal_residence": "부산",
            "current_residence": "부산",
        }
        self.assertTrue(is_strict_busan_speaker(speaker))
        speaker["birthplace"] = "대구"
        self.assertFalse(is_strict_busan_speaker(speaker))

    def test_requires_target_age(self):
        speaker = {
            "age": "60대 이상",
            "birthplace": "부산",
            "principal_residence": "부산",
            "current_residence": "부산",
        }
        self.assertFalse(is_strict_busan_speaker(speaker))

    def test_conservative_gate_accepts_verified_polite_form(self):
        self.assertEqual(
            robot_exclusion_reasons(
                "조금 더 확인해 보세요.", "조금 더 확인해 보이소.", "polite"
            ),
            [],
        )

    def test_conservative_gate_rejects_nider_and_nonhonorific_no(self):
        self.assertIn(
            "excluded_ending",
            robot_exclusion_reasons(
                "확인할 수 없습니다.", "확인할 수 없습니더.", "polite"
            ),
        )
        reasons = robot_exclusion_reasons("무엇을 합니까?", "뭐 하노?", "non_polite")
        self.assertIn("non_polite", reasons)
        self.assertIn("excluded_ending", reasons)

    def test_verified_marker_must_be_explicitly_labelled_by_aihub(self):
        utterance = {
            "eojeolList": [
                {"eojeol": "쫌", "standard": "조금", "isDialect": True},
                {"eojeol": "좋네예.", "standard": "좋네요.", "isDialect": True},
            ]
        }
        changes = changed_dialect_eojeols(utterance)
        self.assertTrue(has_verified_polite_marker(changes))
        self.assertFalse(
            has_verified_polite_marker(
                [{"dialect": "쫌", "standard": "조금"}]
            )
        )

    def test_unlabelled_surface_marker_is_not_treated_as_verified(self):
        utterance = {
            "eojeolList": [
                {"eojeol": "좋네예.", "standard": "좋네요.", "isDialect": False}
            ]
        }
        self.assertEqual(changed_dialect_eojeols(utterance), [])


if __name__ == "__main__":
    unittest.main()
