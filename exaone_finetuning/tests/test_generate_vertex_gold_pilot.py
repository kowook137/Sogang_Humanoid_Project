import unittest

from exaone_finetuning.generate_vertex_gold_pilot import (
    conversation_prompt,
    final_user_message,
    input_messages,
    validate_candidate,
    validate_candidates,
)


class VertexGoldPilotTests(unittest.TestCase):
    def test_formats_multi_turn_input(self):
        row = {
            "messages": [
                {"role": "user", "content": "제 이름은 영희예요."},
                {"role": "assistant", "content": "반갑습니더, 영희님."},
                {"role": "user", "content": "제 이름이 뭐였죠?"},
            ]
        }
        messages = input_messages(row)
        self.assertEqual(final_user_message(messages), "제 이름이 뭐였죠?")
        self.assertIn("AI: 반갑습니더, 영희님.", conversation_prompt(messages))

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
                "간단한 음식이면 괜찮으십니더.",
                "간단한 음식이 괜찮겠심더.",
            ],
        )
        self.assertEqual(errors, [])

    def test_recognizes_honorific_sipnider(self):
        self.assertEqual(
            validate_candidate("따님 성함은 수진님이십니다.", "따님 성함은 수진님이십니더."),
            [],
        )

    def test_accepts_tv_and_television_as_same_latin_term(self):
        self.assertEqual(
            validate_candidate(
                "텔레비전 프로그램을 보세요.",
                "TV 프로그램을 한번 보이소.",
            ),
            [],
        )

    def test_rejects_observed_awkward_forms_and_overuse(self):
        self.assertIn(
            "overdone_style",
            validate_candidate("문을 열어보세요.", "문을 여이소."),
        )
        self.assertIn(
            "dialect_overused",
            validate_candidate(
                "확인하고 선택한 뒤 알려주세요. 그러면 처리하겠습니다.",
                "확인해 보이소. 선택해 보이소. 알려주이소. 처리하겠습니더.",
            ),
        )

    def test_rejects_changed_negation_question_and_critical_term(self):
        errors = validate_candidate(
            "숨을 쉬지 않으면 119에 신고하세요?",
            "숨을 쉬면 112에 신고하이소.",
        )
        self.assertIn("number_changed", errors)
        self.assertIn("negation_changed", errors)
        self.assertIn("critical_term_changed:119", errors)
        self.assertIn("critical_term_changed:112", errors)

    def test_rejects_added_intensifier(self):
        self.assertIn(
            "intensifier_added",
            validate_candidate("먼저 신고하세요.", "무조건 먼저 신고하이소."),
        )

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
