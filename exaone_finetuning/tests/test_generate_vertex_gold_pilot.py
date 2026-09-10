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

    def test_first_candidate_may_remain_standard_for_conservative_generation(self):
        standard = "현재 센서값이 없어 실내 온도를 알 수 없습니다."
        candidates = [
            standard,
            "현재 센서값이 없어서 실내 온도를 알 수 없습니다.",
            "현재 센서값이 없으므로 실내 온도는 확인할 수 없습니다.",
        ]
        self.assertEqual(validate_candidates(standard, candidates), [])

    def test_all_candidates_may_remain_standard_when_dialect_is_uncertain(self):
        standard = "현재 센서값이 없어 실내 온도를 알 수 없습니다."
        errors = validate_candidates(
            standard,
            [standard, "현재 실내 온도를 알 수 없습니다.", "알 수 없네예."],
        )
        self.assertNotIn("candidate_2_dialect_not_detected", errors)

    def test_allows_conservative_first_candidate_and_rejects_changed_number(self):
        errors = validate_candidates(
            "119에 전화하세요.",
            ["119에 전화하세요.", "112에 전화하이소.", "119에 전화해 주이소."],
        )
        self.assertNotIn("candidate_1_dialect_not_detected", errors)
        self.assertIn("candidate_2_number_changed", errors)

    def test_accepts_current_target_endings_when_source_alignment_is_grounded(self):
        for standard, candidate in (
            ("간단한 음식이 좋겠네요.", "간단한 음식이 좋겠네예."),
            ("간단한 음식이 좋지요.", "간단한 음식이 좋지예."),
        ):
            self.assertEqual(
                validate_candidate(standard, candidate, allow_standard=True), []
            )

    def test_only_accepts_aihub_grounded_source_to_ending_alignment(self):
        self.assertEqual(
            validate_candidate(
                "맛있겠네요.", "맛있겠네예.", allow_standard=True
            ),
            [],
        )
        self.assertEqual(
            validate_candidate("그렇지요?", "그지예?", allow_standard=True),
            [],
        )
        self.assertIn(
            "unsupported_jiye_conversion",
            validate_candidate(
                "위험한 상황입니다.", "위험한 상황이지예.", allow_standard=True
            ),
        )
        self.assertIn(
            "unsupported_neye_conversion",
            validate_candidate(
                "만나서 반갑습니다.", "만나서 반갑네예.", allow_standard=True
            ),
        )

    def test_rejects_nider_for_current_target_even_when_honorific(self):
        self.assertIn(
            "overdone_style",
            validate_candidate("따님 성함은 수진님이십니다.", "따님 성함은 수진님이십니더."),
        )

    def test_recognizes_observed_busan_connectives_and_requests(self):
        for standard, candidate in [
            ("권해 드립니다.", "권해 드립니더."),
            ("신분증을 챙겨가세요.", "신분증을 챙겨가이소."),
            ("준비해 두세요.", "준비해 두이소."),
            ("그렇게 말씀하셨는데요.", "그렇게 말씀하셨는데예."),
            ("보리차라고 하셨습니다.", "보리차라꼬 하셨습니다."),
        ]:
            self.assertNotIn(
                "dialect_not_detected", validate_candidate(standard, candidate)
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
        for awkward in (
            "헷갈리시는 갑네요.",
            "제가 인공지능이라가 확인을 못 합니다.",
            "위로 밀쳐 올리이소.",
            "확인해 보이시면 됩니다.",
            "잡고 계셔보이소.",
            "일정이 있습니다예.",
        ):
            self.assertIn("overdone_style", validate_candidate("확인하세요.", awkward))

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

    def test_rejects_nider_for_current_target_voice(self):
        self.assertIn(
            "overdone_style",
            validate_candidate(
                "먼저 확인하세요.",
                "먼저 확인하시면 됩니더.",
            ),
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
