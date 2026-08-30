import unittest

from exaone_finetuning.dialect_style import (
    build_rewrite_messages,
    clean_rewrite,
    has_informal_speech,
    rewrite_is_safe,
    style_gyeongsang,
)


class GyeongsangStyleTests(unittest.TestCase):
    def test_changes_safe_polite_ending_without_changing_fact(self):
        result = style_gyeongsang("현재 온도는 24도입니다. 확인해 보세요.")
        self.assertEqual(result, "현재 온도는 24도입니더. 확인해 보이소.")
        self.assertIn("24도", result)

    def test_uses_specific_request_ending(self):
        self.assertEqual(
            style_gyeongsang("천천히 말씀해 주세요."),
            "천천히 말씀해 주이소.",
        )

    def test_polite_question_does_not_become_ungrammatical(self):
        self.assertEqual(
            style_gyeongsang("어떤 일이 있는지 말씀해 주시겠어요?"),
            "어떤 일이 있는지 말씀해 주실래예?",
        )
        self.assertNotIn("겠네예", style_gyeongsang("말씀해 주시겠어요?"))

    def test_does_not_overwrite_existing_dialect(self):
        text = "바로 일으키지는 마이소. 119에 신고하이소."
        self.assertEqual(style_gyeongsang(text), text)

    def test_existing_marker_does_not_block_other_sentences(self):
        text = "무엇이든 말씀해 주이소. 만나서 반갑습니다. 도움이 필요하시겠네요."
        self.assertEqual(
            style_gyeongsang(text),
            "무엇이든 말씀해 주이소. 만나서 반갑습니더. 도움이 필요하시겠네예.",
        )

    def test_limits_rewrites_to_three_per_response(self):
        text = "사진을 확인해 보세요. 앱을 확인해 보세요. 파일을 확인해 보세요. 설정도 확인해 보세요."
        result = style_gyeongsang(text)
        self.assertEqual(result.count("보이소"), 3)
        self.assertIn("보세요", result)

    def test_adds_conservative_closing_when_no_safe_rewrite_exists(self):
        self.assertEqual(
            style_gyeongsang("오늘 일정은 병원 방문이에요."),
            "오늘 일정은 병원 방문이에요.\n필요하시면 편하게 말씀해 주이소.",
        )

    def test_detects_informal_speech(self):
        self.assertTrue(has_informal_speech("네가 편한 방법으로 해봐."))
        self.assertFalse(has_informal_speech("편한 방법으로 해 보이소."))

    def test_builds_isolated_rewrite_request(self):
        messages = build_rewrite_messages("사진을 선택하세요.")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], "사진을 선택하세요.")

    def test_cleans_rewrite_wrapper(self):
        self.assertEqual(clean_rewrite("출력: 확인해 보이소."), "확인해 보이소.")

    def test_accepts_meaning_preserving_rewrite(self):
        self.assertTrue(
            rewrite_is_safe(
                "사진 3장을 선택하세요.",
                "사진 3장을 선택하이소.",
            )
        )

    def test_rejects_changed_number_and_truncation(self):
        self.assertFalse(rewrite_is_safe("119에 신고하세요.", "112에 신고하이소."))
        self.assertFalse(
            rewrite_is_safe(
                "사진 앱을 열고 사진을 선택한 뒤 전송 버튼을 누르세요.",
                "사진 앱을 여이소.",
            )
        )


if __name__ == "__main__":
    unittest.main()
