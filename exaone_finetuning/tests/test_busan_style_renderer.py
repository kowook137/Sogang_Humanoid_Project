import unittest

from exaone_finetuning.busan_style_renderer import render


class BusanStyleRendererTests(unittest.TestCase):
    def test_uses_grounded_statement_endings(self):
        self.assertEqual(render("맛있겠네요." )[0], "맛있겠네예.")
        self.assertEqual(render("그렇지요?")[0], "그렇지예?")
        self.assertEqual(render("맞죠?")[0], "맞지예?")

    def test_uses_grounded_request_endings(self):
        self.assertEqual(render("다시 말씀해 주세요.")[0], "다시 말씀해 주이소.")
        self.assertEqual(render("한번 확인해 보세요.")[0], "한번 확인해 보이소.")
        self.assertEqual(render("지금은 움직이지 마세요.")[0], "지금은 움직이지 마이소.")
        self.assertEqual(render("먼저 신고하세요.")[0], "먼저 신고하이소.")

    def test_does_not_force_a_change(self):
        text = "현재 온도는 알 수 없습니다."
        self.assertEqual(render(text), (text, []))

    def test_applies_only_one_change_by_default(self):
        styled, changes = render("걱정되시지요. 한번 확인해 보세요.")
        self.assertEqual(styled, "걱정되시지예. 한번 확인해 보세요.")
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["evidence"], "DKSR20003393.1.1.159")

    def test_does_not_replace_inside_a_word(self):
        text = "하세요체라는 표현을 설명합니다."
        self.assertEqual(render(text), (text, []))


if __name__ == "__main__":
    unittest.main()
