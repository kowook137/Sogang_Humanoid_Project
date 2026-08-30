import unittest

from exaone_finetuning.generate_gyeongsang_v6_polite import clean, reject_reason


class V6GenerationGateTests(unittest.TestCase):
    def test_accepts_polite_dialect_with_preserved_number(self):
        self.assertIsNone(reject_reason("119에 전화해라.", "119에 전화하이소."))
        self.assertIsNone(reject_reason("선생님이 오셨다 아이가.", "선생님이 오셨네예."))

    def test_rejects_standard_or_changed_fact(self):
        self.assertEqual(reject_reason("밥 묵었나?", "밥 먹었어요."), "dialect_lost")
        self.assertEqual(reject_reason("119에 전화해라.", "112에 전화하이소."), "number_changed")

    def test_cleans_model_wrapper(self):
        self.assertEqual(clean("출력: 천천히 확인해 보이소."), "천천히 확인해 보이소.")


if __name__ == "__main__":
    unittest.main()
