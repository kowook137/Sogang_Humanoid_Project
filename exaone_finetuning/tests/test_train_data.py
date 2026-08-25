import unittest

from exaone_finetuning.train import expand_assistant_turns


class ExpandAssistantTurnsTest(unittest.TestCase):
    def test_single_turn_becomes_prompt_and_completion(self):
        records = [
            {
                "messages": [
                    {"role": "system", "content": "경상도 말투로 답하세요."},
                    {"role": "user", "content": "밥 먹었어요?"},
                    {"role": "assistant", "content": "예, 방금 묵었어요."},
                ]
            }
        ]

        samples = expand_assistant_turns(records)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["prompt"], records[0]["messages"][:2])
        self.assertEqual(samples[0]["completion"], [records[0]["messages"][2]])

    def test_multi_turn_creates_one_sample_per_assistant_response(self):
        messages = [
            {"role": "system", "content": "친근하게 답하세요."},
            {"role": "user", "content": "오늘 뭐 했어요?"},
            {"role": "assistant", "content": "일 좀 했지예."},
            {"role": "user", "content": "힘들진 않았어요?"},
            {"role": "assistant", "content": "괜찮았어요. 걱정 마이소."},
        ]

        samples = expand_assistant_turns([{"messages": messages}])

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["prompt"], messages[:2])
        self.assertEqual(samples[1]["prompt"], messages[:4])
        self.assertEqual(samples[1]["completion"], [messages[4]])

    def test_assistant_must_follow_user(self):
        records = [
            {
                "messages": [
                    {"role": "system", "content": "시스템"},
                    {"role": "assistant", "content": "잘못된 응답"},
                ]
            }
        ]

        with self.assertRaisesRegex(ValueError, "assistant must immediately follow user"):
            expand_assistant_turns(records)


if __name__ == "__main__":
    unittest.main()
