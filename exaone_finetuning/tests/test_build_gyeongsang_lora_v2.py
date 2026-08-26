import unittest

from exaone_finetuning.build_gyeongsang_lora_v2 import build_dataset


class BuildGyeongsangLoraV2Test(unittest.TestCase):
    def test_builds_dialogue_only_splits(self):
        records = [
            {
                "id": "train_1",
                "split": "train",
                "topic": "greeting",
                "messages": [
                    {"role": "user", "content": "안녕하세요."},
                    {"role": "assistant", "content": "반갑네예."},
                ],
            },
            {
                "id": "validation_1",
                "split": "validation",
                "topic": "greeting",
                "messages": [
                    {"role": "user", "content": "좋은 아침이에요."},
                    {"role": "assistant", "content": "좋은 아침입니더."},
                ],
            },
        ]

        train, validation, summary = build_dataset(records)

        self.assertEqual(len(train), 1)
        self.assertEqual(len(validation), 1)
        self.assertEqual(train[0]["messages"][0]["role"], "system")
        self.assertEqual(train[0]["task"], "dialect_chat")
        self.assertFalse(summary["policy"]["conversion_examples_included"])

    def test_rejects_malformed_dialogue(self):
        records = [
            {
                "id": "bad",
                "split": "train",
                "topic": "test",
                "messages": [
                    {"role": "assistant", "content": "순서가 잘못됐습니더."},
                    {"role": "user", "content": "질문"},
                ],
            }
        ]

        with self.assertRaisesRegex(ValueError, "expected user"):
            build_dataset(records)

    def test_rejects_known_bad_phrase(self):
        records = [
            {
                "id": "bad_phrase",
                "split": "train",
                "topic": "test",
                "messages": [
                    {"role": "user", "content": "반가워요."},
                    {"role": "assistant", "content": "반갑습당하게요!"},
                ],
            }
        ]

        with self.assertRaisesRegex(ValueError, "known bad phrase"):
            build_dataset(records)


if __name__ == "__main__":
    unittest.main()
