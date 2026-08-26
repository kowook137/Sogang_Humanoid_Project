import unittest

from exaone_finetuning.build_gyeongsang_style_v4 import build_dataset


def record(record_id, split, strength, source, target, topic="daily"):
    return {
        "id": f"{split}_{record_id}",
        "register": "polite",
        "strength": strength,
        "source_standard": source,
        "target_dialect": target,
        "metrics": {
            "change_ratio": 0.08,
            "dialect_eojeol_count": 2,
        },
        "source": {"topic": topic, "document_id": f"doc_{split}_{record_id}"},
    }


class BuildGyeongsangStyleV4Test(unittest.TestCase):
    def test_builds_rewrite_only_dataset_without_leakage(self):
        train_source = [
            record("1", "train", "medium", "오늘은 많이 추운 것 같아요.", "오늘은 마이 추운 것 같네예."),
            record("2", "train", "weak", "조금 천천히 가시면 좋겠어요.", "쫌 천천히 가시면 좋겠어요."),
        ]
        validation_source = [
            record("1", "validation", "medium", "내일 다시 확인해 드릴게요.", "내일 다시 확인해 드릴게예."),
        ]

        train, validation, summary = build_dataset(
            train_source,
            validation_source,
            train_medium=10,
            train_weak=10,
            validation_medium=10,
            validation_weak=10,
        )

        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)
        self.assertTrue(all(r["task"] == "dialect_rewrite" for r in train))
        self.assertEqual(train[0]["messages"][0]["role"], "system")
        self.assertEqual(summary["policy"]["runtime_usage"], "second_pass_rewriter")

    def test_rejects_duplicate_pair_across_splits(self):
        duplicate = ("오늘은 많이 추운 것 같아요.", "오늘은 마이 추운 것 같네예.")
        with self.assertRaisesRegex(ValueError, "leakage"):
            build_dataset(
                [record("1", "train", "medium", *duplicate)],
                [record("1", "validation", "medium", *duplicate)],
            )


if __name__ == "__main__":
    unittest.main()
