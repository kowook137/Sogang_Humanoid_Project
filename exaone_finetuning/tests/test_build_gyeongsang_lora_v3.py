import unittest
from pathlib import Path

from exaone_finetuning.build_gyeongsang_lora_v2 import build_dataset, load_jsonl


MODULE_DIR = Path(__file__).resolve().parents[1]


class BuildGyeongsangLoraV3Test(unittest.TestCase):
    def test_v3_has_disjoint_grounded_splits(self):
        records = [
            *load_jsonl(MODULE_DIR / "data/drafts/gyeongsang_chat_v2.jsonl"),
            *load_jsonl(
                MODULE_DIR / "data/drafts/gyeongsang_chat_v3_additions.jsonl"
            ),
        ]

        train, validation, summary = build_dataset(records)

        self.assertEqual(len(train), 48)
        self.assertEqual(len(validation), 24)
        self.assertGreaterEqual(summary["train"]["topics"]["safety_fall"], 3)
        self.assertGreaterEqual(summary["train"]["topics"]["grounding_sensor"], 3)
        self.assertGreaterEqual(summary["train"]["topics"]["memory_correction"], 2)
        train_ids = {record["id"] for record in train}
        validation_ids = {record["id"] for record in validation}
        self.assertFalse(train_ids & validation_ids)


if __name__ == "__main__":
    unittest.main()
