import unittest

from exaone_finetuning.validate_gyeongsang_style_v4 import validate


def item(index, source, target):
    return {
        "id": f"id_{index}",
        "messages": [
            {"role": "system", "content": "rewrite"},
            {"role": "user", "content": source},
            {"role": "assistant", "content": target},
        ],
    }


class ValidateGyeongsangStyleV4Test(unittest.TestCase):
    def test_detects_number_change_and_leakage(self):
        pair = item(1, "약은 하루에 2번 드세요.", "약은 하루에 3번 드이소.")
        report = validate([pair] * 2000, [pair] * 400)
        self.assertFalse(report["passed"])
        self.assertGreater(report["train"]["number_mismatch_rate"], 0)
        self.assertGreater(report["cross_split_pair_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
