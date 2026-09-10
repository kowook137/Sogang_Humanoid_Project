import unittest
from collections import Counter

from exaone_finetuning.build_pilot_100_scenarios import build_records


class Pilot100ScenarioTests(unittest.TestCase):
    def test_expected_distribution_and_unique_ids(self):
        records = build_records()
        self.assertEqual(len(records), 100)
        self.assertEqual(len({row["id"] for row in records}), 100)
        self.assertEqual(
            Counter(row["category"] for row in records),
            Counter(
                {
                    "single_turn_general": 35,
                    "single_turn_emotion_relationship": 15,
                    "multi_turn_context": 20,
                    "multi_turn_correction_topic_shift": 10,
                    "safety_fall_medical": 8,
                    "memory_schedule": 6,
                    "sensor_uncertainty": 6,
                }
            ),
        )
        self.assertEqual(
            Counter(row["input_style"] for row in records),
            Counter(
                {
                    "standard_korean": 60,
                    "busan": 30,
                    "mixed_or_noisy_spoken": 10,
                }
            ),
        )

    def test_multi_turn_records_have_context(self):
        records = build_records()
        multi = [row for row in records if row["category"].startswith("multi_turn")]
        self.assertTrue(multi)
        self.assertTrue(all(len(row["messages"]) >= 3 for row in multi))
        self.assertTrue(all(row["messages"][-1]["role"] == "user" for row in records))

    def test_historical_assistant_context_does_not_seed_old_dialect(self):
        records = build_records()
        assistant_text = "\n".join(
            message["content"]
            for row in records
            for message in row["messages"][:-1]
            if message["role"] == "assistant"
        )
        for form in ("습니더", "입니더", "주이소", "보이소", "드릴게예"):
            self.assertNotIn(form, assistant_text)


if __name__ == "__main__":
    unittest.main()
