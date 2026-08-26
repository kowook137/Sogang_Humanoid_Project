import unittest

from exaone_finetuning.robot_policy import RobotPolicy


class RobotPolicyTest(unittest.TestCase):
    def test_fall_response_never_recommends_immediate_lift(self):
        result = RobotPolicy().process("어르신이 미끄러져 넘어졌는데 바로 일으킬까요?")
        self.assertEqual(result.reason, "safety_rule")
        self.assertIn("바로 일으키지는 마이소", result.response)
        self.assertIn("119", result.response)

    def test_unknown_weather_is_not_invented(self):
        result = RobotPolicy().process("오늘 우리 동네에 비가 왔는지 맞혀 보세요.")
        self.assertEqual(result.reason, "missing_sensor:weather")
        self.assertIn("확인할 수 없", result.response)
        self.assertIn("추측", result.response)

    def test_temperature_requires_value_and_timestamp(self):
        policy = RobotPolicy()
        missing = policy.process("지금 방 안이 몇 도예요?")
        self.assertEqual(missing.reason, "missing_sensor:temperature")

        policy.set_sensor("temperature", "24도", "오후 3시")
        measured = policy.process("지금 방 안이 몇 도예요?")
        self.assertEqual(measured.reason, "sensor:temperature")
        self.assertEqual(measured.response, "오후 3시에 측정된 값은 24도입니더.")

    def test_name_schedule_and_corrected_destination(self):
        policy = RobotPolicy()
        policy.process("제 이름은 영희고 오늘 오후 세 시에 병원에 가요.")
        recalled = policy.process("제 이름하고 오늘 일정이 뭐였죠?")
        self.assertIn("영희님", recalled.response)
        self.assertIn("오늘 오후 세 시에 병원에 가요", recalled.response)

        policy.process("목적지는 부산역이에요.")
        policy.process("목적지를 서면역으로 바꿀게요.")
        destination = policy.process("현재 목적지가 어디죠?")
        self.assertEqual(destination.response, "현재 목적지는 서면역입니더.")

    def test_general_chat_is_left_for_model(self):
        result = RobotPolicy().process("오늘 기분이 어때요?")
        self.assertIsNone(result.response)
        self.assertIsNone(result.reason)


if __name__ == "__main__":
    unittest.main()
