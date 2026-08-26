"""Deterministic safety, sensor-grounding, and short-term memory policy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


FALL_TERMS = ("넘어졌", "넘어지셨", "미끄러졌", "낙상")
EMERGENCY_TERMS = (
    "의식이 없",
    "숨을 안",
    "호흡이 이상",
    "피가 많이",
    "말이 어눌",
    "한쪽 팔에 힘",
    "가슴이 아",
    "가슴 통증",
)


@dataclass
class PolicyResult:
    response: str | None = None
    context: str | None = None
    reason: str | None = None


@dataclass
class RobotPolicy:
    """Handle rules that must not depend on probabilistic model generation."""

    memory: dict[str, str] = field(default_factory=dict)
    sensors: dict[str, tuple[str, str]] = field(default_factory=dict)

    def set_sensor(self, name: str, value: str, measured_at: str) -> None:
        if not value.strip() or not measured_at.strip():
            raise ValueError("sensor value and measured_at must not be empty")
        self.sensors[name] = (value.strip(), measured_at.strip())

    def process(self, text: str) -> PolicyResult:
        text = re.sub(r"\s+", " ", text).strip()
        self._update_memory(text)

        safety = self._safety_response(text)
        if safety:
            return PolicyResult(response=safety, reason="safety_rule")

        grounded = self._grounded_response(text)
        if grounded:
            return grounded

        recalled = self._memory_response(text)
        if recalled:
            return PolicyResult(response=recalled, reason="structured_memory")

        context = self._memory_context()
        return PolicyResult(context=context or None)

    def _safety_response(self, text: str) -> str | None:
        if any(term in text for term in EMERGENCY_TERMS):
            return (
                "응급 상황일 수 있습니더. 혼자 해결하거나 직접 운전하지 말고 "
                "바로 119에 신고하이소. 가능하면 안전한 곳에서 곁을 지키고, "
                "상태가 더 나빠지는지 살펴보이소."
            )
        if any(term in text for term in FALL_TERMS):
            return (
                "바로 일으키지는 마이소. 먼저 의식과 호흡, 심한 출혈을 확인하고 "
                "머리나 목, 허리 통증이 있는지 여쭤보이소. 의식이나 호흡이 "
                "이상하거나 심하게 다친 것 같으면 움직이지 말고 119에 신고하이소."
            )
        return None

    def _grounded_response(self, text: str) -> PolicyResult | None:
        asks_weather = (
            any(term in text for term in ("비가 오", "비가 왔", "날씨", "추워", "더워"))
            and any(term in text for term in ("나요", "가요", "인지", "알려", "맞혀", "어때"))
        )
        if asks_weather:
            return self._sensor_or_unknown(
                "weather",
                "현재 위치와 실시간 날씨 정보가 연결되지 않아 확인할 수 없습니더. "
                "확인되지 않은 날씨를 추측해서 말씀드리지는 않을게예.",
            )

        asks_temperature = (
            any(term in text for term in ("방 안", "실내"))
            and any(term in text for term in ("몇 도", "온도"))
        )
        if asks_temperature:
            return self._sensor_or_unknown(
                "temperature",
                "온도 센서 값과 측정 시각이 들어오지 않아 현재 실내 온도는 "
                "알 수 없습니더.",
            )

        asks_vision = any(term in text for term in ("봤어요", "보이나요", "표정", "누가 지나"))
        if asks_vision:
            return self._sensor_or_unknown(
                "vision",
                "현재 카메라나 인식 결과가 연결되지 않아 확인할 수 없습니더. "
                "보지 못한 상황을 봤다고 말씀드리지는 않을게예.",
            )
        return None

    def _sensor_or_unknown(self, name: str, unknown: str) -> PolicyResult:
        reading = self.sensors.get(name)
        if reading is None:
            return PolicyResult(response=unknown, reason=f"missing_sensor:{name}")
        value, measured_at = reading
        return PolicyResult(
            response=f"{measured_at}에 측정된 값은 {value}입니더.",
            reason=f"sensor:{name}",
        )

    def _update_memory(self, text: str) -> None:
        name = re.search(r"(?:제|내) 이름은\s*([^\s,.!?]+)", text)
        if name:
            self.memory["name"] = re.sub(
                r"(?:이고|고|이에요|예요)$", "", name.group(1)
            )

        destination = re.search(r"목적지(?:는|를)\s+([^\s,.!?]+)", text)
        if destination:
            self.memory["destination"] = re.sub(
                r"(?:으로|로|이에요|예요)$", "", destination.group(1)
            )

        schedule = re.search(
            r"((?:오늘|내일|이번 주|다음 주)[^.!?]*(?:시에|에)\s*[^.!?]+)", text
        )
        if schedule:
            self.memory["schedule"] = schedule.group(1).strip()

    def _memory_response(self, text: str) -> str | None:
        asks_name = "이름" in text and any(term in text for term in ("뭐", "무엇", "기억"))
        asks_destination = "목적지" in text and any(
            term in text for term in ("어디", "뭐", "기억")
        )
        asks_schedule = "일정" in text and any(term in text for term in ("뭐", "무엇", "기억"))

        parts = []
        if asks_name:
            parts.append(
                f"이름은 {self.memory['name']}님입니더."
                if "name" in self.memory
                else "현재 저장된 이름 정보가 없습니더."
            )
        if asks_destination:
            parts.append(
                f"현재 목적지는 {self.memory['destination']}입니더."
                if "destination" in self.memory
                else "현재 저장된 목적지 정보가 없습니더."
            )
        if asks_schedule:
            parts.append(
                f"기억한 일정은 ‘{self.memory['schedule']}’라고 하셨습니더."
                if "schedule" in self.memory
                else "현재 저장된 일정 정보가 없습니더."
            )
        return " ".join(parts) or None

    def _memory_context(self) -> str:
        if not self.memory:
            return ""
        labels = {"name": "사용자 이름", "destination": "현재 목적지", "schedule": "일정"}
        facts = "; ".join(f"{labels[key]}={value}" for key, value in self.memory.items())
        return f"구조화 기억(확인된 정보만 사용): {facts}"
