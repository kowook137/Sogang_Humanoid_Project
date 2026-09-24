"""Build the native-speaker review sheet for modern Busan polite speech."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROWS = [
    ("information", "오늘 오후 두 시에 치과 예약이 있습니다.", "오늘 오후 두 시에 치과 예약 있으시네예.", "일정 확인"),
    ("information", "주민센터는 다음 주 수요일 오전에 가기로 했습니다.", "주민센터는 다음 주 수요일 오전에 가기로 하셨습니다.", "표면 방언 없음도 허용"),
    ("information", "안경은 현관 서랍에 두셨습니다.", "안경은 현관 서랍에 두셨다고 했지예.", "기억 확인"),
    ("information", "현재 날씨 정보는 확인할 수 없습니다.", "현재 날씨 정보는 제가 확인할 수 없네예.", "불확실성"),
    ("information", "스마트폰 건강 앱에서 걸음 수를 확인할 수 있습니다.", "스마트폰 건강 앱에서 걸음 수를 확인하시면 됩니다.", "표준어 허용"),
    ("empathy", "많이 속상하셨겠어요.", "많이 속상하셨겠네예.", "공감"),
    ("empathy", "그런 일이 있으면 누구나 걱정될 수 있습니다.", "그런 일이 있으면 누구나 걱정되지예.", "공감·동의"),
    ("empathy", "오늘 하루가 많이 힘드셨군요.", "오늘 하루가 많이 힘드셨겠네예.", "공감"),
    ("empathy", "답장이 오지 않아서 마음이 쓰이시겠어요.", "답장이 안 와서 마음이 쓰이시겠네예.", "구어 어휘+공감"),
    ("empathy", "천천히 말씀해 주셔도 괜찮습니다.", "천천히 말씀해 주셔도 괜찮습니다.", "표준어 허용"),
    ("request", "창밖을 한번 확인해 보세요.", "창밖을 한번 확인해 보이소.", "부드러운 요청"),
    ("request", "중요한 일 세 가지만 먼저 적어 보세요.", "중요한 일 세 가지만 먼저 적어 보이소.", "권유"),
    ("request", "무리하지 말고 잠시 쉬세요.", "무리하지 마시고 잠시 쉬이소.", "금지+권유"),
    ("request", "모르는 링크는 누르지 마세요.", "모르는 링크는 누르지 마이소.", "금지"),
    ("request", "가까운 약국에 문의해 보세요.", "가까운 약국에 한번 물어보이소.", "권유+어휘 변환"),
    ("warning_dei", "바닥이 미끄러우니 조심하세요.", "바닥이 미끄러우니 조심하셔야 됩니데이.", "주의의 -데이"),
    ("warning_dei", "약을 임의로 한 번 더 드시면 안 됩니다.", "약을 임의로 한 번 더 드시면 안 됩니데이.", "강한 주의의 -데이"),
    ("warning_dei", "가스 냄새가 나면 불을 켜면 안 됩니다.", "가스 냄새가 나면 불 켜시면 안 됩니데이.", "안전 경고의 -데이"),
    ("warning_dei", "오늘은 바람이 많이 부니 따뜻하게 입으세요.", "오늘은 바람이 많이 분데이, 따뜻하게 입고 나가이소.", "상황 알림+권유"),
    ("warning_dei", "그 길은 공사 중이라 돌아가셔야 합니다.", "그 길은 공사 중이라 돌아가셔야 됩니데이.", "정보 강조"),
    ("question", "어디가 가장 불편하신가요?", "어디가 제일 불편하신가예?", "높임 질문"),
    ("question", "어떤 방법이 가장 편하세요?", "어떤 방법이 제일 편하세요?", "억양 의존·표준 표기"),
    ("question", "오늘은 무엇을 드시고 싶으세요?", "오늘은 뭐 드시고 싶으세요?", "구어체·표준 종결"),
    ("question", "지금 숨을 쉬기 어렵습니까?", "지금 숨쉬기가 어려우세요?", "응급 질문은 명료성 우선"),
    ("question", "제가 기억한 시간이 맞나요?", "제가 기억한 시간이 맞지예?", "확인 질문"),
    ("formal_nider", "제가 직접 확인할 수 없습니다.", "제가 직접 확인할 수는 없습니더.", "제한적 -니더"),
    ("formal_nider", "지금 센서값이 없습니다.", "지금 센서값이 없습니더.", "제한적 -니더"),
    ("formal_nider", "알겠습니다. 그렇게 수정하겠습니다.", "알겠습니다. 그렇게 수정해 둘게예.", "-니더 대신 자연스러운 확인"),
    ("safety", "바로 일으키지 말고 먼저 의식과 호흡을 확인하세요.", "바로 일으키지는 마이소. 먼저 의식하고 호흡부터 확인해 보이소.", "안전 행동 보존"),
    ("safety", "즉시 119에 신고하고 상담원의 안내를 따르세요.", "바로 119에 신고하시고 상담원이 안내하는 대로 하이소.", "숫자·우선순위 보존"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "function", "standard", "proposed_busan", "design_note", "verdict", "native_edit", "reason"]
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (function, standard, proposed, note) in enumerate(ROWS, 1):
            writer.writerow({
                "id": f"busan_cal_{index:03d}",
                "function": function,
                "standard": standard,
                "proposed_busan": proposed,
                "design_note": note,
                "verdict": "",
                "native_edit": "",
                "reason": "",
            })
    print(f"calibration CSV: {args.output} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
