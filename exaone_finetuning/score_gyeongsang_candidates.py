"""Score extracted Busan dialect chat pairs with transparent heuristics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = MODULE_DIR / "data" / "processed" / "gyeongsang"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "scored"

STRONG_DIALECT = re.compile(
    r"(겠지예|네예|지예|입니더|습니더|는교|은교|아입니|하이소|보이소|"
    r"주이소|가이소|드이소|데이(?=[.!?~\s]|$)|아이가|우짜|쪼매|단디|가꼬)"
)
WEAK_DIALECT = re.compile(
    r"(쫌|이케|가주구|거진|인제|같애요|구요|함|땜에|그닥|글케)"
)
TRANSCRIPTION_NOISE = re.compile(
    r"(&[^&]+&|@[가-힣A-Za-z0-9_-]+|\(\(.*?\)\)|-[가-힣]+-|\{[^}]+\}|/\([^)]*\))"
)
FILLER = re.compile(r"(?:^|\s)(음|어|뭐|막|약간)(?=\s|$)")
TOKEN = re.compile(r"[가-힣A-Za-z0-9]+")
PARTICLES = re.compile(r"(은|는|이|가|을|를|에|의|도|로|으로|와|과|하고|에서|에게)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def tokens(text: str) -> set[str]:
    result = set()
    for value in TOKEN.findall(text):
        value = PARTICLES.sub("", value)
        if len(value) >= 2:
            result.add(value)
    return result


def relevance(user: str, assistant: str) -> float:
    left, right = tokens(user), tokens(assistant)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def repetition_penalty(text: str) -> float:
    words = TOKEN.findall(text)
    if len(words) < 6:
        return 0.0
    counts = Counter(words)
    return min(2.0, sum(count - 2 for count in counts.values() if count > 2) * 0.5)


def score_record(record: dict) -> dict:
    user = record["messages"][0]["content"]
    assistant = record["messages"][1]["content"]
    standard = record.get("assistant_standard", "")
    strong = len(STRONG_DIALECT.findall(assistant))
    weak = len(WEAK_DIALECT.findall(assistant))
    fillers = len(FILLER.findall(assistant))
    change_ratio = 1.0 - SequenceMatcher(None, standard, assistant).ratio()
    topic_relevance = relevance(user, assistant)
    noise = bool(TRANSCRIPTION_NOISE.search(user + " " + assistant))
    repeated = repetition_penalty(assistant)

    score = 0.0
    reasons: list[str] = []
    if strong:
        score += min(6.0, strong * 3.0)
        reasons.append(f"strong_dialect={strong}")
    if weak:
        score += min(2.0, weak * 0.5)
        reasons.append(f"weak_dialect={weak}")
    score += min(3.0, change_ratio * 12.0)
    reasons.append(f"change_ratio={change_ratio:.3f}")
    score += min(2.0, topic_relevance * 3.0)
    reasons.append(f"topic_relevance={topic_relevance:.3f}")
    if user.endswith("?"):
        score += 0.5
        reasons.append("question_input")
    if fillers:
        score -= min(2.0, fillers * 0.5)
        reasons.append(f"fillers={fillers}")
    if repeated:
        score -= repeated
        reasons.append(f"repetition_penalty={repeated:.1f}")
    if noise:
        score -= 10.0
        reasons.append("transcription_noise")

    if noise or score < 1.5:
        decision = "exclude"
    elif score >= 4.0 and strong > 0:
        decision = "recommended"
    else:
        decision = "hold"

    enriched = dict(record)
    enriched["quality"] = {
        "decision": decision,
        "score": round(score, 3),
        "strong_dialect_markers": strong,
        "weak_dialect_markers": weak,
        "change_ratio": round(change_ratio, 4),
        "topic_relevance": round(topic_relevance, 4),
        "reasons": reasons,
    }
    return enriched


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in ("train", "validation"):
        source = args.input_dir / f"{split}_candidates.jsonl"
        scored = [score_record(record) for record in load_jsonl(source)]
        scored.sort(key=lambda record: record["quality"]["score"], reverse=True)
        decisions = Counter(record["quality"]["decision"] for record in scored)
        summary[split] = {"total": len(scored), **dict(decisions)}
        write_jsonl(args.output_dir / f"{split}_scored.jsonl", scored)
        for decision in ("recommended", "hold", "exclude"):
            write_jsonl(
                args.output_dir / f"{split}_{decision}.jsonl",
                [r for r in scored if r["quality"]["decision"] == decision],
            )
        print(split, summary[split])

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
