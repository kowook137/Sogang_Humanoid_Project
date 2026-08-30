"""Fail closed when v5 is still dominated by copy-like examples."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT = MODULE_DIR / "data/processed/gyeongsang/style_v5"


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def analyze(records):
    similarities = [SequenceMatcher(None, r["messages"][1]["content"], r["messages"][2]["content"]).ratio() for r in records]
    return {
        "records": len(records),
        "median_similarity": round(statistics.median(similarities), 4),
        "p90_similarity": round(sorted(similarities)[int(len(similarities) * .9)], 4),
        "register": dict(Counter(r["register"] for r in records)),
        "strength": dict(Counter(r["strength"] for r in records)),
        "weak": sum(r["strength"] == "weak" for r in records),
        "number_mismatch": sum(
            __import__("re").findall(r"\d+(?:[.,]\d+)*", r["messages"][1]["content"])
            != __import__("re").findall(r"\d+(?:[.,]\d+)*", r["messages"][2]["content"])
            for r in records
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    report = {name: analyze(load(args.data_dir / f"{name}.jsonl")) for name in ("train", "validation")}
    failures = []
    if report["train"]["records"] < 5000: failures.append("train below 5000")
    if report["validation"]["records"] < 500: failures.append("validation below 500")
    for split, metrics in report.items():
        if metrics["weak"]: failures.append(f"{split}: weak examples present")
        if metrics["median_similarity"] > .92: failures.append(f"{split}: too copy-like")
        if metrics["number_mismatch"]: failures.append(f"{split}: number mismatch")
        if not metrics["register"].get("polite"): failures.append(f"{split}: no polite data")
        if not metrics["strength"].get("strong"): failures.append(f"{split}: no strong data")
    report["passed"] = not failures
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    (args.data_dir / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit("v5 quality gate failed")


if __name__ == "__main__":
    main()
