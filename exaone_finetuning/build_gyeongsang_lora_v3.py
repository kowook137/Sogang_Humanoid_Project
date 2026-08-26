"""Build the safety-, grounding-, and memory-focused Gyeongsang v3 dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_gyeongsang_lora_v2 import build_dataset, load_jsonl, write_jsonl


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = MODULE_DIR / "data" / "drafts" / "gyeongsang_chat_v2.jsonl"
DEFAULT_ADDITIONS = (
    MODULE_DIR / "data" / "drafts" / "gyeongsang_chat_v3_additions.jsonl"
)
DEFAULT_OUTPUT_DIR = MODULE_DIR / "data" / "processed" / "gyeongsang" / "lora_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--additions", type=Path, default=DEFAULT_ADDITIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [*load_jsonl(args.base), *load_jsonl(args.additions)]
    train, validation, summary = build_dataset(records)
    for record in [*train, *validation]:
        record["source_kind"] = "curated_chat_v3"
    summary["version"] = "gyeongsang_lora_v3_grounded_pilot"
    summary["policy"].update(
        {
            "target_model": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
            "safety_grounding_memory_examples_included": True,
            "external_sensor_values_must_be_injected_by_runtime": True,
            "scale_up_ready": False,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
