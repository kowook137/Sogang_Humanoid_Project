"""Generate a fixed side-by-side comparison for the base model and LoRA adapters."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Mapping
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DEFAULT_QUESTIONS = MODULE_DIR / "data" / "evaluation" / "gyeongsang_v2_questions.jsonl"
DEFAULT_OUTPUT = MODULE_DIR / "outputs" / "gyeongsang_v2_comparison.jsonl"
SYSTEM_PROMPT = (MODULE_DIR / "prompts" / "gyeongsang.txt").read_text(
    encoding="utf-8"
).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--revision",
        help="Pinned Hugging Face model revision (commit hash, tag, or branch)",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Adapter to compare; repeat for multiple adapters",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def parse_adapter(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError(f"adapter must use LABEL=PATH format: {value!r}")
    return label.strip(), Path(raw_path).expanduser()


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    seen = set()
    for line_number, record in enumerate(records, start=1):
        if record.get("id") in seen:
            raise ValueError(f"line {line_number}: duplicate question id")
        seen.add(record.get("id"))
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
            raise ValueError(f"line {line_number}: conversation must end with user")
    return records


def generate_variant(model_id, revision, adapter, questions, max_new_tokens):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter is not None:
        if not adapter.is_dir():
            raise FileNotFoundError(f"adapter directory not found: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    responses = []
    for question in questions:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *question["messages"]]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        if isinstance(inputs, Mapping):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            prompt_length = inputs["input_ids"].shape[-1]

            def run_generate(**kwargs):
                return model.generate(**inputs, **kwargs)

        else:
            inputs = inputs.to(model.device)
            prompt_length = inputs.shape[-1]

            def run_generate(**kwargs):
                return model.generate(inputs, **kwargs)

        with torch.inference_mode():
            output = run_generate(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        responses.append(tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip())

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return responses


def main() -> int:
    args = parse_args()
    questions = load_questions(args.questions)
    variants = [("base", None)]
    variants.extend(parse_adapter(value) for value in args.adapter)
    labels = [label for label, _ in variants]
    if len(labels) != len(set(labels)):
        raise ValueError("variant labels must be unique")

    results = {
        question["id"]: {
            "id": question["id"],
            "topic": question.get("topic", "unknown"),
            "messages": question["messages"],
            "responses": {},
        }
        for question in questions
    }
    for label, adapter in variants:
        print(f"Generating: {label}")
        responses = generate_variant(
            args.model_id, args.revision, adapter, questions, args.max_new_tokens
        )
        for question, response in zip(questions, responses, strict=True):
            results[question["id"]]["responses"][label] = response

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for question in questions:
            stream.write(json.dumps(results[question["id"]], ensure_ascii=False) + "\n")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
