"""Compare original standard answers with a dialect-rewrite LoRA."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
REVISION = "0ff6b5ec7c13b049b253a16a889aa269e6b79a94"
SYSTEM = (
    "표준어 답변을 현대 부산·경남 지역의 친근한 존댓말로 바꾸세요. "
    "뜻, 사실, 숫자, 고유명사와 존댓말 수준은 그대로 유지하고 정보를 추가하거나 삭제하지 마세요. "
    "과장된 방송식 사투리와 같은 어미의 반복은 피하세요. 변환된 답변만 출력하세요."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def standard_response(record):
    responses = record.get("responses")
    if isinstance(responses, dict) and isinstance(responses.get("base"), str):
        return responses["base"]
    if isinstance(record.get("response"), str):
        return record["response"]
    raise ValueError(f"record {record.get('id')!r} has no base response")


def main():
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.adapter.is_dir():
        raise FileNotFoundError(f"adapter not found: {args.adapter}")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.revision, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            source = standard_response(record)
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": source}]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, enable_thinking=False,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            rewritten = tokenizer.decode(
                output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            ).strip()
            result = dict(record)
            result["standard_response"] = source
            result["dialect_response"] = rewritten
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
