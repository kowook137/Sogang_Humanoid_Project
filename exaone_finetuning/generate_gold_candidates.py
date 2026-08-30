"""Generate a standard answer and three review-only dialect candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
REVISION = "0ff6b5ec7c13b049b253a16a889aa269e6b79a94"
STANDARD_SYSTEM = """어르신을 돕는 대화형 로봇입니다. 질문에 정확하고 친절한 현대 한국어 존댓말로 답하세요.
모르는 현재 정보나 센서값을 추측하지 마세요. 응급 상황은 짧고 안전하게 안내하세요. 답변만 출력하세요."""
STYLE_SYSTEMS = [
    """표준어 답변의 뜻, 사실, 조언, 문장 수와 질문 여부를 그대로 유지하면서 현대 부산·경남의 자연스러운 친근한 존댓말로 바꾸세요. 내부 내용을 새로 쓰지 말고 과장된 사투리를 피하세요. 변환문만 출력하세요.""",
    """다음 답변을 부산에서 어르신께 실제로 말할 법한 부드러운 존댓말로 바꾸세요. 의미와 안전 정보는 한 글자도 훼손하지 말고, 표준어에 어미만 억지로 붙인 표현이나 방송식 사투리는 피하세요. 변환문만 출력하세요.""",
    """다음 내용은 그대로 두고 자연스러운 부산·경남 존댓말로 표현하세요. 모든 문장을 사투리로 과장할 필요는 없지만 전체 말투는 지역 화자가 자연스럽게 느껴야 합니다. 정보를 추가하거나 삭제하지 말고 변환문만 출력하세요.""",
]


def clean(text: str) -> str:
    text = text.strip().removeprefix("```text").removeprefix("```").removesuffix("```").strip()
    for prefix in ("답변:", "출력:", "변환:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        return_dict=True, enable_thinking=False,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return clean(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))


def main() -> None:
    options = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    questions = [json.loads(line) for line in options.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = set()
    if options.output.exists():
        completed = {row["id"] for row in (json.loads(line) for line in options.output.read_text(encoding="utf-8").splitlines() if line.strip())}
    tokenizer = AutoTokenizer.from_pretrained(options.model_id, revision=options.revision)
    model = AutoModelForCausalLM.from_pretrained(
        options.model_id, revision=options.revision, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ).eval()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(questions, 1):
            if row["id"] in completed:
                continue
            standard = generate(model, tokenizer, [
                {"role": "system", "content": STANDARD_SYSTEM},
                {"role": "user", "content": row["user"]},
            ], options.max_new_tokens)
            candidates = []
            for system in STYLE_SYSTEMS:
                candidate = generate(model, tokenizer, [
                    {"role": "system", "content": system},
                    {"role": "user", "content": standard},
                ], options.max_new_tokens)
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
            result = {**row, "standard_answer": standard, "candidates": candidates}
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{index}/{len(questions)} candidates={len(candidates)}", flush=True)


if __name__ == "__main__":
    main()
