"""Resumably create strong polite Busan targets from real strong Busan speech."""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
REVISION = "0ff6b5ec7c13b049b253a16a889aa269e6b79a94"
POLITE = re.compile(r"(?:요|예|입니더|습니더|하이소|보이소|주이소)[.!?~]*$")
DIALECT = re.compile(
    r"(?:마이|몬|우째|묵|이케|그라모|아이가|데이|나예|네예|제예|지예|"
    r"인가예|는가예|입니더|습니더|하이소|보이소|주이소|"
    r"(?:노|나)(?=[.!?~]*$))"
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
SYSTEM = """실제 부산 사투리 문장을 어르신께 말하는 현대 부산·경남 존댓말로 바꾸세요.
뜻, 사실, 숫자, 고유명사, 주어와 부산 사투리 어휘·문장 구조는 그대로 유지하세요.
단어나 내용을 다시 쓰지 말고 반말 종결어미만 자연스러운 존댓말로 바꾸세요. 정보를 추가하거나
삭제하지 마세요. 과장된 방송식 표현을 쓰지 말고 변환된 한 문장만 출력하세요.

입력: 니 오늘 밥 묵었나?
출력: 오늘 밥 묵으셨는가예?
입력: 이거 우째 하는지 모르겠다.
출력: 이거 우째 하는지 모르겠네예.
입력: 천천히 확인해 봐라.
출력: 천천히 확인해 보이소."""


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3800)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def clean(text):
    text = text.strip().removeprefix("```text").removeprefix("```").removesuffix("```").strip()
    return re.sub(r"^(?:출력|변환)[：:]\s*", "", text).strip()


def reject_reason(source, target):
    if not target: return "empty"
    if not POLITE.search(target): return "not_polite"
    if not DIALECT.search(target): return "dialect_lost"
    if NUMBER.findall(source) != NUMBER.findall(target): return "number_changed"
    if LATIN.findall(source) != LATIN.findall(target): return "latin_token_changed"
    ratio = len(re.sub(r"\s+", "", target)) / max(1, len(re.sub(r"\s+", "", source)))
    if not .65 <= ratio <= 1.35: return "length_changed"
    if SequenceMatcher(None, re.sub(r"\s+", "", source), re.sub(r"\s+", "", target)).ratio() < .55:
        return "meaning_drift"
    return None


def load_candidates(path):
    seen = set()
    candidates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("register") != "non_polite" or record.get("strength") != "strong":
            continue
        source = record["messages"][2]["content"]
        # First/second-person spoken utterances are prone to role reversal when
        # converted to elder-directed speech. Neutral sentences still leave over
        # 2,900 strong real-speaker candidates, enough for the v6 target.
        if re.search(r"(?:^|\s)(?:내가|나는|난|내는|우리|내|니가|니는|너가|너는|넌|니)(?:\s|[,.!?]|$)", source):
            continue
        if source in seen: continue
        seen.add(source)
        candidates.append({"id": record["id"], "dialect_non_polite": source})
    return candidates


def completed_ids(*paths):
    done = set()
    for path in paths:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip(): done.add(json.loads(line)["id"])
    return done


def main():
    options = args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ).eval()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.rejected.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(options.output, options.rejected)
    candidates = load_candidates(options.source)[: options.limit]
    accepted_count = rejected_count = 0
    with options.output.open("a", encoding="utf-8") as good, options.rejected.open("a", encoding="utf-8") as bad:
        for index, record in enumerate(candidates, 1):
            if record["id"] in done: continue
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": record["dialect_non_polite"]}]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                return_dict=True, enable_thinking=False,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=options.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            target = clean(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
            reason = reject_reason(record["dialect_non_polite"], target)
            result = {**record, "dialect_polite": target}
            if reason:
                result["rejected_reason"] = reason
                bad.write(json.dumps(result, ensure_ascii=False) + "\n"); bad.flush(); rejected_count += 1
            else:
                good.write(json.dumps(result, ensure_ascii=False) + "\n"); good.flush(); accepted_count += 1
            if index % 25 == 0:
                print(f"{index}/{len(candidates)} accepted={accepted_count} rejected={rejected_count}", flush=True)


if __name__ == "__main__":
    main()
