"""Generate resumable standard-Korean robot answers with Vertex AI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .generate_vertex_gold_pilot import (
        ANSWER_INSTRUCTION,
        ANSWER_SCHEMA,
        conversation_prompt,
        final_user_message,
        input_messages,
        read_jsonl,
    )
except ImportError:  # Allow direct execution from exaone_finetuning/.
    from generate_vertex_gold_pilot import (
        ANSWER_INSTRUCTION,
        ANSWER_SCHEMA,
        conversation_prompt,
        final_user_message,
        input_messages,
        read_jsonl,
    )


DEFAULT_PROJECT = "project-a965291e-1224-4ea7-bf1"


def is_resource_exhausted(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    text = str(error)
    return status == 429 or "429" in text or "RESOURCE_EXHAUSTED" in text


def generate_with_backoff(call, attempts: int = 8, initial_wait: int = 30):
    """Retry transient Vertex capacity errors with bounded exponential backoff."""
    wait = initial_wait
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as error:
            if not is_resource_exhausted(error) or attempt == attempts:
                raise
            print(
                f"Vertex 429: waiting {wait}s before retry {attempt + 1}/{attempts}",
                flush=True,
            )
            time.sleep(wait)
            wait = min(wait * 2, 600)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--request-delay", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    options = parse_args()
    from google import genai
    from google.genai.types import GenerateContentConfig, HttpOptions

    rows = read_jsonl(options.input)[: options.limit]
    done = set()
    if options.output.exists():
        done.update(row["id"] for row in read_jsonl(options.output))

    client = genai.Client(
        vertexai=True,
        project=options.project,
        location=options.location,
        http_options=HttpOptions(api_version="v1"),
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.errors.parent.mkdir(parents=True, exist_ok=True)
    print(f"resume: {len(done)}/{len(rows)} already completed", flush=True)

    with options.output.open("a", encoding="utf-8") as output, options.errors.open(
        "a", encoding="utf-8"
    ) as errors:
        for index, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            messages = input_messages(row)
            context = conversation_prompt(messages)
            try:
                response = generate_with_backoff(
                    lambda: client.models.generate_content(
                        model=options.model,
                        contents=(
                            "다음 대화의 마지막 사용자 발화에 답하세요. 이전 대화가 "
                            f"있으면 그 문맥과 수정 사항을 반영하세요.\n\n{context}"
                        ),
                        config=GenerateContentConfig(
                            system_instruction=ANSWER_INSTRUCTION,
                            temperature=0.1,
                            response_mime_type="application/json",
                            response_schema=ANSWER_SCHEMA,
                        ),
                    )
                )
                standard = json.loads(response.text)["standard_answer"].strip()
                result = {
                    "id": row["id"],
                    "category": row.get("category", row.get("topic", "general")),
                    "input_style": row.get("input_style", "standard_korean"),
                    "user": final_user_message(messages),
                    "context_messages": messages,
                    "standard_answer": standard,
                    "teacher_model": options.model,
                    "status": "standard_answer_generated",
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                print(f"{index}/{len(rows)} generated", flush=True)
            except Exception as error:
                failure = {
                    "id": row["id"],
                    "error": type(error).__name__,
                    "message": str(error),
                }
                errors.write(json.dumps(failure, ensure_ascii=False) + "\n")
                errors.flush()
                print(f"{index}/{len(rows)} ERROR {type(error).__name__}", flush=True)
            time.sleep(max(0.0, options.request_delay))


if __name__ == "__main__":
    main()
