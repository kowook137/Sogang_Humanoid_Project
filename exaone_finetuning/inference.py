"""Interactive, persistent multi-turn chat interface for EXAONE 4.0."""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Mapping
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat_session import ChatSession, find_session, list_sessions
from dialect_style import has_informal_speech, style_gyeongsang
from robot_policy import RobotPolicy


DEFAULT_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
DEFAULT_MODEL_REVISION = "0ff6b5ec7c13b049b253a16a889aa269e6b79a94"
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER_PATH = MODULE_DIR / "exaone-4.0-1.2b-finetuned"
DEFAULT_LOG_ROOT = MODULE_DIR / "chat_logs"
DEFAULT_SYSTEM_PROMPT_PATH = MODULE_DIR / "system_prompt.txt"
DIALECTS = ("standard", "gyeongsang", "jeolla", "chungcheong")
DIALECT_PROMPTS = {
    "standard": DEFAULT_SYSTEM_PROMPT_PATH,
    "gyeongsang": MODULE_DIR / "prompts" / "gyeongsang.txt",
    "jeolla": MODULE_DIR / "prompts" / "jeolla.txt",
    "chungcheong": MODULE_DIR / "prompts" / "chungcheong.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--dialect", choices=DIALECTS, default="standard")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument(
        "--system-prompt",
        type=Path,
        help="Override the prompt selected by --dialect",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-session", help="Session ID or unique session ID prefix")
    return parser.parse_args()


class ExaoneRuntime:
    def __init__(
        self, model_id: str, revision: str | None, adapter_path: Path, max_new_tokens: int
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path.is_dir():
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"Loaded adapter from {adapter_path}")
        else:
            print(f"Running base model (adapter not found at {adapter_path})")

    def generate_response(
        self, messages: list[dict[str, str]], reasoning_mode: bool = False
    ) -> str:
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=reasoning_mode,
        ).to(self.model.device)
        if isinstance(model_inputs, Mapping):
            prompt_length = model_inputs["input_ids"].shape[1]

            def run_generate(**kwargs):
                return self.model.generate(**model_inputs, **kwargs)

        else:
            prompt_length = model_inputs.shape[1]

            def run_generate(**kwargs):
                return self.model.generate(model_inputs, **kwargs)

        generation_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.1 if not reasoning_mode else 0.6,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # 일부 Transformers 버전은 presence_penalty를 지원하지 않으므로
        # 기존 작업에서 추가한 호환성 fallback을 유지한다.
        try:
            outputs = run_generate(
                presence_penalty=1.5,
                **generation_kwargs,
            )
        except (TypeError, ValueError):
            outputs = run_generate(**generation_kwargs)

        response = self.tokenizer.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        )
        return response.strip()


def print_help() -> None:
    print("Commands:")
    print("  think              reasoning mode on/off")
    print("  new                start a new conversation")
    print("  history            show the current conversation")
    print("  sessions           list saved sessions")
    print("  load <session-id>  load a saved session")
    print("  exit               quit")


def load_system_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"System prompt is empty: {path}")
    return prompt


def build_policy(messages: list[dict[str, str]]) -> RobotPolicy:
    policy = RobotPolicy()
    for message in messages:
        if message["role"] == "user":
            policy.process(message["content"])
    return policy


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    prompt_path = args.system_prompt or DIALECT_PROMPTS[args.dialect]
    default_log_dir = (
        DEFAULT_LOG_ROOT
        if args.dialect == "standard"
        else DEFAULT_LOG_ROOT / args.dialect
    )
    log_dir = args.log_dir or default_log_dir
    system_prompt = load_system_prompt(prompt_path)
    if args.load_session:
        session = ChatSession.load(
            find_session(log_dir, args.load_session),
            system_prompt,
            dialect=args.dialect,
        )
    else:
        session = ChatSession(log_dir, system_prompt, dialect=args.dialect)

    policy = build_policy(session.messages)

    runtime = ExaoneRuntime(
        args.model_id, args.revision, args.adapter_path, args.max_new_tokens
    )
    reasoning_mode = False
    print("-" * 60)
    print("EXAONE persistent multi-turn chat")
    print(f"Dialect: {args.dialect}")
    print(f"Session: {session.session_id}")
    print(f"Log: {session.log_path}")
    print_help()
    print("-" * 60)

    while True:
        mode = "REASONING" if reasoning_mode else "NORMAL"
        try:
            user_input = input(f"[{mode}] User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n대화를 종료합니다.")
            break
        if not user_input:
            continue
        command, _, argument = user_input.partition(" ")
        command = command.lower()

        if command in {"exit", "quit", "종료"} and not argument:
            break
        if command == "think" and not argument:
            reasoning_mode = not reasoning_mode
            print(f"> Reasoning mode: {reasoning_mode}")
            continue
        if command == "new" and not argument:
            session = ChatSession(log_dir, system_prompt, dialect=args.dialect)
            policy = RobotPolicy()
            print(f"> New session: {session.session_id}")
            continue
        if command == "history" and not argument:
            print(session.display_history())
            continue
        if command == "sessions" and not argument:
            saved = list_sessions(log_dir)
            print("\n".join(path.stem for path in saved) or "(저장된 세션이 없습니다.)")
            continue
        if command == "load" and argument:
            try:
                session = ChatSession.load(
                    find_session(log_dir, argument.strip()),
                    system_prompt,
                    dialect=args.dialect,
                )
            except (FileNotFoundError, ValueError) as error:
                print(f"> {error}")
            else:
                policy = build_policy(session.messages)
                print(f"> Loaded session: {session.session_id}")
            continue
        if command == "help" and not argument:
            print_help()
            continue

        session.add_message("user", user_input, reasoning_mode)
        policy_result = policy.process(user_input)
        if policy_result.response is not None:
            response = policy_result.response
        else:
            model_messages = [message.copy() for message in session.messages]
            if policy_result.context:
                model_messages[0]["content"] += "\n\n" + policy_result.context
            try:
                response = runtime.generate_response(model_messages, reasoning_mode)
            except Exception as error:
                print(f"> Generation failed: {type(error).__name__}: {error!r}")
                traceback.print_exc()
                continue
            if args.dialect == "gyeongsang":
                if has_informal_speech(response):
                    repair_messages = [
                        *model_messages,
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": (
                                "방금 답변의 사실과 의미는 유지하고, 어르신께 드리는 "
                                "친근한 존댓말로만 고쳐서 답변 본문만 다시 작성하세요."
                            ),
                        },
                    ]
                    try:
                        response = runtime.generate_response(repair_messages, False)
                    except Exception:
                        pass
                response = style_gyeongsang(response)
        session.add_message("assistant", response, reasoning_mode)
        print(f"AI: {response}\n")

    print(f"대화 기록: {session.log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
