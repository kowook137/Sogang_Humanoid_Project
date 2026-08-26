import argparse
import inspect
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = MODULE_DIR / "data" / "processed" / "gyeongsang" / "lora_v2"


def parse_args():
    parser = argparse.ArgumentParser(description="EXAONE Gyeongsang QLoRA training")
    parser.add_argument("--model-id", default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_DATA_DIR / "train.jsonl")
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=DEFAULT_DATA_DIR / "validation.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODULE_DIR / "outputs" / "exaone-gyeongsang-lora-v2",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def expand_assistant_turns(records):
    """Convert conversations into prompt/completion samples per assistant turn."""
    samples = []
    for record_index, record in enumerate(records, start=1):
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"record {record_index}: messages must be a non-empty list")

        normalized = []
        for message_index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                raise ValueError(
                    f"record {record_index}, message {message_index}: message must be an object"
                )
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(
                    f"record {record_index}, message {message_index}: invalid role {role!r}"
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"record {record_index}, message {message_index}: content must be non-empty"
                )

            normalized.append({"role": role, "content": content})
            if role != "assistant":
                continue
            if len(normalized) < 2 or normalized[-2]["role"] != "user":
                raise ValueError(
                    f"record {record_index}, message {message_index}: "
                    "assistant must immediately follow user"
                )
            samples.append(
                {
                    "prompt": normalized[:-1].copy(),
                    "completion": [normalized[-1].copy()],
                }
            )

    if not samples:
        raise ValueError("dataset contains no assistant responses")
    return samples


def load_sft_dataset(path):
    from datasets import Dataset, load_dataset

    if not path.is_file():
        raise FileNotFoundError(
            f"학습 데이터 파일이 없습니다: {path}\n"
            "먼저 v2 대화형 데이터셋을 생성하세요."
        )
    records = load_dataset("json", data_files=str(path), split="train")
    return Dataset.from_list(expand_assistant_turns(records))


def main():
    args = parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다. QLoRA 학습은 GPU VM에서 실행하세요.")

    train_dataset = load_sft_dataset(args.train_file)
    eval_dataset = load_sft_dataset(args.validation_file)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    config_values = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.05,
        "bf16": True,
        "logging_steps": 10,
        "num_train_epochs": args.epochs,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "completion_only_loss": True,
        "optim": "paged_adamw_8bit",
        "max_length": args.max_length,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": "none",
    }
    supported_config = inspect.signature(SFTConfig).parameters
    if "completion_only_loss" not in supported_config:
        raise RuntimeError(
            "설치된 TRL은 completion_only_loss를 지원하지 않습니다. "
            "TRL을 업데이트한 뒤 다시 실행하세요."
        )
    unsupported = sorted(set(config_values) - set(supported_config))
    if unsupported:
        print(f"Ignoring unsupported SFTConfig options: {', '.join(unsupported)}")
    training_args = SFTConfig(
        **{key: value for key, value in config_values.items() if key in supported_config}
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
