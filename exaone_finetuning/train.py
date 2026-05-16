import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. 설정 (EXAONE 4.0 반영)
model_id = "LGAI-EXAONE/EXAONE-4.0-1.2B"
dataset_name = "silk-road/ChatML-Bactrian-ko" 
output_dir = "./exaone-4.0-1.2b-finetuned"

# 2. 모델 및 토크나이저 로드
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# 3. LoRA 설정
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear", 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. 데이터셋 로드 및 Chat Template 포맷팅
dataset = load_dataset(dataset_name, split="train")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        messages = [
            {"role": "system", "content": "공손하고 도움이 되는 AI 어시스턴트입니다."},
            {"role": "user", "content": example['instruction'][i]},
            {"role": "assistant", "content": example['output'][i]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)
    return output_texts

# 5. 트레이너 설정
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=10,
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=100,
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    max_seq_length=2048,
)

# 6. 학습 및 저장
trainer.train()
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
