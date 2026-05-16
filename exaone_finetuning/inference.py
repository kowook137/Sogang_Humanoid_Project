import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "LGAI-EXAONE/EXAONE-4.0-1.2B"
adapter_path = "./exaone-4.0-1.2b-finetuned"

# 1. 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 2. 파인튜닝 어댑터 병합 (학습된 결과가 있을 경우만)
try:
    model = PeftModel.from_pretrained(model, adapter_path)
    print(f"Loaded adapter from {adapter_path}")
except Exception as e:
    print(f"Running base model for testing (Adapter not found at {adapter_path})")

def generate_response(user_input, reasoning_mode=False):
    # 가이드에 따른 메시지 구성
    messages = [
        {"role": "system", "content": "공손하고 도움이 되는 AI 어시스턴트입니다."},
        {"role": "user", "content": user_input}
    ]
    
    # EXAONE 4.0 전용 기능: enable_thinking=True 시 추론(Reasoning) 과정 활성화
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=reasoning_mode
    ).to(model.device)
    
    # 가이드 권장 파라미터 적용 (Temperature=0.1, Presence Penalty=1.5)
    # 추론 모드일 때는 좀 더 창의적인 temperature=0.6 권장
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1 if not reasoning_mode else 0.6,
        top_p=0.95,
        presence_penalty=1.5,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 답변 부분만 디코딩
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    print("-" * 50)
    print("EXAONE 4.0 1.2B Chat Interface")
    print("Commands: 'exit' to quit, 'think' to toggle reasoning mode")
    print("-" * 50)
    
    r_mode = False
    while True:
        mode_str = "REASONING" if r_mode else "NORMAL"
        user_input = input(f"[{mode_str}] User: ")
        
        if user_input.lower() in ["exit", "quit", "종료"]:
            break
        if user_input.lower() == "think":
            r_mode = not r_mode
            print(f"> Reasoning mode set to: {r_mode}")
            continue
            
        response = generate_response(user_input, reasoning_mode=r_mode)
        print(f"AI: {response}\n")
