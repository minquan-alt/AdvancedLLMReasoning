import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ADAPTER_PATH = "model/math_sft_model/checkpoint-31680" 
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

def load_model():
    print("⏳ Đang load Base Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Đang ghép LoRA Adapter từ: {ADAPTER_PATH}")
    # load trọng số từ adapter đã huấn luyện
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    return model, tokenizer

def solve_math_problem(model, tokenizer, question):
    prompt = (
        f"### Question:\n{question}\n\n"
        "### Instruction:\n"
        "Let's think step by step. You can use Python code if needed. "
        "At the end, output the final answer inside \\boxed{}.\n\n"
        "### Solution:\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    print("\nModel đang suy nghĩ...\n")
    print("-" * 50)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    solution = full_response.split("### Solution:\n")[-1].strip()
    
    print(solution)
    print("-" * 50)

if __name__ == "__main__":
    model, tokenizer = load_model()
    while True:
        question = input("\nNhập bài toán (gõ 'exit' để thoát): ")
        if question.lower() in ['exit', 'quit']:
            break
            
        solve_math_problem(model, tokenizer, question)