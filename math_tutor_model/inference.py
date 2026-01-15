import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
import os
import re
import sys
from io import StringIO
from dotenv import load_dotenv
from utils.inference_utils import solve_math_problem

load_dotenv()
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')
login(HF_AUTH_TOKEN)

ADAPTER_PATH = "/home/guest/AdvancedLLMReasoning/math_tutor_model/math_sft_adapter/v3/final_checkpoint"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

def load_model():
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
    
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.padding_side = "left"
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("Đã load Model...")
    
    return model, tokenizer

def solve_math_problem_wrapper(model, tokenizer, question, max_length=512):
    system_prompt = (
            "You are a math reasoning assistant.\n"
            "Solve the problem step by step.\n"
            "You can use Python code if needed.\n"
            "If you write code, put it inside a Python code block:\n"
            "```python\n"
            "...\n"
            "```\n"
            "Output ONLY the final number inside \\boxed{}."
    )
    
    print("\n🤖 Model đang suy nghĩ...\n")
    print("-" * 50)
    
    return solve_math_problem(model, tokenizer, question, system_prompt, max_length)

if __name__ == "__main__":
    model, tokenizer = load_model()
    while True:
        question = input("\nNhập bài toán (gõ 'exit' để thoát): ")
        if question.lower() in ['exit', 'quit']:
            break
            
        solution = solve_math_problem_wrapper(model, tokenizer, question)
        print(solution)