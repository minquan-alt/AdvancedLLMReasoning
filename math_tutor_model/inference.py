import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
import os
import re
import sys
from io import StringIO
from dotenv import load_dotenv

load_dotenv()
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')
login(HF_AUTH_TOKEN)

ADAPTER_PATH = "/home/guest/AdvancedLLMReasoning/math_tutor_model/math_sft_adapter/v3/final_checkpoint"
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
    
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.padding_side = "left"  # left for inference
    
    print(f"Đang ghép LoRA Adapter từ: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    return model, tokenizer

def execute_python_code(code_str):
    """Execute Python code and return the output."""
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Execute code
        exec_globals = {}
        exec(code_str, exec_globals)
        
        # Get output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # If no print output, try to get the last expression value
        if not output.strip():
            # Try to get the last variable or expression result
            code_lines = code_str.strip().split('\n')
            if code_lines:
                last_line = code_lines[-1].strip()
                # If last line is not an assignment or import
                if '=' not in last_line and not last_line.startswith('import'):
                    try:
                        result = eval(last_line, exec_globals)
                        output = str(result)
                    except:
                        pass
        
        return output.strip()
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout

def post_process_solution(generated_text):
    """
    Post-process the generated solution:
    1. Trim to line containing \\boxed{...}
    2. If code is present, execute it and replace result in boxed
    """
    match = re.search(r'^.*\\boxed\{[^}]+\}.*$', generated_text, re.MULTILINE)
    if match:
        trimmed_text = generated_text[:match.end()]
    else:
        trimmed_text = generated_text
    code_match = re.search(r'```python\s*\n(.*?)\n```', trimmed_text, re.DOTALL)
    if code_match:
        code_str = code_match.group(1)
        # Remove <llm></llm> or <llm-code-output></llm-code-output> patterns
        trimmed_text = re.sub(r'<llm>.*?</llm>', '', trimmed_text, flags=re.DOTALL)
        trimmed_text = re.sub(r'<llm-code-output>.*?</llm-code-output>', '', trimmed_text, flags=re.DOTALL)
        
        # Execute code
        result = execute_python_code(code_str)
        
        # Replace result to \boxed{}
        if result:
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', trimmed_text)
            if boxed_match:
                trimmed_text = re.sub(r'\\boxed\{[^}]+\}', f'\\\\boxed{{{result}}}', trimmed_text)
            else:
                trimmed_text += f'\n\nTherefore, the answer is \\boxed{{{result}}}.'
    
    return trimmed_text

def solve_math_problem(model, tokenizer, question, max_length=1024):
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
    
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
    ]
    
    prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
    )
    
    inputs = tokenizer(
            prompt, 
            padding=False, 
            truncation=True, 
            max_length=max_length, 
            add_special_tokens=False,
            return_tensors="pt"
    ).to("cuda")
    
    print("\n🤖 Model đang suy nghĩ...\n")
    print("-" * 50)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    processed_solution = post_process_solution(generated_text)
    
    return processed_solution

if __name__ == "__main__":
    model, tokenizer = load_model()
    while True:
        question = input("\nNhập bài toán (gõ 'exit' để thoát): ")
        if question.lower() in ['exit', 'quit']:
            break
            
        solution = solve_math_problem(model, tokenizer, question)
        print(solution)