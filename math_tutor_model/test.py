import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import re
import argparse
import json
import sys
from io import StringIO
from math_tutor_model.math_equivalence import is_equiv
from utils.prompt import PROMPT_V0, PROMPT_V1, PROMPT_V2, SYSTEM_PROMPT_V3

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

def load_model(model_id="sft", data_path=1):
    if model_id == "rl":
        ADAPTER_PATH = "math_tutor_model/math_rl_adapter/final_checkpoint"
    elif model_id == "sft":
        ADAPTER_PATH = f"/home/guest/AdvancedLLMReasoning/math_tutor_model/math_sft_adapter/v{data_path}/final_checkpoint" 
    else:
        ADAPTER_PATH = None

    print(f"Loading...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16
    )
    print("Base Model loaded")
    
    # Load tokenizer from adapter for v3 (has chat_template), otherwise from base model
    if ADAPTER_PATH:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    tokenizer.padding_side = "left"
    if data_path < 3:
        tokenizer.pad_token = tokenizer.eos_token
    
    if ADAPTER_PATH is None:
        model.eval()
        return model, tokenizer
    
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print(f"Adapter loaded from: {ADAPTER_PATH}")
    except:
        print("Không load được Adapter.")
        exit(1)

    model.eval()
    return model, tokenizer

def execute_python_code(code_str):
    """Execute Python code and return the output."""
    try:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        exec_globals = {}
        exec(code_str, exec_globals)
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # If no print output, try to get last expression value
        if not output.strip():
            code_lines = code_str.strip().split('\n')
            if code_lines:
                last_line = code_lines[-1].strip()
                if '=' not in last_line and not last_line.startswith('import'):
                    try:
                        result = eval(last_line, exec_globals)
                        return str(result)
                    except:
                        pass
        
        return output.strip()
    except Exception as e:
        return None
    finally:
        sys.stdout = old_stdout

def post_process_solution_v0_v1_v2(generated_text):
    """
    Post-process solution for v0/v1/v2 format (using <llm-code> tags):
    1. Trim to line containing \\boxed{...}
    2. If code is present, execute it and replace result in boxed
    """
    match = re.search(r'^.*\\boxed\{[^}]+\}.*$', generated_text, re.MULTILINE)
    if match:
        trimmed_text = generated_text[:match.end()]
    else:
        trimmed_text = generated_text
    
    # Check for Python code in <llm-code> tags
    code_match = re.search(r'<llm-code>\s*(.*?)\s*</llm-code>', trimmed_text, re.DOTALL)
    
    if code_match:
        code_str = code_match.group(1)
        
        # Remove output tags
        trimmed_text = re.sub(r'<llm>.*?</llm>', '', trimmed_text, flags=re.DOTALL)
        trimmed_text = re.sub(r'<llm-code-output>.*?</llm-code-output>', '', trimmed_text, flags=re.DOTALL)
        
        # Execute code
        result = execute_python_code(code_str)
        
        # Only replace if execution was successful
        if result:
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', trimmed_text)
            if boxed_match:
                trimmed_text = re.sub(r'\\boxed\{[^}]+\}', f'\\\\boxed{{{result}}}', trimmed_text)
            else:
                trimmed_text += f'\n\nTherefore, the answer is \\boxed{{{result}}}.'
    
    return trimmed_text

def post_process_solution_v3(generated_text):
    """
    Post-process solution for v3 format (using ```python code blocks):
    1. Trim to line containing \\boxed{...}
    2. If code is present, execute it and replace result in boxed
    """
    match = re.search(r'^.*\\boxed\{[^}]+\}.*$', generated_text, re.MULTILINE)
    if match:
        trimmed_text = generated_text[:match.end()]
    else:
        trimmed_text = generated_text
    
    # Check for Python code in ```python blocks
    code_match = re.search(r'```python\s*\n(.*?)\n```', trimmed_text, re.DOTALL)
    
    if code_match:
        code_str = code_match.group(1)
        
        # Remove output tags
        trimmed_text = re.sub(r'<llm>.*?</llm>', '', trimmed_text, flags=re.DOTALL)
        trimmed_text = re.sub(r'<llm-code-output>.*?</llm-code-output>', '', trimmed_text, flags=re.DOTALL)
        
        # Execute code
        result = execute_python_code(code_str)
        
        # Only replace if execution was successful
        if result:
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', trimmed_text)
            if boxed_match:
                trimmed_text = re.sub(r'\\boxed\{[^}]+\}', f'\\\\boxed{{{result}}}', trimmed_text)
            else:
                trimmed_text += f'\n\nTherefore, the answer is \\boxed{{{result}}}.'
    
    return trimmed_text

def extract_answer(text):
    if "\\boxed{" in text:
        idx = text.rfind("\\boxed{")
        content = ""
        count = 0
        started = False
        for char in text[idx:]:
            if char == "{":
                count += 1
                started = True
                if count == 1: continue 
            elif char == "}":
                count -= 1
            if started:
                if count == 0: break
                content += char
        return content.strip()
    
    match = re.search(r'[Tt]he answer is[:\s]+(-?[\d,\.]+)', text)
    if match:
        return match.group(1)
        
    return None

def evaluate(model_id=None, dataset_name="gsm8k", data_path=1, num_samples=-1):
    model, tokenizer = load_model(model_id, data_path)
    model_id = model_id if model_id in ['sft', 'rl'] else "base"
    output_file = f"result_{model_id}_{dataset_name}_v{data_path}.json"
    
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        # get truth and remove commas (e.g. 128,000 -> 128000)
        get_truth = lambda x: x['answer'].split("####")[-1].strip().replace(',', '')
    elif dataset_name == "math":
        ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
        get_truth = lambda x: extract_answer(x['solution'])

    if num_samples > 0:
        ds = ds.select(range(min(len(ds), num_samples)))

    print(f"\n{'='*60}")
    print(f"BENCHMARK (Pass@1, Zero-shot)")
    print(f"Model: {model_id}")
    print(f"Data-processed: v{data_path}")
    print(f"Dataset: {dataset_name} | Samples: {len(ds)}")
    print(f"{'='*60}")

    results = []
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    for item in tqdm(ds, desc="Evaluating"):
        question = item['question'] if dataset_name == 'gsm8k' else item['problem']
        ground_truth = get_truth(item)
        
        # chọn prompt theo data_path
        if data_path >= 3:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_V3},
                {"role": "user", "content": question},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            if data_path == 0:
                prompt = PROMPT_V0.format(question=question)
            elif data_path == 1:
                prompt = PROMPT_V1.format(question=question)
            else:  # data_path == 2
                prompt = PROMPT_V2.format(question=question)
        
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=terminators if data_path < 3 else tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        if data_path >= 3:
            processed_solution = post_process_solution_v3(generated_text)
        else:
            processed_solution = post_process_solution_v0_v1_v2(generated_text)
        
        # trích xuất đáp án và so sánh
        predicted_answer = extract_answer(processed_solution)
        correct = is_equiv(predicted_answer, ground_truth)

        results.append({
            "question": question,
            "generated_solution": processed_solution,
            "predicted_answer": predicted_answer,
            "truth": ground_truth,
            "correct": correct
        })
    
    # tổng hợp và in kết quả
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"Correct: {correct_count}/{total_count}")
    print(f"Accuracy (PASS@1): {accuracy*100:.2f}%")
    print(f"{'='*60}\n")

    if output_file:
        with open(output_file, "w") as f:
            json.dump({
                "summary": {
                    "model": model_id,
                    "dataset": dataset_name,
                    "data_version": f"v{data_path}",
                    "total": total_count,
                    "correct": correct_count,
                    "accuracy": accuracy
                },
                "details": results
            }, f, indent=2)
        print(f"Saved details to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--data", type=str, default="gsm8k")
    parser.add_argument("--data_path", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    
    evaluate(args.model, args.data, args.data_path, args.limit)