import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import re
import argparse
import json
import os
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import signal

def handler(signum, frame):
    raise TimeoutError
signal.signal(signal.SIGALRM, handler)

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

def load_model(model_id="sft"):
    if model_id == "rl":
        ADAPTER_PATH = "math_tutor_model/math_rl_adapter/final_checkpoint"
    elif model_id == "sft":
        ADAPTER_PATH = "math_tutor_model/math_sft_adapter/final_checkpoint" 
    else:
        raise ValueError("model_id phải là 'sft' hoặc 'rl'")

    print(f"Loading Base Model & Adapter...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    except:
        print("Không load được Adapter, chạy Base Model.")
        model = base_model

    model.eval()
    return model, tokenizer

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

def math_equal(pred, gold):
    if pred is None or gold is None: return False

    def clean(t): 
        return str(t).replace(",", "").replace("$", "").strip()
    
    pred, gold = clean(pred), clean(gold)
    if pred == gold: return True

    try:
        transformations = (standard_transformations + (implicit_multiplication_application,))
        signal.alarm(2) # Timeout 2s
        diff = sympy.simplify(parse_expr(pred, transformations=transformations) - 
                              parse_expr(gold, transformations=transformations))
        signal.alarm(0)
        if diff == 0: return True
    except:
        signal.alarm(0)
    
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except:
        return False

def create_prompt(question):
    return (
        f"### Question:\n{question}\n\n### Solution:\n"
    )

def evaluate(model_id="sft", dataset_name="gsm8k", num_samples=-1, output_file="eval_result.json"):
    model, tokenizer = load_model(model_id)
    
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        get_truth = lambda x: x['answer'].split("####")[-1].strip()
    elif dataset_name == "math":
        ds = load_dataset("hendrycks/competition_math", split="test")
        get_truth = lambda x: extract_answer(x['solution'])

    if num_samples > 0:
        ds = ds.select(range(min(len(ds), num_samples)))

    print(f"\n{'='*60}")
    print(f"BENCHMARK (Pass@1, Zero-shot)")
    print(f"Dataset: {dataset_name} | Samples: {len(ds)}")
    print(f"{'='*60}")

    correct = 0
    total = 0
    results = []
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    for item in tqdm(ds, desc="Evaluating"):
        question = item['question'] if dataset_name == 'gsm8k' else item['problem']
        ground_truth = get_truth(item)
        
        prompt = create_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id
            )
        
        full_res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_res = full_res.split("### Solution:\n")[-1] if "### Solution:\n" in full_res else full_res

        model_ans = extract_answer(gen_res)
        is_correct = math_equal(model_ans, ground_truth)

        if is_correct: correct += 1
        total += 1

        results.append({
            "question": question,
            "pred_raw": model_ans,
            "truth": ground_truth,
            "is_correct": is_correct
        })

    acc = correct / total
    print(f"\nFINAL RESULT: {acc:.2%} ({correct}/{total})")
    
    if output_file:
        with open(output_file, "w") as f:
            json.dump({"accuracy": acc, "details": results}, f, indent=2)
        print(f"Saved details to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sft")
    parser.add_argument("--data", type=str, default="gsm8k")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    
    evaluate(args.model, args.data, args.limit)