import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import re
import argparse
import json
import sys
import os
import gc
import warnings
from io import StringIO
from math_tutor_model.math_equivalence import is_equiv
from utils.prompt import PROMPT_V0, PROMPT_V1, PROMPT_V2, SYSTEM_PROMPT_V3
from utils.inference_utils import execute_python_code, solve_math_problem, extract_answer

# Suppress transformer warnings
warnings.filterwarnings("ignore", message=".*The following generation config.*")
warnings.filterwarnings("ignore", category=UserWarning)

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

def load_model(model_id="sft", data_path=1):
    if model_id == "rl":
        ADAPTER_PATH = "math_tutor_model/math_rl_adapter/final_checkpoint"
    elif model_id == "sft":
        ADAPTER_PATH = f"/home/quang_ai/AdvancedLLMReasoning/math_tutor_model/math_sft_adapter/v{data_path}/final_checkpoint"
    elif model_id == "grpo":
        ADAPTER_PATH = "math_tutor_model/math_grpo_adapter/gsm8k_v1/checkpoint-100"
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
    
    if ADAPTER_PATH:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    tokenizer.padding_side = "left"
    if data_path < 3 and model_id != "grpo":
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print(f"Adapter loaded from: {ADAPTER_PATH}")
    except:
        print("Không load được Adapter.")
        exit(1)

    model.eval()
    return model, tokenizer

def execute_python_code(code_str):
    try:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        exec_globals = {}
        exec(code_str, exec_globals)
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
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
    match = re.search(r'^.*\\boxed\{[^}]+\}.*$', generated_text, re.MULTILINE)
    if match:
        trimmed_text = generated_text[:match.end()]
    else:
        trimmed_text = generated_text
    
    code_match = re.search(r'<llm-code>\s*(.*?)\s*</llm-code>', trimmed_text, re.DOTALL)
    
    if code_match:
        code_str = code_match.group(1)
        
        trimmed_text = re.sub(r'<llm>.*?</llm>', '', trimmed_text, flags=re.DOTALL)
        trimmed_text = re.sub(r'<llm-code-output>.*?</llm-code-output>', '', trimmed_text, flags=re.DOTALL)
        
        result = execute_python_code(code_str)
        
        if result:
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', trimmed_text)
            if boxed_match:
                trimmed_text = re.sub(r'\\boxed\{[^}]+\}', f'\\\\boxed{{{result}}}', trimmed_text)
            else:
                trimmed_text += f'\n\nTherefore, the answer is \\boxed{{{result}}}.'
    
    return trimmed_text

def solve_math_problem_v3(model, tokenizer, question, max_length=512, action='test', temperature=0.8, top_p=0.9):
    return solve_math_problem(model, tokenizer, question, SYSTEM_PROMPT_V3, max_length, action=action, temperature=temperature, top_p=top_p)

def load_checkpoint(checkpoint_file):
    """Load checkpoint nếu tồn tại"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✅ Loaded checkpoint: {len(data['details'])} samples completed")
            return data
    return None

def save_checkpoint(checkpoint_file, data):
    """Lưu checkpoint sau mỗi câu"""
    temp_file = checkpoint_file + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, checkpoint_file)

def evaluate(model_id=None, dataset_name="gsm8k", data_path=1, num_samples=-1):
    model, tokenizer = load_model(model_id, data_path)
    model_id = model_id if model_id in ['sft', 'rl', 'grpo'] else "base"
    
    # Setup checkpoint files
    checkpoint_dir = "/home/quang_ai/AdvancedLLMReasoning/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"test_{model_id}_{dataset_name}_v{data_path}_checkpoint.json")
    output_file = f"result_{model_id}_{dataset_name}_v{data_path}.json"
    
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
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
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"{'='*60}")

    # Load checkpoint if exists
    checkpoint_data = load_checkpoint(checkpoint_file)
    if checkpoint_data:
        results = checkpoint_data['details']
        processed_questions = {r['question'] for r in results}
        start_idx = len(results)
        print(f"📌 Resuming from sample {start_idx}")
    else:
        results = []
        processed_questions = set()
        start_idx = 0
    
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    for idx, item in enumerate(tqdm(ds, desc="Evaluating", initial=start_idx, total=len(ds))):
        question = item['question'] if dataset_name == 'gsm8k' else item['problem']
        
        # Skip if already processed
        if question in processed_questions:
            continue
        
        ground_truth = get_truth(item)
        
        try:
            # Clear CUDA cache more frequently
            if idx > 0 and idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Print progress
            print(f"\n🔄 Processing sample {idx+1}/{len(ds)}...")
            
            if data_path >= 3:
                processed_solution = solve_math_problem_v3(model, tokenizer, question, max_length=512, action='test')
            else:
                if data_path == 0:
                    prompt = PROMPT_V0.format(question=question)
                elif data_path == 1:
                    prompt = PROMPT_V1.format(question=question)
                else:
                    prompt = PROMPT_V2.format(question=question)
                
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        eos_token_id=terminators,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                processed_solution = post_process_solution_v0_v1_v2(generated_text)
            
            predicted_answer = extract_answer(processed_solution)
            correct = is_equiv(predicted_answer, ground_truth)

            result_item = {
                "question": question,
                "generated_solution": processed_solution,
                "predicted_answer": predicted_answer,
                "truth": ground_truth,
                "correct": correct
            }
            results.append(result_item)
            processed_questions.add(question)
            
            # Save checkpoint after each sample
            correct_count = sum(1 for r in results if r['correct'])
            checkpoint_data = {
                "summary": {
                    "model": model_id,
                    "dataset": dataset_name,
                    "data_version": f"v{data_path}",
                    "total": len(results),
                    "correct": correct_count,
                    "accuracy": correct_count / len(results) if results else 0,
                    "progress": f"{len(results)}/{len(ds)}"
                },
                "details": results
            }
            save_checkpoint(checkpoint_file, checkpoint_data)
            
            print(f"✅ Sample {idx+1} completed: {'CORRECT' if correct else 'INCORRECT'}")
            print(f"📊 Running accuracy: {correct_count}/{len(results)} = {correct_count/len(results)*100:.2f}%")
            
        except Exception as e:
            print(f"Error processing sample {idx+1}: {e}")
            result_item = {
                "question": question,
                "generated_solution": f"ERROR: {str(e)}",
                "predicted_answer": None,
                "truth": ground_truth,
                "correct": False,
                "error": str(e)
            }
            results.append(result_item)
            processed_questions.add(question)
            
            correct_count = sum(1 for r in results if r['correct'])
            checkpoint_data = {
                "summary": {
                    "model": model_id,
                    "dataset": dataset_name,
                    "data_version": f"v{data_path}",
                    "total": len(results),
                    "correct": correct_count,
                    "accuracy": correct_count / len(results) if results else 0,
                    "progress": f"{len(results)}/{len(ds)}",
                    "last_error": str(e)
                },
                "details": results
            }
            save_checkpoint(checkpoint_file, checkpoint_data)
            
            torch.cuda.empty_cache()
            gc.collect()
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"Correct: {correct_count}/{total_count}")
    print(f"Accuracy (PASS@1): {accuracy*100:.2f}%")
    print(f"{'='*60}\n")

    if output_file:
        final_data = {
            "summary": {
                "model": model_id,
                "dataset": dataset_name,
                "data_version": f"v{data_path}",
                "total": total_count,
                "correct": correct_count,
                "accuracy": accuracy
            },
            "details": results
        }
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved final results to {output_file}")
        
        # Clean up checkpoint file after successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"🗑️  Removed checkpoint file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--data", type=str, default="gsm8k")
    parser.add_argument("--data_path", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    
    evaluate(args.model, args.data, args.data_path, args.limit)