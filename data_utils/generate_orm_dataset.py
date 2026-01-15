import sys
sys.path.append('/home/guest/AdvancedLLMReasoning/')

from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import os
import random
import warnings
warnings.filterwarnings("ignore")

from math_tutor_model.math_equivalence import is_equiv
from utils.prompt import SYSTEM_PROMPT_V3
from utils.inference_utils import solve_math_problem, extract_answer, load_model, clean_text

seed = 42
ran = random.Random(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# data load
ds = load_dataset("nvidia/OpenMathInstruct-1", split='train')
with open("/home/guest/AdvancedLLMReasoning/data/full_questions_seed42.json", "r", encoding="utf-8") as f:
    data = json.load(f)
groups = data['groups']
gsm8k_questions = data['gsm8k_questions']
math_questions = data['math_questions']

# model load
generator_model, generator_tokenizer = load_model()
generator_model.eval()

generator_temperature = 0.6
generator_top_p = 0.95
N_solutions = 4
GENERATOR_MAX_NEW_TOKENS = 512

def process_question(q, dataset_type, truth):
    metadata_dict = {}
    metadata_dict['question'] = q
    metadata_dict['expected_answer'] = truth
    metadata_dict['solutions'] = []
    
    for sol_idx in range(N_solutions):
        try:
            solution_metadata = {}
            
            answer = solve_math_problem(
                generator_model, 
                generator_tokenizer, 
                clean_text(q), 
                SYSTEM_PROMPT_V3, 
                max_length=GENERATOR_MAX_NEW_TOKENS,
                action='inference', 
                temperature=generator_temperature, 
                top_p=generator_top_p
            )
            
            pred = extract_answer(answer)
            solution_correct = is_equiv(pred, truth)
            
            tokens = generator_tokenizer.encode(answer)
            token_count = len(tokens)
            has_boxed = r'\boxed{' in answer
            
            if token_count > 800 and not has_boxed:
                print(f"Skipping solution {sol_idx + 1}: Too long ({token_count} tokens) and no boxed answer")
                continue
            
            solution_metadata['solution'] = answer
            solution_metadata['predicted_answer'] = pred
            solution_metadata['token_count'] = token_count
            
            if solution_correct:
                solution_metadata['label'] = "[CORRECT]"
            else:
                solution_metadata['label'] = "[INCORRECT]"
            
            metadata_dict['solutions'].append(solution_metadata)
            
        except Exception as e:
            print(f"Error processing solution {sol_idx} for question: {e}")
    
    return metadata_dict

CHECKPOINT_DIR = "/home/guest/AdvancedLLMReasoning/data/checkpoints"
GSM8K_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "gsm8k_orm_checkpoint.json")
MATH_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "math_orm_checkpoint.json")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_checkpoint(checkpoint_file, data):
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

gsm8k_metadata = load_checkpoint(GSM8K_CHECKPOINT)
processed_gsm8k = {item['question'] for item in gsm8k_metadata}

for q in tqdm(gsm8k_questions, desc="Processing GSM8K"):
    if q in processed_gsm8k:
        continue
    truth = ds[groups['gsm8k'][q][0]]['expected_answer']
    metadata_dict = process_question(q, 'gsm8k', truth)
    gsm8k_metadata.append(metadata_dict)
    
    save_checkpoint(GSM8K_CHECKPOINT, gsm8k_metadata)

math_metadata = load_checkpoint(MATH_CHECKPOINT)
processed_math = {item['question'] for item in math_metadata}

for q in tqdm(math_questions, desc="Processing MATH"):
    if q in processed_math:
        continue
    truth = ds[groups['math'][q][0]]['expected_answer']
    metadata_dict = process_question(q, 'math', truth)
    math_metadata.append(metadata_dict)
    
    save_checkpoint(MATH_CHECKPOINT, math_metadata)

output_data = {
    "gsm8k_metadata": gsm8k_metadata,
    "math_metadata": math_metadata
}

os.makedirs("/home/guest/AdvancedLLMReasoning/data", exist_ok=True)
with open("/home/guest/AdvancedLLMReasoning/data/orm_dataset.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print("\nORM Dataset generation completed!")
print(f"GSM8K: {len(gsm8k_metadata)} questions")
print(f"MATH: {len(math_metadata)} questions")
