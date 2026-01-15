import sys
sys.path.append('/home/guest/AdvancedLLMReasoning/')

from datasets import load_dataset, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
import json
from collections import defaultdict
from tqdm import tqdm
import os
import random
import warnings
import signal
from contextlib import contextmanager
warnings.filterwarnings("ignore")

from math_tutor_model.math_equivalence import is_equiv
from utils.prompt import SYSTEM_PROMPT_V3
from utils.inference_utils import solve_math_problem, extract_answer, clean_text, load_model

seed = 42
ran = random.Random(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
ds = load_dataset("nvidia/OpenMathInstruct-1", split='train')

with open("/home/guest/AdvancedLLMReasoning/data/full_questions_seed42.json", "r", encoding="utf-8") as f:
    data = json.load(f)
groups = data['groups']
math_questions = data['math_questions']

generator_model, generator_tokenizer = load_model()
generator_model.eval()

generator_temperature = 0.6
generator_top_p = 0.95
N_solutions = 4
GENERATOR_MAX_NEW_TOKENS = 512

def process_question(q, dataset_type, truth, verbose=False):
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
                continue
            
            solution_metadata['solution'] = answer
            solution_metadata['predicted_answer'] = pred
            solution_metadata['token_count'] = token_count
            
            if solution_correct:
                solution_metadata['label'] = "[CORRECT]"
            else:
                solution_metadata['label'] = "[INCORRECT]"
            
            metadata_dict['solutions'].append(solution_metadata)
            
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\nError in solution {sol_idx}: {str(e)[:100]}")
            continue
    
    return metadata_dict

CHECKPOINT_DIR = "/home/guest/AdvancedLLMReasoning/data/checkpoints"
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

math_metadata = load_checkpoint(MATH_CHECKPOINT)
processed_math = {item['question'] for item in math_metadata}

print(f"Already processed: {len(processed_math)} questions")
print(f"Remaining to process: {len(math_questions) - len(processed_math)} questions\n")

processed_count = 0
skipped_count = 0

pbar = tqdm(total=len(math_questions), desc="Processing MATH")

for idx, q in enumerate(math_questions):
    pbar.update(1)
    if q in processed_math:
        skipped_count += 1
        continue
    try:
        truth = ds[groups['math'][q][0]]['expected_answer']
        metadata_dict = process_question(q, 'math', truth)
        math_metadata.append(metadata_dict)
        save_checkpoint(MATH_CHECKPOINT, math_metadata)
        processed_count += 1
        
        if processed_count % 100 == 0:
            pbar.write(f"Processed {processed_count} new questions (skipped {skipped_count})")
        
    except KeyboardInterrupt:
        pbar.write(f"\n\nInterrupted! Processed {processed_count} new questions.")
        break
    except Exception as e:
        pbar.write(f"\nError processing question {idx}: {e}")
        pbar.write(f"Question: {q[:100]}...")
        continue
pbar.close()
print(f"\nMATH Dataset generation completed!")
