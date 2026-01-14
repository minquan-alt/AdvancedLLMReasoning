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
warnings.filterwarnings("ignore")

from math_tutor_model.math_equivalence import is_equiv
from utils.prompt import SYSTEM_PROMPT_V3
from utils.inference_utils import solve_math_problem

seed = 42
ran = random.Random(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
ds = load_dataset("nvidia/OpenMathInstruct-1", split='train')

with open("/home/guest/AdvancedLLMReasoning/data/full_questions_seed42.json", "r", encoding="utf-8") as f:
    data = json.load(f)
groups = data['groups']
gsm8k_questions = data['gsm8k_questions']

def load_model(model_id="sft", data_path=3, BASE_MODEL_ID="meta-llama/Llama-3.2-1B"):
    if model_id == "rl":
        ADAPTER_PATH = "math_tutor_model/math_rl_adapter/final_checkpoint"
    elif model_id == "sft":
        ADAPTER_PATH = f"/home/guest/AdvancedLLMReasoning/math_tutor_model/math_sft_adapter/v{data_path}/final_checkpoint" 
    else:
        ADAPTER_PATH = None

    print(f"Loading model for GSM8K...")
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
    if data_path < 3:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print(f"Adapter loaded from: {ADAPTER_PATH}")
    except:
        print("Không load được Adapter.")
        exit(1)

    model.eval()
    return model, tokenizer

def extract_answer(text: str) -> str:
    """
    Extract answer from \\boxed{} or <llm>...</llm> as fallback
    Priority: \\boxed{} > <llm>...</llm>
    """
    # Try to extract from \boxed{} first
    if "\\boxed{" in text:
        idx = text.rfind("\\boxed{")
        content = ""
        count = 0
        started = False
        for char in text[idx:]:
            if char == "{":
                count += 1
                if count == 1:
                    started = True
                    continue
            elif char == "}":
                count -= 1
                if count == 0:
                    break
            if started:
                content += char
        return content.strip()
    
    # Fallback: extract from <llm>...</llm> tag
    if "<llm>" in text and "</llm>" in text:
        match = re.search(r'<llm>(.*?)</llm>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

# Load generator model
generator_model, generator_tokenizer = load_model()
generator_model.eval()

generator_temperature = 0.6
generator_top_p = 0.95
N_solutions = 4
GENERATOR_MAX_NEW_TOKENS = 512

def process_question(q, dataset_type, truth):
    """
    Process a question for ORM dataset generation.
    ORM evaluates complete solutions (outcome-based), not individual steps.
    """
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

# Checkpoint files
CHECKPOINT_DIR = "/home/guest/AdvancedLLMReasoning/data/checkpoints"
GSM8K_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "gsm8k_orm_checkpoint.json")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# load ckpt
def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_checkpoint(checkpoint_file, data):
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# process GSM8K
print("=" * 60)
print("Processing GSM8K Dataset")
print("=" * 60)

gsm8k_metadata = load_checkpoint(GSM8K_CHECKPOINT)
processed_gsm8k = {item['question'] for item in gsm8k_metadata}

for q in tqdm(gsm8k_questions, desc="Processing GSM8K"):
    if q in processed_gsm8k:
        continue
    truth = ds[groups['gsm8k'][q][0]]['expected_answer']
    metadata_dict = process_question(q, 'gsm8k', truth)
    gsm8k_metadata.append(metadata_dict)
    
    save_checkpoint(GSM8K_CHECKPOINT, gsm8k_metadata)

print(f"\nGSM8K Dataset generation completed!")
print(f"Total GSM8K questions: {len(gsm8k_metadata)}")
