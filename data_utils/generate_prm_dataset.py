import torch
import gc
import json
import os
from datasets import load_dataset, Dataset
from collections import defaultdict, deque
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re
import random
from huggingface_hub import login

login(token=os.getenv('HF_AUTH_TOKEN'))

ds = load_dataset("nvidia/OpenMathInstruct-1", split='train')

g_cycle = deque(['gsm8k', 'math'])
s_deque = {'gsm8k': deque(), 'math': deque()}
unique_questions = set()

for i, ex in enumerate(tqdm(ds, desc='Iterating dataset')):
    dataset_name = ex.get('dataset')
    question = ex.get('question')
    # chỉ xét sample đúng
    if ex['is_correct'] != True:
        continue
    if question in unique_questions or dataset_name not in ('gsm8k', 'math'):
        continue
    
    unique_questions.add(question)
    s_deque[dataset_name].append({
        'question': question,
        'answer': ex.get('expected_answer'),
    })

del ds
gc.collect()
print(f"Loaded {len(unique_questions)} unique questions")

# ============= Helper Functions =============
def parse_solution_into_steps(solution):
    steps = []
    
    block_pattern = r'(<llm-code>.*?</llm-code>|<llm-code-output>.*?</llm-code-output>)'
    parts = re.split(block_pattern, solution, flags=re.DOTALL)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # code block or code-output block, keep as one step
        if part.startswith('<llm-code>') or part.startswith('<llm-code-output>'):
            steps.append(part)
        
        # Plain text - split by sentences (dấu chấm)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', part)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    steps.append(sent)
    
    return steps

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def save_checkpoint(prm_dataset, total_steps, total_questions, checkpoint_num):
    """Save checkpoint to disk and return empty list to free memory"""
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"prm_dataset_checkpoint_{checkpoint_num}.json")
    
    # Chỉ lưu data mới từ checkpoint này
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(prm_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n[Checkpoint {checkpoint_num}] Saved: {len(prm_dataset)} samples, {total_steps} total steps")
    
    # Return empty list để giải phóng memory
    return []

def load_latest_checkpoint():
    """Load metadata from latest checkpoint"""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return 0, 0, 0
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("prm_dataset_checkpoint_")]
    if not checkpoints:
        return 0, 0, 0
    
    # Get latest checkpoint number
    checkpoint_nums = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    latest_num = max(checkpoint_nums)
    
    # Count total steps and questions từ tất cả checkpoints
    total_steps = 0
    total_questions = 0
    
    for num in sorted(checkpoint_nums):
        checkpoint_path = os.path.join(checkpoint_dir, f"prm_dataset_checkpoint_{num}.json")
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            total_questions += len(data)
            for item in data:
                total_steps += len(item['solution_steps'])
    
    print(f"Resuming from checkpoint {latest_num}: {total_questions} questions, {total_steps} steps")
    return total_steps, total_questions, latest_num

# ============= Load Model =============
print("Loading model...")
ADAPTER_PATH = "/home/guest/AdvancedLLMReasoning/math_tutor_model/math_sft_adapter/v2/final_checkpoint" 
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'  # Fix cho decoder-only models

sft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
sft_model.eval()

# ============= Setup =============
seed = 42
random.seed(seed)

instruction = (
    "Solve the problem step by step. You can use Python code if needed.\n"
    "If you write code, wrap it inside <llm-code> ... </llm-code>.\n"
    "Output ONLY the final number inside \\boxed{}."
)

# ============= Main Generation Loop =============
TARGET_SIZE = 200000
BATCH_SIZE = 64
CHECKPOINT_INTERVAL = 5000  # Save mỗi 5000 steps

# Load checkpoint nếu có
total_steps, total_questions, checkpoint_num = load_latest_checkpoint()

# prm_dataset chỉ lưu tạm, sẽ flush mỗi checkpoint
prm_dataset = []

g_cycle = deque(['gsm8k', 'math'])
batch_questions = []
batch_answers = []
batch_prompts = []

print(f"\nStarting generation from step {total_steps}/{TARGET_SIZE}")
print(f"Batch size: {BATCH_SIZE}, Checkpoint every {CHECKPOINT_INTERVAL} steps\n")

with tqdm(total=TARGET_SIZE, initial=total_steps, desc="Steps collected") as pbar:
    while total_steps < TARGET_SIZE:
        # Thu thập batch
        while len(batch_prompts) < BATCH_SIZE:
            g = g_cycle.popleft()
            g_cycle.append(g)

            if not s_deque[g]:
                continue

            s = s_deque[g].popleft()
            question = s['question']
            answer = s['answer']
            
            prompt = (
                f"### Question:\n{clean_text(question)}\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Solution:\n"
            )
            
            batch_prompts.append(prompt)
            batch_questions.append(question)
            batch_answers.append(answer)
        
        # Wrap generation trong try-except để handle OOM
        try:
            # Tokenize và generate cho batch
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(sft_model.device)

            with torch.no_grad():
                outputs = sft_model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    temperature=0.7, 
                    top_p=0.9, 
                    do_sample=True, 
                    pad_token_id=tokenizer.eos_token_id
                )
        
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n⚠ CUDA OOM detected! Saving progress and waiting for memory...")
            
            # Clear GPU cache
            del inputs
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()
            gc.collect()
            
            # Save current progress nếu có data
            if prm_dataset:
                checkpoint_num += 1
                prm_dataset = save_checkpoint(prm_dataset, total_steps, total_questions, checkpoint_num)
            
            # Wait for GPU memory
            wait_for_gpu_memory(required_gb=10, check_interval=30)
            
            # Retry batch này
            print("Retrying current batch...")
            continue

        # Xử lý từng output trong batch
        for idx in range(len(batch_prompts)):
            generated_ids = outputs[idx][inputs["input_ids"].shape[-1]:]
            solution = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            steps = parse_solution_into_steps(solution)
            if not steps:
                continue

            num_steps = len(steps)
            if total_steps + num_steps > TARGET_SIZE:
                break

            total_steps += num_steps
            total_questions += 1
            pbar.update(num_steps)

            prm_dataset.append({
                "question": batch_questions[idx],
                "expected_answer": batch_answers[idx],
                "solution_steps": steps
            })
        
        # Clear batch và GPU cache
        batch_prompts = []
        batch_questions = []
        batch_answers = []
        del inputs, outputs
        torch.cuda.empty_cache()
        
        # Save checkpoint và flush data
        if total_steps // CHECKPOINT_INTERVAL > checkpoint_num:
            checkpoint_num = total_steps // CHECKPOINT_INTERVAL
            prm_dataset = save_checkpoint(prm_dataset, total_steps, total_questions, checkpoint_num)
            gc.collect()  # Garbage collection
        
        # Check nếu hết data
        if not s_deque['gsm8k'] and not s_deque['math']:
            print("\nRan out of data!")
            break

# ============= Save Final Dataset =============
# Save phần còn lại nếu có
if prm_dataset:
    checkpoint_num += 1
    save_checkpoint(prm_dataset, total_steps, total_questions, checkpoint_num)

# Merge tất cả checkpoints thành 1 file final
print("\nMerging all checkpoints into final dataset...")
checkpoint_dir = "checkpoints"
all_data = []

checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("prm_dataset_checkpoint_")],
                     key=lambda x: int(x.split('_')[-1].split('.')[0]))

for ckpt_file in checkpoints:
    checkpoint_path = os.path.join(checkpoint_dir, ckpt_file)
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_data.extend(data)

final_path = "prm_dataset_final.json"
with open(final_path, 'w', encoding='utf-8') as f:
    json.dump({
        "total_steps": total_steps,
        "total_questions": total_questions,
        "dataset": all_data
    }, f, ensure_ascii=False, indent=2)

print(f"\n✓ Final dataset saved to {final_path}")
print(f"Total questions: {total_questions}")
print(f"Total steps: {total_steps}")
