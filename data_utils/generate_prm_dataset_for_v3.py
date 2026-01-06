import torch
import gc
import json
import os
import sys
from datasets import load_dataset, Dataset
from collections import defaultdict, deque
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re
import random
from huggingface_hub import login

from utils.prompt import SYSTEM_PROMPT_V3
from utils.inference_utils import StopOnTripleBacktickNewline, execute_python_code, solve_math_problem

from dotenv import load_dotenv
load_dotenv()
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
len_questions = len(unique_questions)

# ============= Helper Functions =============
def parse_solution_into_steps(solution):
    """
    Parse solution into steps, handling code blocks with outputs (v3 format).
    Code blocks + <llm>...</llm> outputs are single steps.
    Text is split by sentences.
    """
    steps = []
    
    # Pattern: code block followed by <llm> output
    pattern = r'(```python\s*\n.*?\n```\s*<llm>.*?</llm>)'
    parts = re.split(pattern, solution, flags=re.DOTALL)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Code block with output
        if part.startswith('```python'):
            steps.append(part)
        else:
            # Split remaining text by sentences
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

# ============= Load Model =============
print("Loading model...")
ADAPTER_PATH = "/home/guest/AdvancedLLMReasoning/math_tutor_model/math_sft_adapter/v3/final_checkpoint" 
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
tokenizer.padding_side = 'left'  # left for implement

tokenizer.chat_template = """{{ bos_token }}
{% for message in messages -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>{{ message['content'] | trim }}
<|eot_id|>
{%- endfor %}
{% if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{%- endif %}
"""

sft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
sft_model.eval()

# ============= Setup =============
seed = 42
random.seed(seed)
system_prompt = SYSTEM_PROMPT_V3

# ============= Main Generation Loop =============
GSM8K_STEPS_TARGET = 100000
MATH_STEPS_TARGET = 200000

# dùng để reset khi hết data
original_data = {
    'gsm8k': list(s_deque['gsm8k']),
    'math': list(s_deque['math'])
}

# prm_dataset sẽ lưu kết quả
prm_dataset = []

gsm8k_steps_count = 0
math_steps_count = 0
total_questions = 0

batch_questions = []
batch_answers = []
batch_prompts = []
batch_datasets = [] # lưu loại dataset của từng sample trong batch

print(f"\nStarting generation...")
print(f"Targets: GSM8K={GSM8K_STEPS_TARGET}, MATH={MATH_STEPS_TARGET}")
print(f"Sequential iterative inference\n")

total_target = GSM8K_STEPS_TARGET + MATH_STEPS_TARGET

with tqdm(total=total_target, desc="Steps collected") as pbar:
    while gsm8k_steps_count < GSM8K_STEPS_TARGET or math_steps_count < MATH_STEPS_TARGET:
        
        # Chọn dataset nào để lấy sample
        current_dataset = None
        
        need_gsm8k = gsm8k_steps_count < GSM8K_STEPS_TARGET
        need_math = math_steps_count < MATH_STEPS_TARGET
        
        if need_gsm8k and need_math:
            current_dataset = g_cycle[0]
            g_cycle.rotate(-1)
        elif need_gsm8k:
            current_dataset = 'gsm8k'
        elif need_math:
            current_dataset = 'math'
        else:
            break

        # Lấy sample từ queue
        dq = s_deque[current_dataset]
        
        if not dq:
            random.shuffle(original_data[current_dataset])
            dq.extend(original_data[current_dataset])
            s_deque[current_dataset] = dq

        s = dq.popleft()
        question = s['question']
        answer = s['answer']
        
        # Sử dụng inference mới với iterative generation
        solution = solve_math_problem(sft_model, tokenizer, clean_text(question), SYSTEM_PROMPT_V3, max_length=512)
        
        steps = parse_solution_into_steps(solution)
        if not steps:
            continue

        num_steps = len(steps)
        
        # Check xem dataset này còn cần thêm steps không
        if current_dataset == 'gsm8k':
            if gsm8k_steps_count >= GSM8K_STEPS_TARGET:
                continue
            gsm8k_steps_count += num_steps
        else:
            if math_steps_count >= MATH_STEPS_TARGET:
                continue
            math_steps_count += num_steps

        total_questions += 1
        pbar.update(num_steps)

        prm_dataset.append({
            "question": question,
            "expected_answer": answer,
            "solution_steps": steps,
            "dataset": current_dataset
        })
        
        # Clear GPU cache
        torch.cuda.empty_cache()

# ============= Save Final Dataset =============
final_path = "/home/guest/AdvancedLLMReasoning/data/prm_dataset_final.json"
os.makedirs(os.path.dirname(final_path), exist_ok=True)
with open(final_path, 'w', encoding='utf-8') as f:
    json.dump({
        "total_steps": gsm8k_steps_count + math_steps_count,
        "gsm8k_steps": gsm8k_steps_count,
        "math_steps": math_steps_count,
        "total_questions": total_questions,
        "dataset": prm_dataset
    }, f, ensure_ascii=False, indent=2)

print(f"\n✓ Final dataset saved to {final_path}")
print(f"Total questions: {total_questions}")
print(f"GSM8K Steps: {gsm8k_steps_count}")
print(f"MATH Steps: {math_steps_count}")