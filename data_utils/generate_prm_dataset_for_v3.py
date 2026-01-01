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
len_questions = len(unique_questions)

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
# tokenizer.pad_token = tokenizer.eos_token
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

# ============= Main Generation Loop =============
GSM8K_STEPS_TARGET = 100000
MATH_STEPS_TARGET = 200000
BATCH_SIZE = 64

# Backup data ban đầu để reset khi cần
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
batch_datasets = [] # Lưu loại dataset của từng sample trong batch

print(f"\nStarting generation...")
print(f"Targets: GSM8K={GSM8K_STEPS_TARGET}, MATH={MATH_STEPS_TARGET}")
print(f"Batch size: {BATCH_SIZE}\n")

total_target = GSM8K_STEPS_TARGET + MATH_STEPS_TARGET

with tqdm(total=total_target, desc="Steps collected") as pbar:
    while gsm8k_steps_count < GSM8K_STEPS_TARGET or math_steps_count < MATH_STEPS_TARGET:
        
        # Thu thập batch
        while len(batch_prompts) < BATCH_SIZE:
            
            # Chọn dataset nào để lấy sample
            # Logic: Lấy từ dataset chưa đủ chỉ tiêu. Nếu cả 2 đều chưa đủ, lấy xen kẽ.
            current_dataset = None
            
            need_gsm8k = gsm8k_steps_count < GSM8K_STEPS_TARGET
            need_math = math_steps_count < MATH_STEPS_TARGET
            
            if need_gsm8k and need_math:
                current_dataset = g_cycle[0]
                g_cycle.rotate(-1) # Xoay vòng
            elif need_gsm8k:
                current_dataset = 'gsm8k'
            elif need_math:
                current_dataset = 'math'
            else:
                break # Đã đủ cả 2

            # Lấy sample từ queue
            dq = s_deque[current_dataset]
            
            # Nếu hết data, reset từ backup
            if not dq:
                random.shuffle(original_data[current_dataset]) # Shuffle cho đa dạng
                dq.extend(original_data[current_dataset])
                s_deque[current_dataset] = dq
                # print(f"\nReset queue for {current_dataset}")

            s = dq.popleft()
            question = s['question']
            answer = s['answer']
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": clean_text(question)}
            ]

            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            batch_prompts.append(prompt)
            batch_questions.append(question)
            batch_answers.append(answer)
            batch_datasets.append(current_dataset)
        
        # Nếu đã đủ chỉ tiêu cả 2 mà batch trống -> Dừng
        if len(batch_prompts) == 0:
            break
        
        # Tokenize và generate cho batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(sft_model.device)

        with torch.no_grad():
            outputs = sft_model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=1.0, 
                top_p=0.95, 
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id
            )

        # Xử lý từng output trong batch
        for idx in range(len(batch_prompts)):
            generated_ids = outputs[idx][inputs["input_ids"].shape[-1]:]
            solution = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            steps = parse_solution_into_steps(solution)
            if not steps:
                continue

            num_steps = len(steps)
            ds_type = batch_datasets[idx]
            
            # Check xem dataset này còn cần thêm steps không
            if ds_type == 'gsm8k':
                if gsm8k_steps_count >= GSM8K_STEPS_TARGET:
                    continue # Bỏ qua nếu đã đủ
                gsm8k_steps_count += num_steps
            else: # math
                if math_steps_count >= MATH_STEPS_TARGET:
                    continue
                math_steps_count += num_steps

            total_questions += 1
            pbar.update(num_steps)

            prm_dataset.append({
                "question": batch_questions[idx],
                "expected_answer": batch_answers[idx],
                "solution_steps": steps,
                "dataset": ds_type
            })
        
        # Clear batch và GPU cache
        batch_prompts = []
        batch_questions = []
        batch_answers = []
        batch_datasets = []
        del inputs, outputs
        torch.cuda.empty_cache()

# ============= Save Final Dataset =============
final_path = "prm_dataset_final.json"
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