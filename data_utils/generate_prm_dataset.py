import sys
sys.path.append('/home/guest/AdvancedLLMReasoning/')

from datasets import load_dataset, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
from collections import defaultdict
from tqdm import tqdm
import random
import json
import os

from math_tutor_model.math_equivalence import is_equiv
from utils.prompt import SYSTEM_PROMPT_V3, COMPLETER_SYSTEM_PROMPT
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
math_questions = data['math_questions']

def load_model(model_id="sft", data_path=3, BASE_MODEL_ID="meta-llama/Llama-3.2-1B"):
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

def load_completer_model(BASE_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Load model dùng để complete/đánh giá các bước giải
    Dùng cho PRM (Process Reward Model) hoặc step completion
    """
    print(f"Loading completer model: {BASE_MODEL_ID}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
    )
    
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print("Completer model loaded successfully!")
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
                started = True
                if count == 1: 
                    continue
            elif char == "}":
                count -= 1
            if started:
                if count == 0: 
                    break
                content += char
        return content.strip()
    
    # Fallback: extract from <llm>...</llm> tag
    if "<llm>" in text and "</llm>" in text:
        match = re.search(r'<llm>(.*?)</llm>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return None

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

completer_model, completer_tokenizer = load_completer_model()
generator_model, generator_tokenizer = load_model()

completer_tokenizer.chat_template = """{{ bos_token }}
{% for message in messages -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{{ message['content'] | trim }}
{% if not loop.last or message['role'] != 'assistant' -%}
<|eot_id|>
{% endif -%}
{%- endfor %}
{% if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{%- endif %}
"""
# generator có chat_template khi train rồi nên k cần thiết lập
completer_model.eval()
generator_model.eval()

generator_temperature = 0.6
generator_top_p = 0.95

completer_temperature = 0.9
completer_top_p = 0.95
# N = 8
N_solutions = 8 # For testing, infer use N = 8
N_trajectories = 4 # For testing, infer use N = 4

# Unified configuration
GENERATOR_MAX_NEW_TOKENS = 740
COMPLETER_MAX_NEW_TOKENS = 740
COMPLETER_SYSTEM_PROMPT = """You are a math reasoning expert.
Your task is to continue a partial solution to a math problem and reach the final answer.

### Instructions:
1. **Context:** You will be given a "Problem" and a "Partial Solution".
2. **Continuity:** Continue the reasoning logic naturally from the last step of the Partial Solution.
3. **Tools:** You can use Python code if needed. If you write code, put it inside a Python code block:
```python
...
```
4. Constraints:
    ◦ CRITICAL: Do NOT repeat, rephrase, or correct the Partial Solution.
    ◦ Treat the Partial Solution as immutable history.
    ◦ You can use AT MOST ONE Python code block, DO NOT use multiple code blocks.
    ◦ STRICTLY: Your final output must be the final answer to the problem.
5. Format: After finding the result, stop generating immediately and output ONLY the final number inside \\boxed{}.
**MANDATORY** - Every response MUST contain \\boxed{number}. Responses without \\boxed{} are INVALID and will be rejected.

### WRONG Examples (DO NOT DO THIS):
❌ "So the answer is that these sacks will last for 2 days."
❌ "The number of students is 25."
❌ "The class can take 25 students on the field trip."

### CORRECT Examples:

**Example 1:**
Problem: "A car travels 60 km in 2 hours. How fast is it going?"
Partial Solution: "Let's calculate the speed."
YOUR RESPONSE: 
```python
speed = 60 / 2
```
\\boxed{30}

**Example 2:**
Problem: "What is 15 + 27?"
Partial Solution: "We need to add these two numbers."
YOUR RESPONSE: \\boxed{42}

**Example 3:**
Problem: "John has 5 apples. He buys 3 more. How many does he have?"
Partial Solution: "Let's solve this step by step.
```python
initial = 5
bought = 3
```"
YOUR RESPONSE:
```python
total = initial + bought
```
\\boxed{8}

### Your Turn:
Now continue the Partial Solution and output ONLY \\boxed{answer}."""

def process_question(q, dataset_type, truth):
    metadata_dict = {}
    metadata_dict['question'] = q
    metadata_dict['expected_answer'] = truth
    metadata_dict['question_metadata'] = []
    
    for sol_idx in range(N_solutions):
        try:
            question_metadata_dict = {}
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
            solution_answer = extract_answer(answer)
            solution_correct = is_equiv(solution_answer, truth)
            
            # Validate solution quality
            tokens = generator_tokenizer.encode(answer)
            token_count = len(tokens)
            has_boxed = r'\boxed{' in answer
            
            # Skip problematic solutions: both conditions must be true
            if token_count > 800 and not has_boxed:
                print(f"Skipping solution {sol_idx + 1}: Too long ({token_count} tokens) and no boxed answer")
                continue
            
            steps = parse_solution_into_steps(answer)
            
            question_metadata_dict['solution'] = answer
            question_metadata_dict['solution_answer'] = solution_answer
            question_metadata_dict['solution_correct'] = solution_correct
            question_metadata_dict['steps'] = steps
            question_metadata_dict['labels'] = []
            question_metadata_dict['step_metadata'] = []
            
            for step_idx in range(len(steps)-1):
                try:
                    step_metadata_dict = {}
                    past_steps_str = "\n".join(steps[:step_idx+1]) + "\n"
                    
                    messages = [
                        {"role": "system", "content": COMPLETER_SYSTEM_PROMPT},
                        {"role": "user", "content": clean_text(q)},
                        {"role": "assistant", "content": past_steps_str}
                    ]

                    prompt = completer_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    
                    inputs = completer_tokenizer(
                        prompt, 
                        padding=False, 
                        add_special_tokens=False,
                        return_tensors="pt"
                    ).to(completer_model.device)
                    
                    step_metadata_dict['previous_steps'] = past_steps_str.strip()
                    step_metadata_dict['completions'] = ['-'] * N_trajectories
                    step_metadata_dict['is_correct'] = ['-'] * N_trajectories
                    
                    # Sequential generation with early stopping
                    for traj_idx in range(N_trajectories):
                        try:
                            with torch.no_grad():
                                outputs = completer_model.generate(
                                    **inputs,
                                    max_new_tokens=COMPLETER_MAX_NEW_TOKENS,
                                    temperature=completer_temperature,
                                    top_p=completer_top_p,
                                    do_sample=True,
                                    pad_token_id=completer_tokenizer.eos_token_id,
                                    eos_token_id=completer_tokenizer.eos_token_id,
                                )
                            completion = completer_tokenizer.decode(
                                outputs[0][inputs['input_ids'].shape[1]:], 
                                skip_special_tokens=True
                            )
                            completion_answer = extract_answer(completion)
                            
                            step_metadata_dict['completions'][traj_idx] = completion.strip()
                            
                            if is_equiv(completion_answer, truth):
                                step_metadata_dict['is_correct'][traj_idx] = "[CORRECT]"
                                step_metadata_dict['label'] = "[CORRECT]"
                                question_metadata_dict['labels'].append("[CORRECT]")
                                break  # Early stopping: found correct answer
                            else:
                                step_metadata_dict['is_correct'][traj_idx] = "[INCORRECT]"
                        except Exception as e:
                            print(f"Error in trajectory {traj_idx}: {e}")
                            step_metadata_dict['is_correct'][traj_idx] = "[ERROR]"
                            
                    if 'label' not in step_metadata_dict:
                        step_metadata_dict['label'] = "[INCORRECT]"
                        question_metadata_dict['labels'].append("[INCORRECT]")
                    
                    question_metadata_dict['step_metadata'].append(step_metadata_dict)
                except Exception as e:
                    print(f"Error processing step {step_idx}: {e}")
                    
            metadata_dict['question_metadata'].append(question_metadata_dict)
        except Exception as e:
            print(f"Error processing solution {sol_idx} for question: {e}")
            
    return metadata_dict

# Checkpoint files
CHECKPOINT_DIR = "/home/guest/AdvancedLLMReasoning/data/checkpoints"
GSM8K_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "gsm8k_checkpoint.json")
MATH_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "math_checkpoint.json")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load existing checkpoints if available
def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded checkpoint from {checkpoint_file}: {len(data)} questions completed")
            return data
    return []

def save_checkpoint(checkpoint_file, data):
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Process GSM8K with checkpointing
gsm8k_metadata = load_checkpoint(GSM8K_CHECKPOINT)
processed_gsm8k = {item['question'] for item in gsm8k_metadata}

for q in tqdm(gsm8k_questions[:2], desc="Processing GSM8K"):
    if q in processed_gsm8k:
        continue
    truth = ds[groups['gsm8k'][q][0]]['expected_answer']
    metadata_dict = process_question(q, 'gsm8k', truth)
    gsm8k_metadata.append(metadata_dict)
    
    # Save checkpoint every question
    save_checkpoint(GSM8K_CHECKPOINT, gsm8k_metadata)

# Process MATH with checkpointing
math_metadata = load_checkpoint(MATH_CHECKPOINT)
processed_math = {item['question'] for item in math_metadata}

for q in tqdm(math_questions[:2], desc="Processing MATH"):
    if q in processed_math:
        continue
    truth = ds[groups['math'][q][0]]['expected_answer']
    metadata_dict = process_question(q, 'math', truth)
    math_metadata.append(metadata_dict)
    
    # Save checkpoint every question
    save_checkpoint(MATH_CHECKPOINT, math_metadata)

# lưu kết quả ra file
output_data = {
    "gsm8k_metadata": gsm8k_metadata,
    "math_metadata": math_metadata
}

os.makedirs("/home/guest/AdvancedLLMReasoning/data", exist_ok=True)
with open("/home/guest/AdvancedLLMReasoning/data/prm_dataset.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print("\n✅ Dataset generation completed!")
print(f"GSM8K: {len(gsm8k_metadata)} questions")
print(f"MATH: {len(math_metadata)} questions")
