import torch
import sys
import re
from io import StringIO
from transformers import StoppingCriteria
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

class StopOnTripleBacktickNewline(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < len(self.stop_ids):
            return False
        return (
            input_ids[0, -len(self.stop_ids):].tolist()
            == self.stop_ids
        )


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


def solve_math_problem(model, tokenizer, question, system_prompt, max_length=512, action='test', temperature=0.8, top_p=0.9):
    stop_token_ids = tokenizer.encode("```\n", add_special_tokens=False)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        prompt, 
        padding=False, 
        truncation=True, 
        max_length=max_length, 
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    
    input_ids = inputs["input_ids"]
    full_response = ""
    total_generated = 0  # Tổng số token đã sinh ra
    max_iterations = 10  # Prevent infinite loops

    iteration = 0
    while total_generated < max_length and iteration < max_iterations:
        iteration += 1
        # Số token còn lại được phép sinh ra
        remaining_tokens = max_length - total_generated
        if remaining_tokens <= 0:
            break
        
        # Limit remaining tokens to prevent too long generation
        remaining_tokens = min(remaining_tokens, 256)
        
        with torch.no_grad():
            if action == 'test':
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=remaining_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    stopping_criteria=[StopOnTripleBacktickNewline(stop_token_ids)]
                )
            else: # action == 'inference'
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=remaining_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                    stopping_criteria=[StopOnTripleBacktickNewline(stop_token_ids)]
                )
                
        new_ids = output_ids[0, input_ids.shape[1]:]
        num_new = new_ids.shape[0]
        total_generated += num_new

        new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        full_response += new_text

        if "\\boxed{" in full_response:
            match = re.search(r'^.*\\boxed\{[^}]+\}.*$', full_response, re.MULTILINE)
            if match:
                full_response = full_response[:match.end()]
            break
        input_ids = output_ids

        code_match = re.search(r'```python\s*\n(.*?)\n```', full_response, re.DOTALL)

        if code_match:
            code_str = code_match.group(1)
            code_output = execute_python_code(code_str)
            
            # If code execution failed or returned None, continue without injecting output
            if code_output is None:
                code_output = "Error executing code"

            result_text = (
                "<llm>\n"
                f"{code_output}\n"
                "</llm>\n"
            )

            result_ids = tokenizer.encode(
                result_text,
                return_tensors="pt",
                add_special_tokens=False
            ).to(model.device)

            max_result_tokens = 256  # Reduced from 512
            if result_ids.shape[1] > max_result_tokens:
                result_ids = result_ids[:, :max_result_tokens]
            input_ids = torch.cat([input_ids, result_ids], dim=1)
            full_response += result_text
            continue

        # Check for end of text token
        if (output_ids[0, -1].item() == tokenizer.convert_tokens_to_ids("<|eot_id|>")):
            break
        
        # If no code block found and no boxed answer, break to prevent infinite loop
        if iteration > 5 and "```python" not in new_text and "\\boxed{" not in full_response:
            break

    return full_response

def extract_answer(text: str) -> str:
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