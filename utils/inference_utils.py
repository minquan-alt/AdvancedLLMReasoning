import torch
import sys
import re
from io import StringIO
from transformers import StoppingCriteria


class StopOnTripleBacktickNewline(StoppingCriteria):
    """Stopping criteria to stop generation at ```\\n pattern."""
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
    """Execute Python code and return the output."""
    try:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        exec_globals = {}
        exec(code_str, exec_globals)
        
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # If no print output, try to get last expression value
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
    """
    Iterative inference with inline code execution.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        question: Math problem question
        system_prompt: System prompt to use
        max_length: Maximum tokens to generate
        action: 'inference' or 'test' to specify the mode of operation
    
    Returns:
        Full response string with code execution results
    """
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

    while total_generated < max_length:
        # Số token còn lại được phép sinh ra
        remaining_tokens = max_length - total_generated
        if remaining_tokens <= 0:
            break
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

            max_result_tokens = 512
            if result_ids.shape[1] > max_result_tokens:
                result_ids = result_ids[:, :max_result_tokens]
            input_ids = torch.cat([input_ids, result_ids], dim=1)
            full_response += result_text
            continue

        if (output_ids[0, -1].item() == tokenizer.convert_tokens_to_ids("<|eot_id|>")):
            break

    return full_response
