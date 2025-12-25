PROMPT_V0 = """
### Question:\n{question}\n\n### Solution:\n
"""

PROMPT_V1 = """
### Question:\n{question}\n\n### Instruction:\n
Solve the problem step by step. You can use Python code if needed.\n
If you write code, wrap it inside <llm-code> ... </llm-code>.\n
Output ONLY the final number inside \\boxed{}. Example: \\boxed{42}.\n\n
### Solution:\n
"""

PROMPT_V2 = """
### Question:\n{question}\n\n### Instruction:\n
Solve the problem step by step. You can use Python code if needed.\n
If you write code, wrap it inside <llm-code> ... </llm-code>.\n
Output ONLY the final number inside \\boxed{}. Example: \\boxed{42}.\n\n
### Solution:\n
"""

PROMPT_V3 = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a precise math assistant.
Solve the problem step by step reasoning.
You may use Python code if needed.
If you write code, wrap it inside <llm-code> ... </llm-code>.
Output ONLY the final number inside \\boxed{{}}. Example: \\boxed{{42}}.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""