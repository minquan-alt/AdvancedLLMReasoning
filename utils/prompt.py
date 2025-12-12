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