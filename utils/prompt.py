PROMPT_V0 = """### Question:
{question}

### Solution:
"""

PROMPT_V1 = """### Question:
{question}

### Instruction:
Solve the problem step by step. You can use Python code if needed.
If you write code, wrap it inside <llm-code> ... </llm-code>.
Output ONLY the final number inside \\boxed{{}}. Example: \\boxed{{42}}.

### Solution:
"""

PROMPT_V2 = """### Question:
{question}

### Instruction:
Solve the problem step by step. You can use Python code if needed.
If you write code, wrap it inside <llm-code> ... </llm-code>.
Output ONLY the final number inside \\boxed{{}}. Example: \\boxed{{42}}.

### Solution:
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

SYSTEM_PROMPT_V3 = """You are a math reasoning assistant.
Solve the problem step by step.
You can use Python code if needed.
If you write code, put it inside a Python code block:
```python
...
```
Output ONLY the final number inside \\boxed{}."""

COMPLETER_SYSTEM_PROMPT = """You are a mathematical reasoning assistant. Your task is to continue a partially completed solution to a math problem.

INPUT CONTEXT:
1. A math problem.
2. A sequence of reasoning steps that have already been generated (the "Partial Solution").

INSTRUCTIONS:
- Continue the solution logically from the EXACT state of the last step provided.
- CRITICAL: Do NOT correct, modify, or rewrite the history. Even if the previous steps contain errors, you must pretend they are correct and proceed with the logical consequences of those steps.
- Do not repeat the previous steps. Start generating immediately from where the last step ended.
- Use Python code to verify calculations or solve complex sub-problems. Wrap code in:
```python
...
```
• Continue reasoning until you reach a final answer.
• Enclose the final result strictly in LaTeX box format: \boxed{answer}."""

