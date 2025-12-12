"""
Solution parsing utilities
"""

import re
from typing import List, Optional


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{}"""
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
    return None


def parse_solution_into_steps(solution: str) -> List[str]:
    """Parse solution into reasoning steps"""
    steps = []
    
    # Split by code blocks
    code_pattern = r'<llm-code>.*?</llm-code>'
    parts = re.split(f'({code_pattern})', solution, flags=re.DOTALL)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # If it's a code block, keep it as one step
        if part.startswith('<llm-code>'):
            steps.append(part)
        else:
            # Split text into sentences
            sentences = re.split(r'(?<=[.!?])\s+', part)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 15:  # Filter very short fragments
                    steps.append(sent)
    
    return steps


def create_prompt(question: str) -> str:
    """Create prompt for generation"""
    return (
        f"### Question:\n{question}\n\n"
        "### Instruction:\n"
        "Solve the given math problem step by step. Break down your solution into clear, logical steps.\n\n"
        "You have three approaches:\n"
        "1. **Code-based solution**: Directly solve using Python code wrapped in <llm-code>...</llm-code> tags.\n"
        "2. **Reasoning + Code**: First explain the mathematical reasoning, then verify with code.\n"
        "3. **Pure reasoning**: Solve using mathematical explanations only (no code needed).\n\n"
        "Guidelines:\n"
        "- Present each reasoning step clearly and separately\n"
        "- Show your work for each calculation or logical deduction\n"
        "- Make each step verifiable and self-contained\n\n"
        "IMPORTANT: Always provide your final answer inside \\boxed{} at the end of your solution.\n\n"
        "### Solution:\n"
    )
