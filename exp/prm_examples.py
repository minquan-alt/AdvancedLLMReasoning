"""
Example: How to use PRM modules independently
"""

# Example 1: Parse a solution
print("="*70)
print("EXAMPLE 1: Parse Solution")
print("="*70)

from prm.parsing import extract_answer, parse_solution_into_steps

solution = """
Let's solve this step by step.

First, we identify the variables: x and y.
<llm-code>
x = 10
y = 5
result = x + y
result
</llm-code>
<llm-code-output>
15
</llm-code-output>

Therefore, the answer is \\boxed{15}.
"""

steps = parse_solution_into_steps(solution)
answer = extract_answer(solution)

print(f"\nSolution:\n{solution}")
print(f"\nExtracted {len(steps)} steps:")
for i, step in enumerate(steps, 1):
    print(f"  Step {i}: {step[:60]}...")
print(f"\nFinal answer: {answer}")


# Example 2: Compute rewards
print("\n" + "="*70)
print("EXAMPLE 2: Compute Rewards")
print("="*70)

from prm.reward import compute_reward

steps = ["Step 1", "Step 2", "Step 3"]
step_scores = [1, 1, -1]  # First two correct, last one wrong
is_correct = False  # Final answer is wrong

he_rewards = compute_reward("HE", steps, step_scores, is_correct)
se_rewards = compute_reward("SE", steps, step_scores, is_correct)

print(f"\nSteps: {len(steps)}")
print(f"Scores: {step_scores}")
print(f"Solution correct: {is_correct}")
print(f"\nHE rewards: {he_rewards}")
print(f"SE rewards: {[f'{r:.2f}' for r in se_rewards]}")


# Example 3: Answer normalization
print("\n" + "="*70)
print("EXAMPLE 3: Normalize Answers")
print("="*70)

from prm.utils import normalize_answer

test_answers = [
    "1,234.56",
    "42",
    "3.14159",
    "  100  ",
    "2/3",
]

print("\nAnswer normalization:")
for ans in test_answers:
    normalized = normalize_answer(ans)
    print(f"  '{ans}' → '{normalized}'")


# Example 4: Create prompt
print("\n" + "="*70)
print("EXAMPLE 4: Create Prompt")
print("="*70)

from prm.parsing import create_prompt

question = "If x + y = 10 and x - y = 4, what is x?"
prompt = create_prompt(question)

print(f"\nGenerated prompt:")
print(prompt[:300] + "...")


# Example 5: Load and use config
print("\n" + "="*70)
print("EXAMPLE 5: Configuration")
print("="*70)

from prm.config import PRMConfig

config = PRMConfig(
    method=1,
    reward_type="HE",
    sft_model_path="math_tutor_model/math_sft_adapter/v2/final_checkpoint",
    num_rollouts=5,
    num_samples_from_mistakes=1000,
    difficulty_threshold=0.8
)

print(f"\nPRM Configuration:")
print(f"  Method: {config.method} ({'SFT+Verifier' if config.method == 1 else 'Verifier-All'})")
print(f"  Reward Type: {config.reward_type}")
print(f"  SFT Model: {config.sft_model_path}")
print(f"  Rollouts per question: {config.num_rollouts}")
print(f"  Mistake samples: {config.num_samples_from_mistakes}")
print(f"  Difficulty threshold: {config.difficulty_threshold}")


# Example 6: Check GPU
print("\n" + "="*70)
print("EXAMPLE 6: GPU Stats")
print("="*70)

from prm.utils import stats

try:
    util, used, total = stats()
    free = total - used
    print(f"\nGPU Status:")
    print(f"  Utilization: {util}%")
    print(f"  Used: {used} MB")
    print(f"  Free: {free} MB")
    print(f"  Total: {total} MB")
except Exception as e:
    print(f"\nCannot get GPU stats: {e}")


print("\n" + "="*70)
print("EXAMPLES COMPLETE")
print("="*70)
