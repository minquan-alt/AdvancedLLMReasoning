"""
Reward computation
"""

from typing import List


def compute_hard_exact_reward(steps: List[str], step_scores: List[int], 
                              is_solution_correct: bool) -> List[float]:
    """
    Hard Exact (HE) reward: +1 if step correct AND final answer correct, else -1
    """
    rewards = []
    for score in step_scores:
        if is_solution_correct and score == 1:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def compute_soft_exact_reward(steps: List[str], step_scores: List[int], 
                              is_solution_correct: bool) -> List[float]:
    """
    Soft Exact (SE) reward: Partial credit based on step position and correctness
    Later steps get higher weight (closer to answer)
    """
    rewards = []
    num_steps = len(steps)
    
    for i, score in enumerate(step_scores):
        position_weight = (i + 1) / num_steps  # 0.1 to 1.0
        
        if score == 1:
            # Correct step: positive reward, higher for later steps
            rewards.append(0.5 + 0.5 * position_weight)
        else:
            # Incorrect step: negative reward, more severe for later steps
            rewards.append(-0.5 - 0.5 * position_weight)
    
    return rewards


def compute_reward(reward_type: str, steps: List[str], step_scores: List[int], 
                  is_solution_correct: bool) -> List[float]:
    """
    Compute reward for each step based on reward type
    
    Args:
        reward_type: "HE" or "SE"
        steps: List of reasoning steps
        step_scores: List of scores (+1/-1) for each step
        is_solution_correct: Whether final answer is correct
        
    Returns:
        List of reward values for each step
    """
    if reward_type == "HE":
        return compute_hard_exact_reward(steps, step_scores, is_solution_correct)
    elif reward_type == "SE":
        return compute_soft_exact_reward(steps, step_scores, is_solution_correct)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
