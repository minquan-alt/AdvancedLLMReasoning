"""
PRM Configuration
"""

from dataclasses import dataclass


@dataclass
class PRMConfig:
    """Configuration for PRM training"""
    method: int  # 1: SFT-Generate + Verifier-Score, 2: Verifier-All
    reward_type: str  # "HE" (Hard Exact) or "SE" (Soft Exact)
    sft_model_path: str
    num_rollouts: int = 5
    num_samples_from_mistakes: int = 1000
    difficulty_threshold: float = 0.8  # Loại bỏ sample quá khó (>80% fail rate)
