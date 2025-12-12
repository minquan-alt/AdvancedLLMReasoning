"""
PRM (Process Reward Model) Training Script

Supports 2 methods:
1. SFT-Generate + Verifier-Score: Use SFT model to generate CoT, then use stronger LLM to verify and score
2. Verifier-All: Use stronger LLM to generate, rollout, and score everything

Reward types: HE (Hard Exact) or SE (Soft Exact)

Usage:
    python prm_exp.py --method 1 --reward HE --sft_model path/to/sft --num_samples 1000
"""

import argparse
import os
from dotenv import load_dotenv
from datasets import Dataset

from prm import (
    PRMConfig,
    PRMDataGenerator,
    PRMTrainer,
    wait_for_gpu
)

load_dotenv()
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')


def main():
    parser = argparse.ArgumentParser(description="PRM Training")
    
    parser.add_argument("--method", type=int, required=True, choices=[1, 2],
                       help="1: SFT-Generate + Verifier-Score, 2: Verifier-All")
    parser.add_argument("--reward", type=str, required=True, choices=["HE", "SE"],
                       help="Reward type: HE (Hard Exact) or SE (Soft Exact)")
    parser.add_argument("--sft_model", type=str, required=True,
                       help="Path to SFT model checkpoint")
    parser.add_argument("--num_rollouts", type=int, default=5,
                       help="Number of solution rollouts per question")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of mistake samples to use")
    parser.add_argument("--skip_generation", action="store_true",
                       help="Skip data generation, use existing dataset")
    
    args = parser.parse_args()
    
    # Wait for GPU
    print("Waiting for GPU availability...")
    wait_for_gpu(min_free_mb=20000)
    print("GPU ready!\n")
    
    # Config
    config = PRMConfig(
        method=args.method,
        reward_type=args.reward,
        sft_model_path=args.sft_model,
        num_rollouts=args.num_rollouts,
        num_samples_from_mistakes=args.num_samples
    )
    
    # Generate PRM dataset
    dataset_path = f"data/prm_dataset_method{args.method}_{args.reward}"
    
    if not args.skip_generation:
        print("\n" + "="*70)
        print("PHASE 1: DATA GENERATION")
        print("="*70)
        
        generator = PRMDataGenerator(config, ANTHROPIC_API_KEY)
        prm_dataset = generator.generate_prm_dataset()
        
        # Save dataset
        prm_dataset.save_to_disk(dataset_path)
        print(f"✓ PRM dataset saved to {dataset_path}")
    else:
        print(f"\nLoading existing dataset from {dataset_path}...")
        prm_dataset = Dataset.load_from_disk(dataset_path)
    
    # Split train/eval
    print("\n" + "="*70)
    print("PHASE 2: DATASET SPLIT")
    print("="*70)
    
    split = prm_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Eval size: {len(eval_dataset)}")
    
    # Train PRM
    print("\n" + "="*70)
    print("PHASE 3: PRM MODEL TRAINING")
    print("="*70)
    
    prm_trainer = PRMTrainer(config)
    prm_trainer.load_prm_model()
    prm_trainer.train(train_dataset, eval_dataset)
    
    print("\n" + "="*70)
    print("✓ PRM TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: math_tutor_model/math_prm_adapter/method{args.method}_{args.reward}/")
    print(f"Dataset saved to: {dataset_path}")


if __name__ == "__main__":
    main()
