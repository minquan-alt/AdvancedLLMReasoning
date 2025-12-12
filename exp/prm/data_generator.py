"""
Data generation for PRM training
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import anthropic

from .config import PRMConfig
from .parsing import extract_answer, parse_solution_into_steps, create_prompt
from .utils import normalize_answer
from .reward import compute_reward


class PRMDataGenerator:
    """Generate PRM training data from mistakes"""
    
    def __init__(self, config: PRMConfig, anthropic_api_key: str):
        self.config = config
        self.sft_model = None
        self.sft_tokenizer = None
        self.verifier_client = None
        self.anthropic_api_key = anthropic_api_key
        
    def load_sft_model(self):
        """Load SFT model for generation (Method 1)"""
        print(f"Loading SFT model from {self.config.sft_model_path}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        self.sft_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.sft_tokenizer.pad_token = self.sft_tokenizer.eos_token
        
        try:
            self.sft_model = PeftModel.from_pretrained(base_model, self.config.sft_model_path)
            self.sft_model.eval()
            print("✓ SFT model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load SFT adapter: {e}")
            self.sft_model = base_model
            
    def load_verifier(self):
        """Load verifier (Claude Sonnet 4)"""
        if self.anthropic_api_key:
            self.verifier_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            print("✓ Verifier (Claude Sonnet 4) loaded")
        else:
            print("✗ ANTHROPIC_API_KEY not found. Verifier unavailable.")
    
    def get_mistake_samples(self) -> Dataset:
        """
        Extract samples from OpenMathInstruct-1 where SFT model frequently fails
        Filter out too-difficult problems
        """
        print("\n" + "="*70)
        print("EXTRACTING MISTAKE SAMPLES")
        print("="*70)
        
        # Load OpenMathInstruct-1
        print("Loading OpenMathInstruct-1 dataset...")
        full_ds = load_dataset("nvidia/OpenMathInstruct-1", split='train')
        
        # Filter for correct samples only
        print("Filtering correct samples...")
        correct_samples = [ex for ex in tqdm(full_ds, desc="Filtering") 
                          if ex.get('is_correct', False)]
        
        print(f"Found {len(correct_samples)} correct samples")
        
        # Test SFT model on a subset to find mistakes
        test_size = min(5000, len(correct_samples))
        test_samples = correct_samples[:test_size]
        
        print(f"\nTesting SFT model on {test_size} samples to find mistakes...")
        
        mistake_questions = []
        
        for sample in tqdm(test_samples, desc="Testing SFT"):
            question = sample['question']
            ground_truth_solution = sample['generated_solution']
            ground_truth_answer = extract_answer(ground_truth_solution)
            
            if not ground_truth_answer:
                continue
            
            # Generate with SFT model
            prompt = create_prompt(question)
            inputs = self.sft_tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = self.sft_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=3,  # Multiple attempts
                    pad_token_id=self.sft_tokenizer.eos_token_id
                )
            
            # Check if any attempt is correct
            all_wrong = True
            for output in outputs:
                generated = self.sft_tokenizer.decode(output, skip_special_tokens=True)
                pred_answer = extract_answer(generated)
                
                if pred_answer and normalize_answer(pred_answer) == normalize_answer(ground_truth_answer):
                    all_wrong = False
                    break
            
            if all_wrong:
                mistake_questions.append({
                    'question': question,
                    'ground_truth_solution': ground_truth_solution,
                    'ground_truth_answer': ground_truth_answer,
                    'fail_count': 3
                })
        
        # Filter by difficulty: keep questions with moderate fail rate
        filtered_mistakes = [
            m for m in mistake_questions 
            if 0.3 <= (m['fail_count'] / 3) <= self.config.difficulty_threshold
        ]
        
        print(f"\nFound {len(mistake_questions)} questions with mistakes")
        print(f"After difficulty filtering: {len(filtered_mistakes)} samples")
        
        # Sample the desired number
        import random
        random.shuffle(filtered_mistakes)
        final_samples = filtered_mistakes[:self.config.num_samples_from_mistakes]
        
        print(f"Selected {len(final_samples)} samples for PRM training")
        
        return Dataset.from_list(final_samples)
    
    def generate_with_sft(self, question: str, num_rollouts: int) -> List[str]:
        """Generate multiple solutions using SFT model (Method 1)"""
        prompt = create_prompt(question)
        inputs = self.sft_tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.sft_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.8,
                do_sample=True,
                num_return_sequences=num_rollouts,
                pad_token_id=self.sft_tokenizer.eos_token_id,
                top_p=0.95
            )
        
        solutions = []
        for output in outputs:
            solution = self.sft_tokenizer.decode(output, skip_special_tokens=True)
            # Extract only the solution part
            if "### Solution:" in solution:
                solution = solution.split("### Solution:")[-1].strip()
            solutions.append(solution)
        
        return solutions
    
    def generate_with_verifier(self, question: str, num_rollouts: int) -> List[str]:
        """Generate multiple solutions using Claude (Method 2)"""
        solutions = []
        
        prompt = f"""Solve this math problem step by step. Generate {num_rollouts} different solution approaches.

Question: {question}

For each solution:
1. Break down into clear reasoning steps
2. You can use Python code in <llm-code>...</llm-code> tags if needed
3. Make each step verifiable
4. Provide final answer in \\boxed{{answer}}

Generate {num_rollouts} diverse solutions:"""

        try:
            response = self.verifier_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                temperature=0.8,
                messages=[{"role": "user", "content": prompt}]
            )
            
            full_response = response.content[0].text
            
            # Split into individual solutions
            solution_parts = re.split(r'Solution \d+:', full_response)
            solutions = [s.strip() for s in solution_parts if s.strip()][:num_rollouts]
            
        except Exception as e:
            print(f"Verifier generation failed: {e}")
            solutions = []
        
        return solutions
    
    def score_solution_with_verifier(self, question: str, solution: str, 
                                    ground_truth_answer: str) -> Tuple[List[int], str]:
        """
        Use Claude to score each step in the solution
        Returns: (step_scores, explanation)
        """
        steps = parse_solution_into_steps(solution)
        
        prompt = f"""You are a math expert verifying a solution step-by-step.

Question: {question}

Ground Truth Answer: {ground_truth_answer}

Solution to verify:
{solution}

I've broken this into {len(steps)} steps. For EACH step, determine if it is:
- CORRECT (+1): The reasoning or calculation is valid and moves toward the right answer
- INCORRECT (-1): The step contains errors, wrong logic, or leads away from correct answer

Return your analysis in this exact JSON format:
{{
  "steps": [
    {{"step_number": 1, "score": 1, "reasoning": "..."}},
    {{"step_number": 2, "score": -1, "reasoning": "..."}},
    ...
  ],
  "overall_correctness": "correct/incorrect"
}}

Steps to verify:
{chr(10).join([f"{i+1}. {step[:200]}..." for i, step in enumerate(steps)])}
"""

        try:
            response = self.verifier_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                scores = [s['score'] for s in result['steps']]
                return scores[:len(steps)], result.get('overall_correctness', 'unknown')
            
        except Exception as e:
            print(f"Verifier scoring failed: {e}")
        
        # Fallback: simple rule-based scoring
        predicted_answer = extract_answer(solution)
        is_correct = (predicted_answer and 
                     normalize_answer(predicted_answer) == normalize_answer(ground_truth_answer))
        
        if is_correct:
            return [1] * len(steps), "correct"
        else:
            scores = [1] * max(0, len(steps) - 2) + [-1] * min(2, len(steps))
            return scores, "incorrect"
    
    def generate_prm_dataset(self) -> Dataset:
        """
        Main function to generate PRM training dataset
        """
        print("\n" + "="*70)
        print(f"GENERATING PRM DATASET - Method {self.config.method}")
        print(f"Reward Type: {self.config.reward_type}")
        print("="*70)
        
        # Load models based on method
        if self.config.method == 1:
            self.load_sft_model()
        self.load_verifier()
        
        # Get mistake samples
        mistake_samples = self.get_mistake_samples()
        
        prm_data = []
        
        for sample in tqdm(mistake_samples, desc="Generating PRM data"):
            question = sample['question']
            ground_truth_answer = sample['ground_truth_answer']
            
            # Generate solutions
            if self.config.method == 1:
                solutions = self.generate_with_sft(question, self.config.num_rollouts)
            else:
                solutions = self.generate_with_verifier(question, self.config.num_rollouts)
            
            # Process each solution
            for solution in solutions:
                steps = parse_solution_into_steps(solution)
                if len(steps) < 2:
                    continue
                
                # Score each step
                step_scores, correctness = self.score_solution_with_verifier(
                    question, solution, ground_truth_answer
                )
                
                # Ensure alignment
                if len(step_scores) != len(steps):
                    step_scores = step_scores[:len(steps)] + [0] * (len(steps) - len(step_scores))
                
                is_correct = (correctness == "correct")
                
                # Compute rewards
                step_rewards = compute_reward(
                    self.config.reward_type, steps, step_scores, is_correct
                )
                
                prm_data.append({
                    'question': question,
                    'solution': solution,
                    'steps': steps,
                    'step_rewards': step_rewards,
                    'is_correct': is_correct,
                    'ground_truth_answer': ground_truth_answer
                })
        
        print(f"\n✓ Generated {len(prm_data)} PRM training samples")
        
        return Dataset.from_list(prm_data)
