"""
PRM (Process Reward Model) Training Script

Supports 2 methods:
1. SFT-Generate + Verifier-Score: Use SFT model to generate CoT, then use stronger LLM to verify and score
2. Verifier-All: Use stronger LLM to generate, rollout, and score everything

Reward types: HE (Hard Exact) or SE (Soft Exact)
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
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')
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
        generator = PRMDataGenerator(config, ANTHROPIC_API_KEY)
        prm_dataset = generator.generate_prm_dataset()
        
        # Save dataset
        prm_dataset.save_to_disk(dataset_path)
        print(f"✓ PRM dataset saved to {dataset_path}")
    else:
        print(f"Loading existing dataset from {dataset_path}...")
        prm_dataset = Dataset.load_from_disk(dataset_path)
    
    # Split train/eval
    split = prm_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    print(f"\nTrain size: {len(train_dataset)}")
    print(f"Eval size: {len(eval_dataset)}")
    
    # Train PRM
    prm_trainer = PRMTrainer(config)
    prm_trainer.load_prm_model()
    prm_trainer.train(train_dataset, eval_dataset)
    
    print("\n" + "="*70)
    print("PRM TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

    """Generate PRM training data from mistakes"""
    
    def __init__(self, config: PRMConfig):
        self.config = config
        self.sft_model = None
        self.sft_tokenizer = None
        self.verifier_client = None
        
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
        if ANTHROPIC_API_KEY:
            self.verifier_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            print("✓ Verifier (Claude Sonnet 4) loaded")
        else:
            print("✗ ANTHROPIC_API_KEY not found. Verifier unavailable.")
            
    def extract_answer(self, text: str) -> Optional[str]:
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
                    if count == 1: continue
                elif char == "}":
                    count -= 1
                if started:
                    if count == 0: break
                    content += char
            return content.strip()
        return None
    
    def parse_solution_into_steps(self, solution: str) -> List[str]:
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
                    # if len(sent) > 15:  # Filter very short fragments
                    steps.append(sent)
        
        return steps
    
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
        mistake_stats = defaultdict(int)
        
        for sample in tqdm(test_samples, desc="Testing SFT"):
            question = sample['question']
            ground_truth_solution = sample['generated_solution']
            ground_truth_answer = self.extract_answer(ground_truth_solution)
            
            if not ground_truth_answer:
                continue
            
            # Generate with SFT model
            prompt = self.create_prompt(question)
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
                pred_answer = self.extract_answer(generated)
                
                if pred_answer and self.normalize_answer(pred_answer) == self.normalize_answer(ground_truth_answer):
                    all_wrong = False
                    break
            
            if all_wrong:
                mistake_stats[question] += 1
                mistake_questions.append({
                    'question': question,
                    'ground_truth_solution': ground_truth_solution,
                    'ground_truth_answer': ground_truth_answer,
                    'fail_count': 3
                })
        
        # Filter by difficulty: keep questions with moderate fail rate
        # Too easy (always correct) or too hard (always wrong) are not useful
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
    
    def create_prompt(self, question: str) -> str:
        """Create prompt for generation"""
        return (
            f"### Question:\n{question}\n\n"
            "### Instruction:\n"
            "Solve the problem step by step. You can use Python code if needed.\n"
            "If you write code, wrap it inside <llm-code> ... </llm-code>.\n"
            "Output ONLY the final number inside \\boxed{}.\n\n"
            "### Solution:\n"
        )
    
    def normalize_answer(self, ans: str) -> str:
        """Normalize answer for comparison"""
        ans = ans.replace(',', '').replace(' ', '').lower()
        # Try to parse as number
        try:
            return str(float(ans))
        except:
            return ans
    
    def generate_with_sft(self, question: str, num_rollouts: int) -> List[str]:
        """Generate multiple solutions using SFT model (Method 1)"""
        prompt = self.create_prompt(question)
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
            
            # Split into individual solutions (assume separated by "Solution X:" or similar)
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
        steps = self.parse_solution_into_steps(solution)
        
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
        predicted_answer = self.extract_answer(solution)
        is_correct = (predicted_answer and 
                     self.normalize_answer(predicted_answer) == self.normalize_answer(ground_truth_answer))
        
        # If final answer correct, all steps +1; if wrong, last few steps -1
        if is_correct:
            return [1] * len(steps), "correct"
        else:
            scores = [1] * max(0, len(steps) - 2) + [-1] * min(2, len(steps))
            return scores, "incorrect"
    
    def compute_reward(self, steps: List[str], step_scores: List[int], 
                      is_solution_correct: bool) -> List[float]:
        """
        Compute reward for each step based on reward type
        
        HE (Hard Exact): +1 if step correct AND final answer correct, else -1
        SE (Soft Exact): Gradual reward based on step position and correctness
        """
        rewards = []
        
        if self.config.reward_type == "HE":
            # Hard Exact: Only reward if entire solution correct
            for score in step_scores:
                if is_solution_correct and score == 1:
                    rewards.append(1.0)
                else:
                    rewards.append(-1.0)
                    
        elif self.config.reward_type == "SE":
            # Soft Exact: Partial credit based on step correctness
            # Later steps get higher weight (closer to answer)
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
    
    def generate_prm_dataset(self) -> Dataset:
        """
        Main function to generate PRM training dataset
        
        Returns dataset with format:
        {
            'question': str,
            'solution': str,  # Full solution
            'steps': List[str],  # Individual steps
            'step_rewards': List[float],  # Reward for each step
            'is_correct': bool
        }
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
                # Method 1: SFT generates, Claude scores
                solutions = self.generate_with_sft(question, self.config.num_rollouts)
            else:
                # Method 2: Claude generates and scores
                solutions = self.generate_with_verifier(question, self.config.num_rollouts)
            
            # Process each solution
            for solution in solutions:
                # Parse into steps
                steps = self.parse_solution_into_steps(solution)
                if len(steps) < 2:  # Skip too-short solutions
                    continue
                
                # Score each step with verifier
                step_scores, correctness = self.score_solution_with_verifier(
                    question, solution, ground_truth_answer
                )
                
                # Ensure alignment
                if len(step_scores) != len(steps):
                    step_scores = step_scores[:len(steps)] + [0] * (len(steps) - len(step_scores))
                
                is_correct = (correctness == "correct")
                
                # Compute rewards
                step_rewards = self.compute_reward(steps, step_scores, is_correct)
                
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


class PRMTrainer:
    """Train Process Reward Model"""
    
    def __init__(self, config: PRMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_prm_model(self):
        """Initialize PRM model (same architecture as SFT, new LoRA)"""
        print("Initializing PRM model...")
        
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
        
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        base_model = prepare_model_for_kbit_training(base_model)
        
        # LoRA config for PRM
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(base_model, lora_config)
        print("✓ PRM model initialized")
        
    def prepare_prm_training_data(self, dataset: Dataset) -> Dataset:
        """
        Convert PRM dataset to training format
        
        For each sample, create training examples:
        Input: question + partial_solution_up_to_step_i
        Target: reward_for_step_i (encoded as special tokens)
        """
        training_samples = []
        
        for sample in tqdm(dataset, desc="Preparing training data"):
            question = sample['question']
            steps = sample['steps']
            rewards = sample['step_rewards']
            
            # Create incremental examples
            for i in range(len(steps)):
                partial_solution = "\n".join(steps[:i+1])
                reward = rewards[i]
                
                # Format as instruction-following
                input_text = (
                    f"### Question:\n{question}\n\n"
                    f"### Partial Solution:\n{partial_solution}\n\n"
                    f"### Evaluate this step:\n"
                )
                
                # Encode reward as text (for language modeling)
                if reward > 0:
                    target_text = "CORRECT (+1)"
                else:
                    target_text = "INCORRECT (-1)"
                
                training_samples.append({
                    'input_text': input_text,
                    'target_text': target_text,
                    'reward_value': float(reward)
                })
        
        return Dataset.from_list(training_samples)
    
    def tokenize_function(self, examples):
        """Tokenize for PRM training"""
        inputs = [ex for ex in examples['input_text']]
        targets = [ex for ex in examples['target_text']]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            padding=False,
            truncation=True,
            max_length=1024
        )
        
        # Tokenize targets
        target_encodings = self.tokenizer(
            targets,
            padding=False,
            truncation=True,
            max_length=32
        )
        
        # Combine: input + target
        input_ids = []
        attention_masks = []
        labels = []
        
        for i in range(len(inputs)):
            combined_ids = model_inputs['input_ids'][i] + target_encodings['input_ids'][i][1:]
            input_ids.append(combined_ids)
            attention_masks.append([1] * len(combined_ids))
            
            # Mask input part, only train on target
            mask_len = len(model_inputs['input_ids'][i])
            labels.append([-100] * mask_len + target_encodings['input_ids'][i][1:])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }
    
    def train(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Train PRM model"""
        print("\n" + "="*70)
        print("TRAINING PRM MODEL")
        print("="*70)
        
        # Prepare data
        train_data = self.prepare_prm_training_data(train_dataset)
        eval_data = self.prepare_prm_training_data(eval_dataset)
        
        print(f"Training samples: {len(train_data)}")
        print(f"Eval samples: {len(eval_data)}")
        
        # Tokenize
        train_tokenized = train_data.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_data.column_names,
            desc="Tokenizing train"
        )
        
        eval_tokenized = eval_data.map(
            self.tokenize_function,
            batched=True,
            remove_columns=eval_data.column_names,
            desc="Tokenizing eval"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Training arguments
        output_dir = f"math_tutor_model/math_prm_adapter/method{self.config.method}_{self.config.reward_type}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=f"prm-method{self.config.method}-{self.config.reward_type}",
            
            optim="paged_adamw_8bit",
            
            num_train_epochs=3,
            learning_rate=2e-4,
            weight_decay=0.01,
            
            logging_dir="./logs",
            logging_strategy="steps",
            logging_steps=50,
            logging_first_step=True,
            
            eval_strategy="steps",
            eval_steps=500,
            eval_on_start=True,
            
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            
            fp16=False,
            bf16=True,
            bf16_full_eval=True,
            
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            gradient_checkpointing=False,
            
            seed=42,
            data_seed=42,
            
            group_by_length=True,
            dataloader_num_workers=4,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        print("\nStarting training...")
        train_result = trainer.train()
        
        # Save
        final_path = f"{output_dir}/final_checkpoint"
        trainer.save_model(final_path)
        self.tokenizer.save_pretrained(final_path)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        print(f"\n✓ Training complete! Model saved to {final_path}")
        print(f"Training time: {train_result.metrics['train_runtime']/3600:.2f} hours")


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
        generator = PRMDataGenerator(config)
        prm_dataset = generator.generate_prm_dataset()
        
        # Save dataset
        prm_dataset.save_to_disk(dataset_path)
        print(f"✓ PRM dataset saved to {dataset_path}")
    else:
        print(f"Loading existing dataset from {dataset_path}...")
        prm_dataset = Dataset.load_from_disk(dataset_path)
    
    # Split train/eval
    split = prm_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    eval_dataset = split['test']
    
    print(f"\nTrain size: {len(train_dataset)}")
    print(f"Eval size: {len(eval_dataset)}")
    
    # Train PRM
    prm_trainer = PRMTrainer(config)
    prm_trainer.load_prm_model()
    prm_trainer.train(train_dataset, eval_dataset)
    
    print("\n" + "="*70)
    print("PRM TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
