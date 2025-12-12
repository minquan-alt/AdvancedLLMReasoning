"""
PRM Model Trainer
"""

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

from .config import PRMConfig


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
