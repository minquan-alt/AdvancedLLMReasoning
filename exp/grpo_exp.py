import os
import sys
import json
import random
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import torch.nn.functional as F
import numpy as np

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning, module='peft')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')

sys.path.append('/home/guest/AdvancedLLMReasoning/')
from utils.prompt import SYSTEM_PROMPT_V3
from utils.inference_utils import clean_text

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_orm_dataset(orm_checkpoint_path, max_samples=None):
    with open(orm_checkpoint_path, 'r', encoding='utf-8') as f:
        orm_data = json.load(f)
    
    if max_samples:
        orm_data = orm_data[:max_samples]
    
    training_samples = []
    
    for item in orm_data:
        question = item.get('question', '')
        solutions = item.get('solutions', [])
        
        if not question or not solutions:
            continue
        
        data = [(s.get('solution', ''), 1.0 if s.get('label') == "[CORRECT]" else 0.0) 
                for s in solutions if s.get('solution')]
        
        if not data:
            continue
        mean_reward = np.mean([r for _, r in data])
        
        for solution, reward in data:
            training_samples.append({
                'question': clean_text(question),
                'solution': solution,
                'reward': reward,
                'advantage': reward - mean_reward
            })
    
    return training_samples


def grpo_training_step(model, ref_model, tokenizer, batch_samples, max_length, beta=0.1):
    model.train()
    
    texts = []
    advantages = []
    rewards = []
    prompt_lengths = []
    
    for sample in batch_samples:
        if abs(sample['advantage']) < 1e-6:
            continue
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V3},
            {"role": "user", "content": clean_text(sample['question'])},
            {"role": "assistant", "content": clean_text(sample['solution'])},
        ]
        
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        texts.append(full_text)
        advantages.append(sample['advantage'])
        rewards.append(sample['reward'])
        prompt_lengths.append(len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]))
    
    if not texts:
        return torch.tensor(0.0), {
            "loss": 0.0,
            "mean_reward": np.mean([s['reward'] for s in batch_samples]),
            "num_correct": sum(1 for s in batch_samples if s['reward'] > 0.5),
            "total_samples": len(batch_samples)
        }
    
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length + 256
    ).to(model.device)
    
    # forward pass - reference model (frozen)
    with torch.no_grad():
        ref_outputs = ref_model(**inputs)
        ref_logprobs = F.log_softmax(ref_outputs.logits, dim=-1)
    
    # forward pass - trainable model
    outputs = model(**inputs)
    logprobs = F.log_softmax(outputs.logits, dim=-1)
    
    # compute batch loss
    token_ids = inputs["input_ids"][:, 1:]
    logprobs_selected = logprobs[:, :-1, :].gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
    ref_logprobs_selected = ref_logprobs[:, :-1, :].gather(2, token_ids.unsqueeze(-1)).squeeze(-1)
    
    # create masks
    batch_size = len(texts)
    seq_len = logprobs_selected.size(1)
    mask = torch.zeros(batch_size, seq_len, device=model.device)
    
    attention_mask = inputs["attention_mask"][:, 1:]  
    
    for i, prompt_len in enumerate(prompt_lengths):
        if prompt_len > 0 and prompt_len <= len(attention_mask[i]):
            mask[i, prompt_len-1:] = attention_mask[i, prompt_len-1:]
    
    kl_div = (logprobs_selected - ref_logprobs_selected) * mask
    policy_loss = -logprobs_selected * mask
    
    advantages_tensor = torch.tensor(advantages, device=model.device).unsqueeze(1)
    sample_losses = (advantages_tensor * (policy_loss + beta * kl_div)).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    
    total_loss = sample_losses.mean()
    
    return total_loss, {
        "loss": total_loss.item(),
        "mean_reward": np.mean(rewards),
        "num_correct": sum(1 for r in rewards if r > 0.5),
        "total_samples": len(batch_samples)
    }


def load_sft_model(sft_model_path, base_model_id="meta-llama/Llama-3.2-1B"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print("✓ Base Model loaded")
    
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    tokenizer.padding_side = "left"
    
    if not tokenizer.chat_template:
        tokenizer.chat_template = """{{ bos_token }}
{% for message in messages -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>{{ message['content'] | trim }}
<|eot_id|>
{%- endfor %}
{% if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{%- endif %}
"""
    
    try:
        sft_model = PeftModel.from_pretrained(base_model, sft_model_path)
        print(f"SFT adapter loaded from: {sft_model_path}")
    except Exception as e:
        print(f"Error loading adapter: {e}")
        raise
    
    #  reference model (frozen)
    ref_model = PeftModel.from_pretrained(base_model, sft_model_path)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("Reference model created (frozen)")
    
    # prepare trainable model
    model = sft_model.merge_and_unload()
    model = prepare_model_for_kbit_training(model)
    
    # add new LoRA for GRPO
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("✓ New LoRA adapter added for GRPO")
    
    return model, ref_model, tokenizer


def main():
    set_seed(42)
    
    print("GRPO Training")
    
    # config
    SFT_MODEL = "/home/guest/AdvancedLLMReasoning/math_tutor_model/math_sft_adapter/v3/final_checkpoint"
    ORM_CHECKPOINT = "/home/guest/AdvancedLLMReasoning/data/checkpoints/gsm8k_orm_checkpoint.json"
    OUTPUT_DIR = "math_tutor_model/math_grpo_adapter/gsm8k_v1"
    
    MAX_LENGTH = 512
    NUM_EPOCHS = 3
    BATCH_SIZE = 2 
    LEARNING_RATE = 5e-7
    BETA = 0.1
    SAVE_STEPS = 100
    LOG_STEPS = 10
    MAX_SAMPLES = None
    
    training_samples = load_orm_dataset(ORM_CHECKPOINT, MAX_SAMPLES)
    print(f"Loaded {len(training_samples)} training samples\n")
    
    model, ref_model, tokenizer = load_sft_model(SFT_MODEL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    global_step = 0
    best_reward = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        random.shuffle(training_samples)
        
        epoch_metrics = {
            "loss": [],
            "reward": [],
            "correct": []
        }
        
        num_batches = (len(training_samples) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(training_samples))
            batch_samples = training_samples[start_idx:end_idx]
            
            loss, metrics = grpo_training_step(
                model, ref_model, tokenizer,
                batch_samples, MAX_LENGTH, BETA
            )
            
            if isinstance(loss, torch.Tensor) and loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            epoch_metrics["loss"].append(metrics["loss"])
            epoch_metrics["reward"].append(metrics["mean_reward"])
            epoch_metrics["correct"].append(metrics["num_correct"])
            
            global_step += 1
            
            if global_step % LOG_STEPS == 0:
                avg_loss = np.mean(epoch_metrics["loss"][-LOG_STEPS:])
                avg_reward = np.mean(epoch_metrics["reward"][-LOG_STEPS:])
                avg_correct = np.mean(epoch_metrics["correct"][-LOG_STEPS:])
                
                print(f"Step {global_step} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Reward: {avg_reward:.3f} | "
                      f"Correct: {avg_correct:.1f}/{BATCH_SIZE}")
            
            if global_step % SAVE_STEPS == 0:
                checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"\n✓ Checkpoint saved: {checkpoint_dir}\n")
        
        epoch_loss = np.mean(epoch_metrics["loss"])
        epoch_reward = np.mean(epoch_metrics["reward"])
        epoch_correct = np.sum(epoch_metrics["correct"])
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Average Loss: {epoch_loss:.4f}")
        print(f"  Average Reward: {epoch_reward:.3f}")
        print(f"  Total Correct: {epoch_correct}/{len(training_samples)}")
        print(f"  Accuracy: {epoch_correct/len(training_samples)*100:.1f}%")
        print(f"{'='*70}\n")
        
        # save best model
        if epoch_reward > best_reward:
            best_reward = epoch_reward
            best_checkpoint = os.path.join(OUTPUT_DIR, "best_checkpoint")
            model.save_pretrained(best_checkpoint)
            tokenizer.save_pretrained(best_checkpoint)
            print(f"Best model saved (reward: {best_reward:.3f})\n")
    
    # save final model
    final_checkpoint = os.path.join(OUTPUT_DIR, "final_checkpoint")
    model.save_pretrained(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)
    
    print(f"\n{'='*70}")
    print(f"GRPO Training Completed!")
    print(f"{'='*70}")
    print(f"Final model saved to: {final_checkpoint}")
    print(f"Best reward: {best_reward:.3f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
