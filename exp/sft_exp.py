from datasets import Dataset
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

from dotenv import load_dotenv
import os
load_dotenv()
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')
# login(HF_AUTH_TOKEN)

data_path = 'data/processed_data/256K/'
train_processed = Dataset.load_from_disk(data_path + 'train/')
dev_processed = Dataset.load_from_disk(data_path + 'dev/')
test_processed = Dataset.load_from_disk(data_path + 'test/')

# Cấu hình 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer và model
base_model = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model với quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.pad_token_id = tokenizer.pad_token_id

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# Chuẩn bị model cho training
model = prepare_model_for_kbit_training(model)

# Add LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model, 
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt",
    label_pad_token_id=-100
)

training_args = TrainingArguments(
    output_dir="math_tutor_model",
    run_name="llama32-1b-math-lora",
    
    optim="paged_adamw_8bit",
    
    num_train_epochs=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    
    # Logging
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=1000,
    eval_on_start=False,
    
    # Saving
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    
    fp16=False,
    bf16=True,
    bf16_full_eval=True,
    
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 4,
    gradient_checkpointing = False,
    
    seed=42,
    data_seed=42,
    
    group_by_length=True,
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_processed,
    eval_dataset=dev_processed,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# print("Bắt đầu training...")
# train_result = trainer.train()
# metrics = train_result.metrics
# print(f"Training xong! Tổng thời gian: {metrics['train_runtime']} giây ({metrics['train_runtime']/3600:.2f} giờ)")
# trainer.save_metrics("train", metrics)
# trainer.save_state()

train_result = trainer.train(resume_from_checkpoint=True)

print(f"Log đã được lưu tại: {training_args.output_dir}")