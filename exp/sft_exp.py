import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
from data_utils.prepare_data import prepare_dataset
from data_utils.process_data import DataPreprocessing
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

data_path = 'data/subset_openmathinstruct_1'
login('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')

# base model
base_model = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# lora
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1)
model = get_peft_model(model, lora_config)

# prepare data
train_dataset, dev_dataset, test_dataset = prepare_dataset(data_path, tokenizer, batch_size=1)

training_args = TrainingArguments(
    output_dir="./math_tutor_model",
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    warmup_steps=500,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
)

gc.collect()
torch.cuda.empty_cache()

trainer.train()

