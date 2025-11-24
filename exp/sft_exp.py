import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings with multiple workers

import gc
import torch
from data_utils.prepare_data import prepare_dataset
from data_utils.process_data import DataPreprocessing
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

if __name__ == "__main__":

    data_path = 'D:/Learning/cs431/AdvancedLLMReasoning/data/subset_openmathinstruct_1/256K/processed_datasets'

    # base model
    print("Loading base model...")
    base_model = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
    )

    print("Base model loaded.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # lora
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    print("LoRA configuration applied.")

    # prepare data
    print("Preparing dataset...")
    train_dataset = torch.load(os.path.join(data_path, "train_dataset.pt"), weights_only=False)
    dev_dataset = torch.load(os.path.join(data_path, "dev_dataset.pt"), weights_only=False)
    print("Dataset prepared.")

    # training_args = TrainingArguments(
    #     output_dir="./math_tutor_model",
    #     num_train_epochs=3,
    #     learning_rate=2e-5,
    #     weight_decay=0.01,
    #     logging_dir="./logs",
    #     logging_steps=100,
    #     eval_strategy="steps",
    #     eval_steps=500,
    #     save_strategy="steps",
    #     save_steps=500,
    #     warmup_steps=500,
    #     per_device_train_batch_size=1, 
    #     gradient_accumulation_steps=8,
    #     fp16=True,
    # )

    training_args = TrainingArguments(
        output_dir="./output/sft_model",
        num_train_epochs=3,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=200,
        fp16=True,
        optim="adamw_torch_fused", 
        gradient_checkpointing=True,

        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer
    )

    gc.collect()
    torch.cuda.empty_cache()
    print("Starting training...")
    trainer.train()

