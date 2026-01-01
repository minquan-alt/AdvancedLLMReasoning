from datasets import load_dataset, Dataset
from collections import defaultdict, deque
from tqdm import tqdm
import random
import re
from transformers import AutoTokenizer

class DataPreprocessing:
    '''
    Version 3: Llama-3.2 Chat Template Format
    - Sử dụng chat template chuẩn của Llama 3
    - Masking tự động dựa trên độ dài prompt
    '''
    def __init__(self, dataset, tokenizer, train_ratio=0.99, dev_ratio=0.005):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        
        # System prompt từ notebook
        self.system_prompt = (
            "You are a math reasoning assistant.\n"
            "Solve the problem step by step.\n"
            "You can use Python code if needed.\n"
            "If you write code, put it inside a Python code block:\n"
            "```python\n"
            "...\n"
            "```\n"
            "Output ONLY the final number inside \\boxed{}."
        )
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.strip()
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    
    def split_data(self):
        total_size = len(self.dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        
        train_size = int(total_size * self.train_ratio)
        dev_size = int(total_size * self.dev_ratio)
        
        train_indices = indices[:train_size]
        dev_indices = indices[train_size:train_size + dev_size]
        test_indices = indices[train_size + dev_size:]
        
        return (
            self.dataset.select(train_indices),
            self.dataset.select(dev_indices),
            self.dataset.select(test_indices)
        )
    
    def process_example(self, example, max_length=1024):
        '''
        Xử lý từng mẫu sử dụng chat template.
        '''
        q_raw = example['question']
        a_raw = example['generated_solution']
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.clean_text(q_raw)},
            {"role": "assistant", "content": self.clean_text(a_raw)},
        ]
        
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        prompt_messages = messages[:-1]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True # Thêm header assistant vào cuối prompt
        )
        
        full_encoded = self.tokenizer(
            full_text, 
            padding=False, 
            truncation=True, 
            max_length=max_length, 
            add_special_tokens=False
        )
        
        prompt_encoded = self.tokenizer(
            prompt_text,
            padding=False,
            truncation=False, # Không truncate prompt để đo độ dài chính xác
            add_special_tokens=False
        )
        
        input_ids = full_encoded['input_ids']
        labels = input_ids.copy()
        
        prompt_len = len(prompt_encoded['input_ids'])
        if prompt_len < len(input_ids):
            for i in range(prompt_len):
                labels[i] = -100
        else:
            labels = [-100] * len(input_ids)
            
        attention_mask = [1] * len(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def preprocess(self, max_length=1024):
        train_data, dev_data, test_data = self.split_data()
        
        cols_to_remove = train_data.column_names 
        
        train_processed = train_data.map(
            lambda x: self.process_example(x, max_length),
            remove_columns=cols_to_remove, 
            desc="Processing train"
        )
        dev_processed = dev_data.map(
            lambda x: self.process_example(x, max_length),
            remove_columns=cols_to_remove,
            desc="Processing dev"
        )
        test_processed = test_data.map(
            lambda x: self.process_example(x, max_length),
            remove_columns=cols_to_remove,
            desc="Processing test"
        )
        
        return train_processed, dev_processed, test_processed

def process_and_save_data():
    subset_ds = Dataset.load_from_disk("data/subset_openmathinstruct_1_v2/256K")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = """{{ bos_token }}
{% for message in messages -%}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>{{ message['content'] | trim }}
<|eot_id|>
{%- endfor %}
{% if add_generation_prompt -%}
<|start_header_id|>assistant<|end_header_id|>
{%- endif %}
"""
    processor = DataPreprocessing(subset_ds, tokenizer)
    train_processed, dev_processed, test_processed = processor.preprocess(max_length=1024)
    
    output_path = 'data/processed_data_v3/256K/'
    train_processed.save_to_disk(output_path + 'train/')
    dev_processed.save_to_disk(output_path + 'dev/')
    test_processed.save_to_disk(output_path + 'test/')
    
    print("\n=== KẾT QUẢ XỬ LÝ V3 (LLAMA 3 CHAT TEMPLATE) ===")
    print(f"Train samples: {len(train_processed)}")
    print(f"Output Path:   {output_path}")

if __name__ == "__main__":
    process_and_save_data()
