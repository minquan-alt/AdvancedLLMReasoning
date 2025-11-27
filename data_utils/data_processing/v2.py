from datasets import load_dataset, Dataset
from collections import defaultdict, deque
from tqdm import tqdm
import random
import re
from transformers import AutoTokenizer

class DataPreprocessing:
    '''
    Version 2: Instruction Tuning Format
    - Tokenize thủ công
    - Masking (Question + Instruction)
    - Truncation (No Padding)
    '''
    def __init__(self, dataset, tokenizer, train_ratio=0.99, dev_ratio=0.005):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        
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
        Xử lý từng mẫu: Masking (Question + Instruction), giữ lại Solution.
        '''
        eos_id = self.tokenizer.eos_token_id
        
        q_raw = example['question']
        a_raw = example['generated_solution']
        
        # Thêm Instruction để định hướng model
        instruction = (
            "Solve the problem step by step. You can use Python code if needed.\n"
            "If you write code, wrap it inside <llm-code> ... </llm-code>.\n"
            "Output ONLY the final number inside \\boxed{}."
        )

        question_text = (
            f"### Question:\n{self.clean_text(q_raw)}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Solution:\n"
        )
        
        answer_text = f"{self.clean_text(a_raw)}"
        
        question_encoded = self.tokenizer(question_text, padding=False, truncation=False, add_special_tokens=True)
        answer_encoded = self.tokenizer(answer_text, padding=False, truncation=False, add_special_tokens=True)
        
        input_ids = question_encoded['input_ids'] + answer_encoded['input_ids'][1:] + [eos_id]
        
        mask_len = len(question_encoded['input_ids'])
        labels = [-100] * mask_len + answer_encoded['input_ids'][1:] + [eos_id]
        
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            input_ids[-1] = eos_id
            labels[-1] = eos_id
        
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
    try:
        subset_ds = Dataset.load_from_disk("data/subset_openmathinstruct_1/256K")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy dataset gốc")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
        
    processor = DataPreprocessing(subset_ds, tokenizer)
    
    train_processed, dev_processed, test_processed = processor.preprocess(max_length=1024)
    
    output_path = 'data/processed_data_v2/256K/'
    train_processed.save_to_disk(output_path + 'train/')
    dev_processed.save_to_disk(output_path + 'dev/')
    test_processed.save_to_disk(output_path + 'test/')
    
    print("\n=== KẾT QUẢ XỬ LÝ V2 (INSTRUCTION TUNING) ===")
    print(f"Train samples: {len(train_processed)}")
    print(f"Output Path:   {output_path}")

if __name__ == "__main__":
    process_and_save_data()