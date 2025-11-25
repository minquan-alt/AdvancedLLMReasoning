from datasets import load_dataset, Dataset
from collections import defaultdict, deque
from tqdm import tqdm
import random
import re

class DataPreprocessing:
    '''Tiền xử lý dữ liệu gồm làm sạch dữ liệu, tách dữ liệu ra train/dev/test, và tokenize dữ liệu'''
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
    
    def prepare_input(self, question, solution):
        '''
        Thêm instruction để mô hình biết phải làm gì
        '''
        question = self.clean_text(question)
        solution = self.clean_text(solution)
        
        return (
            f"### Question:\n{question}\n\n"
            f"### Instruction:\n"
            f"Solve the problem step by step. Calculate the final value carefully. "
            f"Output ONLY the final number inside \\boxed{{}}.\n\n"
            f"### Solution:\n{solution}"
        )
    
    def tokenize_data(self, text, max_length=1024):
        return self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
    
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
    
    def process_example(self, example, max_length=2048):
        '''
        Chỉnh sửa label - chỉ học phần Solution
        '''
        question = example['question']
        solution = example['generated_solution']
        
        # tokenize câu hỏi + instruction (phần không cần học)
        question_text = (
            f"### Question:\n{self.clean_text(question)}\n\n"
            f"### Instruction:\n"
            f"Solve the problem step by step. Calculate the final value carefully. "
            f"Output ONLY the final number inside \\boxed{{}}.\n\n"
            f"### Solution:\n"
        )
        
        # tokenize phần solution (phần cần học)
        solution_text = self.clean_text(solution)
        
        # tokenize riêng từng phần
        question_tokens = self.tokenizer(
            question_text,
            truncation=False,
            padding=False,
            return_tensors=None
        )
        
        solution_tokens = self.tokenizer(
            solution_text,
            truncation=False,
            padding=False,
            return_tensors=None
        )
        
        input_ids = question_tokens['input_ids'] + solution_tokens['input_ids']
        attention_mask = question_tokens['attention_mask'] + solution_tokens['attention_mask']
        
        labels = [-100] * len(question_tokens['input_ids']) + solution_tokens['input_ids']
        
        # Truncate nếu quá dài
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def preprocess(self, max_length=1024):
        train_data, dev_data, test_data = self.split_data()
        
        train_processed = train_data.map(
            lambda x: self.process_example(x, max_length),
            desc="Processing train"
        )
        dev_processed = dev_data.map(
            lambda x: self.process_example(x, max_length),
            desc="Processing dev"
        )
        test_processed = test_data.map(
            lambda x: self.process_example(x, max_length),
            desc="Processing test"
        )
        
        return train_processed, dev_processed, test_processed

def process_and_save_data():
    subset_ds = Dataset.load_from_disk("data/subset_openmathinstruct_1/256K")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    processor = DataPreprocessing(subset_ds, tokenizer)
    train_processed, dev_processed, test_processed = processor.preprocess(max_length=1024)
    
    output_path = 'data/processed_data_v1/256K/'
    train_processed.save_to_disk(output_path + 'train/')
    dev_processed.save_to_disk(output_path + 'dev/')
    test_processed.save_to_disk(output_path + 'test/')
    
    print("Đã xử lý và lưu dữ liệu thành công!")
    print(f"Train: {len(train_processed)} samples")
    print(f"Dev: {len(dev_processed)} samples")
    print(f"Test: {len(test_processed)} samples")

if __name__ == "__main__":
    process_and_save_data()