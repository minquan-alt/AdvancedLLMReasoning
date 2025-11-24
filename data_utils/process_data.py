from datasets import load_dataset, Dataset
from collections import defaultdict, deque
from tqdm import tqdm
import random
import re

class DataPreprocessing:
    '''Tiền xử lý dữ liệu gồm làm sạch dữ liệu, tách dữ liệu ra train/dev/test, và tokenize dữ liệu'''
    
    def __init__(self, dataset, tokenizer, train_ratio=0.99, dev_ratio=0.005):
        '''
        Khởi tạo với dataset và tokenizer
        Args:
            dataset: Dataset Huggingface
            tokenizer: Tokenizer từ transformers
            train_ratio: Tỉ lệ tập train (mặc định 0.8)
            dev_ratio: Tỉ lệ tập dev (mặc định 0.1)
        '''
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        text = text.strip()
        text = re.sub(r'[ \t]+', ' ', text)  # Chỉ xóa space/tab thừa
        text = re.sub(r'\n{3,}', '\n\n', text)  # Giữ tối đa 2 newlines
        return text
    
    def prepare_input(self, question, solution):
        '''
        Chuẩn bị input cho mô hình
        Args:
            question: Câu hỏi
            solution: Lời giải
        Returns:
            string đã được format theo mẫu cho mô hình
        '''
        question = self.clean_text(question)
        solution = self.clean_text(solution)
        return f"### Question:\n{question}\n\n### Solution:\n{solution}"
    
    def tokenize_data(self, text, max_length=512):
        '''
        Tokenize dữ liệu text
        Args:
            text: string cần tokenize
            max_length: độ dài tối đa sau khi tokenize
        Returns:
            dict chứa input_ids và attention_mask
        '''
        return self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
    
    def split_data(self):
        '''
        Chia dữ liệu thành các tập train/dev/test
        Returns:
            tuple (train_data, dev_data, test_data)
        '''
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
    
    def process_example(self, example, max_length=512):
        '''
        Xử lý một mẫu dữ liệu
        Args:
            example: Một mẫu từ dataset
            max_length: độ dài tối đa cho tokenize
        Returns:
            dict chứa input_ids, attention_mask và labels
        '''
        text = self.prepare_input(
            example['question'],
            example['generated_solution']
        )
        tokenized = self.tokenize_data(text, max_length)
        
        # Thêm labels cho causal LM
        result = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].copy()
        }
        return result
    
    def preprocess(self, max_length=512):
        '''
        Tiền xử lý toàn bộ dataset
        Args:
            max_length: độ dài tối đa cho tokenize
        Returns:
            tuple (train_data, dev_data, test_data) đã được tiền xử lý
        '''
        # Chia dữ liệu
        train_data, dev_data, test_data = self.split_data()
        
        # Áp dụng xử lý cho từng tập
        train_processed = train_data.map(lambda x: self.process_example(x, max_length))
        dev_processed = dev_data.map(lambda x: self.process_example(x, max_length))
        test_processed = test_data.map(lambda x: self.process_example(x, max_length))
        
        return train_processed, dev_processed, test_processed

def process_and_save_data():
    '''
    Tiền xử lý dữ liệu từ dataset và lưu vào đĩa
    Args:
        dataset_path: Đường dẫn tới dataset Huggingface
        tokenizer: Tokenizer từ transformers
        output_path: Đường dẫn lưu dữ liệu đã xử lý
        max_length: độ dài tối đa cho tokenize
    '''
    subset_ds = Dataset.load_from_disk("data/subset_openmathinstruct_1/256K")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    processor = DataPreprocessing(subset_ds, tokenizer)
    train_processed, dev_processed, test_processed = processor.preprocess(1024)
    
    output_path = 'data/processed_data/256K/'
    train_processed.save_to_disk(output_path + 'train/')
    dev_processed.save_to_disk(output_path + 'dev/')
    test_processed.save_to_disk(output_path + 'test/')

if __name__ == "__main__":
    process_and_save_data()