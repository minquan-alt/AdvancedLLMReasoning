from datasets import load_dataset, Dataset
from collections import defaultdict, deque
from tqdm import tqdm
import random
import re
import torch
from data_utils.process_data import DataPreprocessing
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, hf_dataset):
        self.input_ids = [torch.tensor(x['input_ids']) for x in hf_dataset]
        self.attention_mask = [torch.tensor(x['attention_mask']) for x in hf_dataset]
        self.labels = [torch.tensor(x['labels']) for x in hf_dataset]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def prepare_dataset(data_path, tokenizer, batch_size=128, train_ratio=.8, dev_ratio=.1):
    ds = load_from_disk(data_path)
    dp = DataPreprocessing(dataset=ds, tokenizer=tokenizer, train_ratio=train_ratio, dev_ratio=dev_ratio)
    train_processed, dev_processed, test_processed = dp.preprocess()
    # Return torch Dataset objects (not DataLoaders) so they can be used
    # either directly with Trainer or with DataLoader if manual training is desired.
    train_dataset = CustomDataset(train_processed)
    dev_dataset = CustomDataset(dev_processed)
    test_dataset = CustomDataset(test_processed)
    return train_dataset, dev_dataset, test_dataset